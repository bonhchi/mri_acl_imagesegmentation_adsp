# -*- coding: utf-8 -*-
import os, numpy as np, torch
from .losses import compute_loss

def train_one_epoch(model, loader, optimizer, scaler, device, cfg, loss_obj,
                    logger=None, global_step=0):
    model.train(); run=0.0
    for x,y in loader:
        x,y=x.to(device),y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=cfg.get("amp",False)):
            logits=model(x)
            loss=compute_loss(loss_obj,logits,y)
        if not torch.isfinite(loss):  # guard
            raise RuntimeError(f"Non-finite loss at step {global_step}: {loss.item()}")
        scaler.scale(loss).backward()
        # optional: clip để ổn định
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer); scaler.update()
        run += loss.item()*x.size(0)

        if logger is not None:
            logger.log_step(global_step=global_step, epoch=cfg.get("epoch",-1),
                            lr=float(optimizer.param_groups[0]["lr"]),
                            loss=float(loss.item()))
        global_step += 1
    return run/len(loader.dataset), global_step

@torch.no_grad()
def _bin_metrics(preds, masks, eps=1e-7):
    dims = (0, 2, 3)
    inter = (preds * masks).sum(dims)
    dice = (2 * inter + eps) / (preds.sum(dims) + masks.sum(dims) + eps)
    iou  = (inter + eps) / ((preds + masks - preds * masks).sum(dims) + eps)
    return dice.mean().item(), iou.mean().item()

@torch.no_grad()
def validate(model, loader, device, cfg, loss_obj):
    model.eval()
    vloss, dices, ious = 0.0, dices = [], ious = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast("cuda", enabled=cfg.get("amp",False)):
            logits = model(x)
            loss = compute_loss(loss_obj, logits, y)
        vloss += loss.item() * x.size(0)

        if cfg.get("classes",1)==1:
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            d, i = _bin_metrics(preds.cpu(), y.cpu())
            dices.append(d); ious.append(i)
    vloss /= len(loader.dataset)
    md = float(np.mean(dices)) if dices else 0.0
    mi = float(np.mean(ious)) if ious else 0.0
    return vloss, md, mi


# ---- save_samples v2: Input|GT|Pred|Overlay + AMP API mới ----
def _to_uint8(a): 
    a=(a-a.min())/(a.max()-a.min()+1e-6); return (a*255).astype("uint8")
def _overlay(gray, mask_u8, alpha=0.45):
    import cv2
    color=np.zeros((gray.shape[0],gray.shape[1],3),np.uint8); color[...,1]=mask_u8
    return cv2.addWeighted(cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR),1.0,color,alpha,0.0)
def _colorize_mc(mask):
    uniq=np.unique(mask); pal=[(0,0,0),(255,0,0),(0,180,0),(0,0,255),(255,140,0),
                               (180,0,180),(0,160,160),(200,200,0),(255,105,180),(128,64,0)]
    out=np.zeros((mask.shape[0],mask.shape[1],3),np.uint8)
    for i,c in enumerate(uniq): out[mask==c]=pal[i%len(pal)]
    return out


@torch.no_grad()
def save_samples(model, loader, device, out_dir, cfg, max_samples=6):
    import cv2
    os.makedirs(f"{out_dir}/samples", exist_ok=True)
    model.eval(); saved = 0
    for x, y in loader:
        x = x.to(device)
        with torch.amp.autocast("cuda", enabled=cfg.get("amp",False)):
            logits = model(x)
            if cfg.get("classes",1)==1:
                probs=torch.sigmoid(logits).cpu().numpy()      # (N,1,H,W)
                preds=(probs>0.5).astype(np.uint8)*255
                gt=y.cpu().numpy(); 
                gt = gt if gt.ndim==4 else gt[:,None,...]
                gt_u8=(gt>0.5).astype(np.uint8)*255
            else:
                sm=torch.softmax(logits,1).cpu().numpy()
                preds=np.argmax(sm,1).astype(np.uint8); gt=y.cpu().numpy()
        for i in range(x.size(0)):
            if saved >= max_samples: return
            img = x[i, 0].detach().cpu().numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-6)
            img = (img * 255).astype("uint8")
            if cfg.get("classes",1)==1:
                gt_i,pr_i=gt_u8[i,0],preds[i,0]; over=_overlay(img,pr_i)
                h,w=img.shape; grid=np.zeros((h,w*4,3),np.uint8)
                grid[:,0:w]   = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
                grid[:,w:2*w] = cv2.cvtColor(gt_i,cv2.COLOR_GRAY2BGR)
                grid[:,2*w:3*w]=cv2.cvtColor(pr_i,cv2.COLOR_GRAY2BGR)
                grid[:,3*w:4*w]=over
                cv2.imwrite(f"{out_dir}/samples/sample_{sid:04d}.png", grid)
                if save_probs: np.save(f"{out_dir}/samples/sample_{sid:04d}_prob.npy", probs[i,0])
            else:
                gt_rgb=_colorize_mc(gt[i]); pr_rgb=_colorize_mc(preds[i]); h,w=img.shape
                grid=np.zeros((h,w*4,3),np.uint8)
                grid[:,0:w]   = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
                grid[:,w:2*w] = gt_rgb; grid[:,2*w:3*w]=pr_rgb
                grid[:,3*w:4*w]=cv2.addWeighted(cv2.cvtColor(img,cv2.COLOR_GRAY2BGR),1.0,pr_rgb,0.45,0.0)
                cv2.imwrite(f"{out_dir}/samples/sample_{sid:04d}.png", grid)
            saved+=1; sid+=1
