# -*- coding: utf-8 -*-
import os, numpy as np, torch
from .losses import compute_loss

@torch.no_grad()
def _bin_metrics(preds, masks, eps=1e-7):
    dims = (0, 2, 3)
    inter = (preds * masks).sum(dims)
    dice = (2 * inter + eps) / (preds.sum(dims) + masks.sum(dims) + eps)
    iou  = (inter + eps) / ((preds + masks - preds * masks).sum(dims) + eps)
    return dice.mean().item(), iou.mean().item()

def train_one_epoch(model, loader, optimizer, scaler, device, cfg, loss_obj):
    model.train()
    run = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=cfg["amp"]):
            logits = model(x)
            loss = compute_loss(loss_obj, logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        run += loss.item() * x.size(0)
    return run / len(loader.dataset)

@torch.no_grad()
def validate(model, loader, device, cfg, loss_obj):
    model.eval()
    vloss, dices, ious = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.cuda.amp.autocast(enabled=cfg["amp"]):
            logits = model(x)
            loss = compute_loss(loss_obj, logits, y)
        vloss += loss.item() * x.size(0)

        if cfg["classes"] == 1:
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            d, i = _bin_metrics(preds.cpu(), y.cpu())
            dices.append(d); ious.append(i)
    vloss /= len(loader.dataset)
    md = float(np.mean(dices)) if dices else 0.0
    mi = float(np.mean(ious)) if ious else 0.0
    return vloss, md, mi

@torch.no_grad()
def save_samples(model, loader, device, out_dir, cfg, max_samples=6):
    import cv2, numpy as np
    os.makedirs(f"{out_dir}/samples", exist_ok=True)
    model.eval(); saved = 0
    for x, y in loader:
        x = x.to(device)
        with torch.amp.autocast("cuda", enabled=cfg.get("amp", False)):
            logits = model(x)
            if cfg["classes"] == 1:
                preds = (torch.sigmoid(logits) > 0.5).float()
            else:
                preds = torch.argmax(logits, 1, keepdim=True).float()
        for i in range(x.size(0)):
            if saved >= max_samples: return
            img = x[i, 0].detach().cpu().numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-6)
            img = (img * 255).astype("uint8")
            if cfg["classes"] == 1:
                gt  = y[i, 0].cpu().numpy().astype("uint8") * 255
                prd = preds[i, 0].cpu().numpy().astype("uint8") * 255
            else:
                gt  = (y[i].cpu().numpy().astype("uint8")) * 30
                prd = (preds[i, 0].cpu().numpy().astype("uint8")) * 30
            h, w = img.shape
            canvas = np.zeros((h, w*3), np.uint8)
            canvas[:, :w] = img
            canvas[:, w:2*w] = gt
            canvas[:, 2*w:] = prd
            cv2.imwrite(f"{out_dir}/samples/sample_{saved:03d}.png", canvas)
            saved += 1
