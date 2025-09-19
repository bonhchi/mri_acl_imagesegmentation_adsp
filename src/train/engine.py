# -*- coding: utf-8 -*-
"""
Engine: train_one_epoch / validate / save_samples
- AMP API mới: torch.amp.autocast("cuda", enabled=...)
- Có guard non-finite loss + clip grad
- save_samples: grid 4 cột (Input|GT|Pred|Overlay), hỗ trợ binary/multiclass
"""
from __future__ import annotations

import os
import numpy as np
import torch
from .losses import compute_loss


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    cfg: dict,
    loss_obj,
    logger=None,
    global_step: int = 0,
):
    """Train 1 epoch với AMP mới + logger từng step."""
    model.train()
    run = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=cfg.get("amp", False)):
            logits = model(x)
            loss = compute_loss(loss_obj, logits, y)

        # Guard numeric
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {global_step}: {loss.item()}")

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        run += float(loss.item()) * x.size(0)

        if logger is not None:
            logger.log_step(
                global_step=global_step,
                epoch=cfg.get("epoch", -1),
                lr=float(optimizer.param_groups[0]["lr"]),
                loss=float(loss.item()),
            )
        global_step += 1

    avg = run / len(loader.dataset)
    return avg, global_step


@torch.no_grad()
def _bin_metrics(preds: torch.Tensor, masks: torch.Tensor, eps: float = 1e-7):
    """Dice & IoU cho binary (tensor N,1,H,W)."""
    dims = (0, 2, 3)
    inter = (preds * masks).sum(dims)
    dice = (2 * inter + eps) / (preds.sum(dims) + masks.sum(dims) + eps)
    iou = (inter + eps) / ((preds + masks - preds * masks).sum(dims) + eps)
    return dice.mean().item(), iou.mean().item()


@torch.no_grad()
def validate(model, loader, device, cfg: dict, loss_obj):
    """Validate: trả (val_loss, mean_dice, mean_iou)."""
    model.eval()
    vloss, dices, ious = 0.0, [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast("cuda", enabled=cfg.get("amp", False)):
            logits = model(x)
            loss = compute_loss(loss_obj, logits, y)
        vloss += float(loss.item()) * x.size(0)

        if cfg.get("classes", 1) == 1:
            preds = (torch.sigmoid(logits) > 0.5).float()
            d, i = _bin_metrics(preds.cpu(), y.cpu())
            dices.append(d)
            ious.append(i)

    vloss /= len(loader.dataset)
    md = float(np.mean(dices)) if dices else 0.0
    mi = float(np.mean(ious)) if ious else 0.0
    return vloss, md, mi


# ---------- Save samples ----------
def _to_uint8(a: np.ndarray) -> np.ndarray:
    a = (a - a.min()) / (a.max() - a.min() + 1e-6)
    return (a * 255).astype(np.uint8)


def _overlay(gray: np.ndarray, mask_u8: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    import cv2

    color = np.zeros((gray.shape[0], gray.shape[1], 3), np.uint8)
    color[..., 1] = mask_u8  # green
    return cv2.addWeighted(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 1.0, color, alpha, 0.0)


def _colorize_mc(mask: np.ndarray) -> np.ndarray:
    uniq = np.unique(mask)
    palette = [
        (0, 0, 0),
        (255, 0, 0),
        (0, 180, 0),
        (0, 0, 255),
        (255, 140, 0),
        (180, 0, 180),
        (0, 160, 160),
        (200, 200, 0),
        (255, 105, 180),
        (128, 64, 0),
    ]
    out = np.zeros((mask.shape[0], mask.shape[1], 3), np.uint8)
    for i, c in enumerate(uniq):
        out[mask == c] = palette[i % len(palette)]
    return out


@torch.no_grad()
def save_samples(
    model,
    loader,
    device,
    out_dir: str,
    cfg: dict,
    max_samples: int = 8,
    save_probs: bool = False,
):
    """
    Lưu grid 4 cột: Input | GT | Pred | Overlay (binary), hoặc tô màu (multiclass).
    """
    import cv2

    os.makedirs(f"{out_dir}/samples", exist_ok=True)
    model.eval()
    saved = 0
    sid = 0

    for x, y in loader:
        x = x.to(device)
        with torch.amp.autocast("cuda", enabled=cfg.get("amp", False)):
            logits = model(x)

        if cfg.get("classes", 1) == 1:
            probs = torch.sigmoid(logits).cpu().numpy()  # (N,1,H,W)
            preds = (probs > 0.5).astype(np.uint8) * 255
            gt = y.cpu().numpy()
            if gt.ndim == 3:  # (N,H,W) -> (N,1,H,W)
                gt = gt[:, None, ...]
            gt_u8 = (gt > 0.5).astype(np.uint8) * 255
        else:
            sm = torch.softmax(logits, dim=1).cpu().numpy()  # (N,C,H,W)
            preds = np.argmax(sm, axis=1).astype(np.uint8)   # (N,H,W)
            gt = y.cpu().numpy()

        for i in range(x.size(0)):
            if saved >= max_samples:
                return

            img = _to_uint8(x[i, 0].detach().cpu().numpy())

            if cfg.get("classes", 1) == 1:
                gt_i = gt_u8[i, 0]
                pr_i = preds[i, 0]
                over = _overlay(img, pr_i)

                h, w = img.shape
                grid = np.zeros((h, w * 4, 3), np.uint8)
                grid[:, 0:w] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                grid[:, w:2 * w] = cv2.cvtColor(gt_i, cv2.COLOR_GRAY2BGR)
                grid[:, 2 * w:3 * w] = cv2.cvtColor(pr_i, cv2.COLOR_GRAY2BGR)
                grid[:, 3 * w:4 * w] = over

                cv2.imwrite(f"{out_dir}/samples/sample_{sid:04d}.png", grid)
                if save_probs:
                    np.save(f"{out_dir}/samples/sample_{sid:04d}_prob.npy", probs[i, 0])

            else:
                gt_rgb = _colorize_mc(gt[i])
                pr_rgb = _colorize_mc(preds[i])

                h, w = img.shape
                grid = np.zeros((h, w * 4, 3), np.uint8)
                grid[:, 0:w] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                grid[:, w:2 * w] = gt_rgb
                grid[:, 2 * w:3 * w] = pr_rgb
                grid[:, 3 * w:4 * w] = cv2.addWeighted(
                    cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), 1.0, pr_rgb, 0.45, 0.0
                )

                cv2.imwrite(f"{out_dir}/samples/sample_{sid:04d}.png", grid)

            saved += 1
            sid += 1
