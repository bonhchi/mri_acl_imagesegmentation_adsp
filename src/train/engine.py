# src/train/engine.py
# -*- coding: utf-8 -*-
"""
Engine class: train_one_epoch / validate / save_samples (+ optional fit/test)
- AMP API mới: torch.amp.autocast("cuda", enabled=...)
- Guard non-finite loss + clip grad
- save_samples: grid 4 cột (Input|GT|Pred|Overlay), hỗ trợ binary/multiclass
"""
from __future__ import annotations

import os
import numpy as np
import torch
from typing import Optional, Dict, Any
from .losses import compute_loss


class Engine:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        cfg: Dict[str, Any],
        optimizer: torch.optim.Optimizer,
        loss_obj,
        logger=None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.cfg = cfg
        self.optimizer = optimizer
        self.loss_obj = loss_obj
        self.logger = logger
        self.scaler = scaler or torch.cuda.amp.GradScaler(enabled=cfg.get("amp", False))
        self.global_step = 0

        # optional
        self.max_grad_norm = float(cfg.get("max_grad_norm", 5.0))

    # ---------- private helpers ----------
    @staticmethod
    @torch.no_grad()
    def _bin_metrics(preds: torch.Tensor, masks: torch.Tensor, eps: float = 1e-7):
        """Dice & IoU cho binary (tensor N,1,H,W)."""
        dims = (0, 2, 3)
        inter = (preds * masks).sum(dims)
        dice = (2 * inter + eps) / (preds.sum(dims) + masks.sum(dims) + eps)
        iou = (inter + eps) / ((preds + masks - preds * masks).sum(dims) + eps)
        return dice.mean().item(), iou.mean().item()

    @staticmethod
    def _to_uint8(a: np.ndarray) -> np.ndarray:
        a = (a - a.min()) / (a.max() - a.min() + 1e-6)
        return (a * 255).astype(np.uint8)

    @staticmethod
    def _overlay(gray: np.ndarray, mask_u8: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        import cv2
        color = np.zeros((gray.shape[0], gray.shape[1], 3), np.uint8)
        color[..., 1] = mask_u8  # green
        return cv2.addWeighted(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 1.0, color, alpha, 0.0)

    @staticmethod
    def _colorize_mc(mask: np.ndarray) -> np.ndarray:
        uniq = np.unique(mask)
        palette = [
            (0, 0, 0), (255, 0, 0), (0, 180, 0), (0, 0, 255), (255, 140, 0),
            (180, 0, 180), (0, 160, 160), (200, 200, 0), (255, 105, 180), (128, 64, 0),
        ]
        out = np.zeros((mask.shape[0], mask.shape[1], 3), np.uint8)
        for i, c in enumerate(uniq):
            out[mask == c] = palette[i % len(palette)]
        return out

    # ---------- core methods ----------
    def train_one_epoch(self, loader) -> float:
        """Train 1 epoch với AMP + logging từng step. Trả avg loss."""
        self.model.train()
        run = 0.0
        amp_enabled = bool(self.cfg.get("amp", False))
        epoch = int(self.cfg.get("epoch", -1))

        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = self.model(x)
                loss = compute_loss(self.loss_obj, logits, y)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {self.global_step}: {loss.item()}")

            self.scaler.scale(loss).backward()
            if self.max_grad_norm and self.max_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            run += float(loss.item()) * x.size(0)

            if self.logger is not None and hasattr(self.logger, "log_step"):
                self.logger.log_step(
                    global_step=self.global_step,
                    epoch=epoch,
                    lr=float(self.optimizer.param_groups[0]["lr"]),
                    loss=float(loss.item()),
                )
            self.global_step += 1

        avg = run / len(loader.dataset)
        return avg

    @torch.no_grad()
    def validate(self, loader) -> tuple[float, float, float]:
        """Validate: trả (val_loss, mean_dice, mean_iou)."""
        self.model.eval()
        vloss, dices, ious = 0.0, [], []
        amp_enabled = bool(self.cfg.get("amp", False))
        classes = int(self.cfg.get("classes", 1))

        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = self.model(x)
                loss = compute_loss(self.loss_obj, logits, y)
            vloss += float(loss.item()) * x.size(0)

            if classes == 1:
                preds = (torch.sigmoid(logits) > 0.5).float()
                d, i = self._bin_metrics(preds.cpu(), y.cpu())
                dices.append(d); ious.append(i)

        vloss /= len(loader.dataset)
        md = float(np.mean(dices)) if dices else 0.0
        mi = float(np.mean(ious)) if ious else 0.0
        return vloss, md, mi

    @torch.no_grad()
    def save_samples(
        self,
        loader,
        out_dir: str,
        max_samples: int = 8,
        save_probs: bool = False,
    ):
        """
        Lưu grid 4 cột: Input | GT | Pred | Overlay (binary), hoặc tô màu (multiclass).
        """
        import cv2

        os.makedirs(f"{out_dir}/samples", exist_ok=True)
        self.model.eval()
        saved = 0
        sid = 0
        amp_enabled = bool(self.cfg.get("amp", False))
        classes = int(self.cfg.get("classes", 1))

        for x, y in loader:
            x = x.to(self.device)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = self.model(x)

            if classes == 1:
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

                img = self._to_uint8(x[i, 0].detach().cpu().numpy())

                if classes == 1:
                    gt_i = gt_u8[i, 0]
                    pr_i = preds[i, 0]
                    over = self._overlay(img, pr_i)

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
                    gt_rgb = self._colorize_mc(gt[i])
                    pr_rgb = self._colorize_mc(preds[i])

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

    # ---------- optional high-level wrappers ----------
    def fit(
        self,
        train_loader,
        val_loader=None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        out_dir: Optional[str] = None,
        epochs: Optional[int] = None,
        save_every: int = 0,
    ) -> Dict[str, Any]:
        """
        Vòng lặp train/val tối thiểu. Không cố thay code hiện có—chỉ là tiện ích.
        """
        best = {"dice": -1.0, "epoch": -1}
        out_dir = out_dir or self.cfg.get("out_dir", "outputs")
        epochs = int(epochs or self.cfg.get("epochs", 1))
        save_every = int(save_every)

        for ep in range(1, epochs + 1):
            self.cfg["epoch"] = ep
            tr_loss = self.train_one_epoch(train_loader)
            val_loss, val_dice, val_iou = (0.0, 0.0, 0.0)
            if val_loader is not None:
                val_loss, val_dice, val_iou = self.validate(val_loader)

            # step scheduler (nếu có)
            if scheduler is not None:
                # nếu là Plateau, dùng dice để step; còn lại step mỗi epoch
                if hasattr(scheduler, "step") and scheduler.__class__.__name__.lower().startswith("reduce"):
                    scheduler.step(val_dice)
                else:
                    scheduler.step()

            # log tổng hợp
            if self.logger is not None and hasattr(self.logger, "log_epoch"):
                self.logger.log_epoch(
                    epoch=ep, train_loss=float(tr_loss),
                    val_loss=float(val_loss), val_dice=float(val_dice), val_iou=float(val_iou),
                    lr=float(self.optimizer.param_groups[0]["lr"]),
                )

            # best
            if val_loader is not None and val_dice > best["dice"]:
                best.update({"dice": float(val_dice), "epoch": ep})
                if self.cfg.get("save_best", True):
                    os.makedirs(f"{out_dir}/checkpoints", exist_ok=True)
                    torch.save(self.model.state_dict(), f"{out_dir}/checkpoints/best.pt")

            # checkpoint theo chu kỳ
            if save_every and (ep % save_every == 0):
                os.makedirs(f"{out_dir}/checkpoints", exist_ok=True)
                torch.save(self.model.state_dict(), f"{out_dir}/checkpoints/epoch_{ep:03d}.pt")

        return {"best_dice": best["dice"], "best_epoch": best["epoch"]}

    @torch.no_grad()
    def test(self, loader, ckpt_path: Optional[str] = None) -> Dict[str, float]:
        if ckpt_path and os.path.isfile(ckpt_path):
            state = torch.load(ckpt_path, map_location=self.device)
            # chấp nhận cả state_dict full hoặc gói dict
            state_dict = state["model_state"] if isinstance(state, dict) and "model_state" in state else state
            self.model.load_state_dict(state_dict)

        self.model.eval()
        dices, ious = [], []
        amp_enabled = bool(self.cfg.get("amp", False))
        classes = int(self.cfg.get("classes", 1))

        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = self.model(x)

            if classes == 1:
                preds = (torch.sigmoid(logits) > 0.5).float()
                d, i = self._bin_metrics(preds.cpu(), y.cpu())
                dices.append(d); ious.append(i)

        return {
            "dice": float(np.mean(dices)) if dices else 0.0,
            "iou": float(np.mean(ious)) if ious else 0.0,
        }

