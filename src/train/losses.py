# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
try:
    import segmentation_models_pytorch as smp  # type: ignore
    _HAS_SMP = True
except ModuleNotFoundError:  # pragma: no cover - runtime path
    smp = None
    _HAS_SMP = False


def _binary_dice_loss(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    probs = torch.sigmoid(preds)
    dims = (0, 2, 3)
    inter = (probs * targets).sum(dims)
    den = probs.sum(dims) + targets.sum(dims)
    return 1 - (2 * inter + eps) / (den + eps)


class _SoftBCEWithLogitsLoss(nn.Module):
    """Fallback BCE loss when SMP is unavailable."""

    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss(logits, targets)


class _FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.float()
        pt = torch.where(targets == 1, probs, 1 - probs)
        with torch.no_grad():
            alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        loss = -alpha_t * torch.pow(1 - pt, self.gamma) * torch.log(pt.clamp(min=1e-7))
        return loss.mean()


class TverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, eps: float = 1e-7):
        super().__init__()
        self.a, self.b, self.eps = alpha, beta, eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        dims = (0, 2, 3)
        TP = (p * targets).sum(dims)
        FP = (p * (1 - targets)).sum(dims)
        FN = ((1 - p) * targets).sum(dims)
        tv = (TP + self.eps) / (TP + self.a * FP + self.b * FN + self.eps)
        return 1 - tv.mean()


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, gamma: float = 0.75):
        super().__init__()
        self.tv = TverskyLoss(alpha, beta)
        self.g = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        t = 1 - self.tv(logits, targets)  # t ~ dice-like
        return torch.pow(1 - t, self.g)


class LossManager(nn.Module):
    """
    Class tổng quản lý loss cho segmentation.
    - Binary:
        name: "dice_bce" (mặc định), "focal", "tversky", "focal_tversky"
    - Multiclass:
        name: "dice_ce", "ce"
    - Nếu là loss tổ hợp (dice_bce, dice_ce), bạn có thể chỉnh weights.

    Usage:
        lm = LossManager(classes=1, name="dice_bce", weights=(0.5, 0.5))
        loss = lm(logits, targets)

        # hoặc từ config:
        lm = LossManager.from_config({"classes":1, "name":"dice_bce", "weights":[0.5,0.5]})
    """

    def __init__(
        self,
        classes: int = 1,
        name: str = "dice_bce",
        weights: Optional[Tuple[float, float]] = None,
        # Binary-only params:
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        tversky_alpha: float = 0.7,
        tversky_beta: float = 0.3,
        tversky_gamma: float = 0.75,  # cho FocalTversky
    ):
        super().__init__()
        self.classes = int(classes)
        self.name = str(name).lower()
        self.weights = weights if weights is not None else (0.5, 0.5)
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.tversky_gamma = tversky_gamma

        self.criterion = self._build()

    @classmethod
    def from_config(cls, cfg: dict) -> "LossManager":
        return cls(
            classes=cfg.get("classes", 1),
            name=cfg.get("loss_name", cfg.get("loss", "dice_bce")),
            weights=tuple(cfg.get("loss_weights", (0.5, 0.5))) if cfg.get("loss_weights") else None,
            focal_alpha=cfg.get("focal_alpha", 0.25),
            focal_gamma=cfg.get("focal_gamma", 2.0),
            tversky_alpha=cfg.get("tversky_alpha", 0.7),
            tversky_beta=cfg.get("tversky_beta", 0.3),
            tversky_gamma=cfg.get("tversky_gamma", 0.75),
        )

    def _build(self) -> Union[nn.Module, nn.ModuleList]:
        if self.classes == 1:
            # Binary
            if self.name in ("dice_bce", "bce_dice", "dice+bce"):
                if _HAS_SMP:
                    dice_loss = smp.losses.DiceLoss(mode="binary")
                else:
                    class _DiceLoss(nn.Module):
                        def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
                            return _binary_dice_loss(logits, targets)
                    dice_loss = _DiceLoss()
                bce_loss = (
                    smp.losses.SoftBCEWithLogitsLoss() if _HAS_SMP else _SoftBCEWithLogitsLoss()
                )
                return nn.ModuleList([dice_loss, bce_loss])
            if self.name == "focal":
                if _HAS_SMP:
                    return smp.losses.FocalLoss(mode="binary", alpha=self.focal_alpha, gamma=self.focal_gamma)
                return _FocalLoss(alpha=self.focal_alpha, gamma=self.focal_gamma)
            if self.name == "tversky":
                return TverskyLoss(self.tversky_alpha, self.tversky_beta)
            if self.name in ("focal_tversky", "focal-tversky"):
                return FocalTverskyLoss(self.tversky_alpha, self.tversky_beta, self.tversky_gamma)
            raise ValueError(f"Unknown binary loss: {self.name}")
        else:
            # Multiclass
            if self.name in ("dice_ce", "dice+ce", "ce_dice"):
                if _HAS_SMP:
                    dice_loss = smp.losses.DiceLoss(mode="multiclass")
                else:
                    class _MulticlassDice(nn.Module):
                        def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
                            import torch.nn.functional as F
                            probs = torch.softmax(logits, dim=1)
                            targets = targets.long()
                            one_hot = F.one_hot(targets, num_classes=probs.shape[1])
                            one_hot = one_hot.permute(0, 3, 1, 2).float()
                            dims = (0, 2, 3)
                            inter = (probs * one_hot).sum(dim=dims)
                            den = probs.sum(dim=dims) + one_hot.sum(dim=dims)
                            eps = 1e-7
                            dice = (2 * inter + eps) / (den + eps)
                            return 1 - dice.mean()
                    dice_loss = _MulticlassDice()
                return nn.ModuleList([
                    dice_loss,
                    nn.CrossEntropyLoss(),
                ])
            if self.name in ("ce", "cross_entropy"):
                return nn.CrossEntropyLoss()
            raise ValueError(f"Unknown multiclass loss: {self.name}")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # tổ hợp (ModuleList) -> blend theo self.weights, ngược lại -> criterion(logits, targets)
        if isinstance(self.criterion, nn.ModuleList):
            w0, w1 = float(self.weights[0]), float(self.weights[1])
            return w0 * self.criterion[0](logits, targets) + w1 * self.criterion[1](logits, targets)
        return self.criterion(logits, targets)

    # Giữ API cũ nếu ở nơi khác còn gọi:
    def compute(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.forward(logits, targets)


# --------- Backward compatibility (optional) ----------
# Nếu code cũ vẫn import các hàm dưới, bạn có thể giữ lại để không phải sửa nhiều nơi.

def build_loss(classes=1, name="dice_bce"):
    """Deprecated: dùng LossManager(classes, name).criterion."""
    lm = LossManager(classes=classes, name=name)
    return lm.criterion

def compute_loss(loss_obj, logits, targets):
    """Deprecated: dùng LossManager(...)(logits, targets)."""
    if isinstance(loss_obj, nn.ModuleList):
        # mặc định weights 0.5/0.5 như trước
        return 0.5 * loss_obj[0](logits, targets) + 0.5 * loss_obj[1](logits, targets)
    return loss_obj(logits, targets)

