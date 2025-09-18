# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, eps=1e-7):
        super().__init__()
        self.a, self.b, self.eps = alpha, beta, eps
    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        dims = (0, 2, 3)
        TP = (p * targets).sum(dims)
        FP = (p * (1 - targets)).sum(dims)
        FN = ((1 - p) * targets).sum(dims)
        tv = (TP + self.eps) / (TP + self.a * FP + self.b * FN + self.eps)
        return 1 - tv.mean()

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75):
        super().__init__()
        self.tv, self.g = TverskyLoss(alpha, beta), gamma
    def forward(self, logits, targets):
        t = 1 - self.tv(logits, targets)
        return torch.pow(1 - t, self.g)

def build_loss(classes=1, name="dice_bce"):
    if classes == 1:
        if name == "dice_bce":
            return nn.ModuleList([smp.losses.DiceLoss("binary"),
                                  smp.losses.SoftBCEWithLogitsLoss()])
        if name == "focal":
            return smp.losses.FocalLoss(mode="binary", alpha=0.25, gamma=2.0)
        if name == "tversky":
            return TverskyLoss()
        if name == "focal_tversky":
            return FocalTverskyLoss()
        raise ValueError("Unknown binary loss")
    else:
        if name == "dice_ce":
            return nn.ModuleList([smp.losses.DiceLoss("multiclass"),
                                  nn.CrossEntropyLoss()])
        if name == "ce":
            return nn.CrossEntropyLoss()
        raise ValueError("Unknown multiclass loss")

def compute_loss(loss_obj, logits, targets):
    if isinstance(loss_obj, nn.ModuleList):
        return 0.5 * loss_obj[0](logits, targets) + 0.5 * loss_obj[1](logits, targets)
    return loss_obj(logits, targets)
