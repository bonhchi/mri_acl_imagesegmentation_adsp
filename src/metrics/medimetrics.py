# -*- coding: utf-8 -*-
"""
Dice/IoU chung; HD95/ASSD nếu có SciPy. Dùng cho báo cáo.
"""
from __future__ import annotations
import numpy as np

def dice_bin(pred: np.ndarray, gt: np.ndarray, eps=1e-7) -> float:
    # pred, gt: (H,W) {0,1}
    inter = (pred & gt).sum()
    return float((2*inter + eps) / (pred.sum() + gt.sum() + eps))

def iou_bin(pred: np.ndarray, gt: np.ndarray, eps=1e-7) -> float:
    inter = (pred & gt).sum()
    union = pred.sum() + gt.sum() - inter
    return float((inter + eps) / (union + eps))

def _surface_distances(a, b, spacing=None):
    try:
        from scipy.ndimage import distance_transform_edt as edt
    except Exception:
        raise ImportError("HD95/ASSD cần scipy.ndimage (scipy).")
    a = a.astype(bool); b = b.astype(bool)
    if spacing is None: spacing = (1.0, 1.0)
    a_border = a ^ np.logical_and(edt(~a) > 0, a)
    b_border = b ^ np.logical_and(edt(~b) > 0, b)
    if not a_border.any(): a_border = a
    if not b_border.any(): b_border = b
    # distance from A border to nearest B voxel
    dt = edt(~b, sampling=spacing)
    d_ab = dt[a_border]
    dt2 = edt(~a, sampling=spacing)
    d_ba = dt2[b_border]
    return np.concatenate([d_ab, d_ba])

def hd95(pred: np.ndarray, gt: np.ndarray, spacing=None) -> float:
    d = _surface_distances(pred.astype(bool), gt.astype(bool), spacing)
    if d.size == 0: return 0.0
    return float(np.percentile(d, 95))

def assd(pred: np.ndarray, gt: np.ndarray, spacing=None) -> float:
    d = _surface_distances(pred.astype(bool), gt.astype(bool), spacing)
    if d.size == 0: return 0.0
    return float(d.mean())
