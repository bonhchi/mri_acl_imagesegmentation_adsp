# src/preprocess/mri_preprocess_min.py
import numpy as np
import torch, torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional

from skimage.filters import threshold_otsu
from skimage.morphology import binary_opening, binary_closing, remove_small_objects, disk
from skimage.restoration import denoise_nl_means, estimate_sigma

# ---------- FFT helpers (single-coil) ----------
def fftshift2d(x):     return np.fft.fftshift(x, axes=(-2, -1))
def ifftshift2d(x):    return np.fft.ifftshift(x, axes=(-2, -1))
def ifft2c_single(k2d: np.ndarray) -> np.ndarray:
    x = ifftshift2d(k2d); x = np.fft.ifft2(x, norm="ortho"); x = fftshift2d(x)
    return np.abs(x).astype(np.float32)

# ---------- image utils ----------
def resize_np(img: np.ndarray, out_hw: Tuple[int,int]) -> np.ndarray:
    t = torch.from_numpy(img)[None,None].float()
    t = F.interpolate(t, size=out_hw, mode="bilinear", align_corners=False)
    return t[0,0].numpy().astype(np.float32)

def percentile_clip(img: np.ndarray, pmin=1.0, pmax=99.5) -> np.ndarray:
    lo, hi = np.percentile(img, pmin), np.percentile(img, pmax)
    return np.clip(img, lo, hi)

def body_mask(img: np.ndarray) -> np.ndarray:
    v = img - img.min(); vmax = v.max()
    if vmax > 0: v = v / vmax
    m = (v > threshold_otsu(v)).astype(np.uint8)
    se = disk(2)
    m = binary_opening(m, se); m = binary_closing(m, se)
    m = remove_small_objects(m.astype(bool), min_size=256)
    return m.astype(np.uint8)

def zscore_in_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vals = img[mask > 0]
    m, s = (img.mean(), img.std()) if vals.size < 10 else (vals.mean(), vals.std())
    s = s if s > 1e-6 else 1.0
    return ((img - m) / s).astype(np.float32)

def n4_bias_correction(img: np.ndarray, mask: Optional[np.ndarray]=None, shrink=2) -> np.ndarray:
    try:
        import SimpleITK as sitk
    except Exception:
        return img
    nor = (img - img.min()) / (img.max() - img.min() + 1e-8)
    itk_img = sitk.GetImageFromArray(nor)
    itk_mk = sitk.GetImageFromArray(mask.astype(np.uint8)) if mask is not None else sitk.OtsuThreshold(itk_img, 0, 1, 128)
    f = sitk.N4BiasFieldCorrectionImageFilter()
    f.SetMaximumNumberOfIterations([50, 50, 30, 20]); f.SetShrinkFactor(shrink)
    out = sitk.GetArrayFromImage(f.Execute(itk_img, itk_mk)).astype(np.float32)
    return out * (img.max() - img.min() + 1e-8) + img.min()

def rician_denoise(img: np.ndarray) -> np.ndarray:
    sig = float(np.mean(estimate_sigma(img, channel_axis=None)))
    h = 0.8 * sig if sig > 0 else 0.01
    den = denoise_nl_means(img, h=h, patch_size=3, patch_distance=5,
                           fast_mode=True, channel_axis=None)
    return den.astype(np.float32)

# ---------- preprocess 1 slice ----------
def preprocess_slice(
    kspace: Optional[np.ndarray]=None,
    recon: Optional[np.ndarray]=None,
    out_size: Tuple[int,int]=(320,320),
    use_n4: bool=False,
    use_denoise: bool=False
) -> Dict[str, np.ndarray]:
    if kspace is not None:      img = ifft2c_single(kspace)
    elif recon is not None:     img = recon.astype(np.float32)
    else:                       raise ValueError("Cần kspace hoặc recon.")

    img = percentile_clip(img, 1.0, 99.5)
    mk  = body_mask(img)
    if use_n4:      img = n4_bias_correction(img, mk)
    if use_denoise: img = rician_denoise(img)

    img_r = resize_np(img, out_size)
    mk_r  = (resize_np(mk.astype(np.float32), out_size) > 0.5).astype(np.uint8)
    img_z = zscore_in_mask(img_r, mk_r)

    vals = img_r[mk_r > 0]; 
    mm_min, mm_max = (img_r.min(), img_r.max()) if vals.size == 0 else (float(vals.min()), float(vals.max()))
    img_01 = (img_r - mm_min) / (mm_max - mm_min + 1e-6)

    return {"img_z": img_z, "img_01": img_01.astype(np.float32), "mask": mk_r}

# ---------- preprocess nhiều lát (subset giữa volume) ----------
def preprocess_records(
    records: list,                     # list từ Adapter: mỗi item có 'kspace' hoặc 'target' + 'slice_idx'
    out_size: Tuple[int,int]=(320,320),
    slice_keep: Tuple[float,float]=(0.3, 0.7),
    use_n4: bool=False,
    use_denoise: bool=False
) -> Dict[str, Any]:
    ns = len(records)
    s0, s1 = max(0, int(ns*slice_keep[0])), min(ns, int(ns*slice_keep[1]))
    imgs, prevs, masks, idxs = [], [], [], []
    for i in range(s0, s1):
        r = records[i]
        p = preprocess_slice(kspace=r.get("kspace"), recon=r.get("target"),
                             out_size=out_size, use_n4=use_n4, use_denoise=use_denoise)
        imgs.append(p["img_z"][None, ...])   # (1,H,W)
        prevs.append(p["img_01"])
        masks.append(p["mask"])
        idxs.append(r["slice_idx"])
    vol = np.stack(imgs, axis=0).astype(np.float32)    # (S,1,H,W)
    prv = np.stack(prevs, axis=0).astype(np.float32)   # (S,H,W)
    msk = np.stack(masks, axis=0).astype(np.uint8)     # (S,H,W)
    return {"tensor": torch.from_numpy(vol), "preview": prv, "mask": msk, "indices": idxs}