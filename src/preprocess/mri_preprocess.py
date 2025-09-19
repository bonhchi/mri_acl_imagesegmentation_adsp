"""
MRI Preprocess (fastMRI single-coil) — adapter-agnostic

Hỗ trợ đầu vào từ Adapter theo "contract" linh hoạt:
- record["image"]  : magnitude float32 (H, W)  [ưu tiên]
- record["target"] : reconstruction float32 (H, W) (hoặc 'reconstruction[_rss/_esc]') [fallback]
- record["kspace"] : complex (H, W) single-coil  [cuối cùng mới dùng]
- record["meta"]   : dict (filepath, slice_idx, adapter, ...)

Pipeline:
  (1) Recon (nếu kspace) → magnitude
  (2) Intensity clip (percentile 1..99.5)
  (3) Body mask (Otsu + morphology)
  (4) (Optional) N4 bias correction (SimpleITK)
  (5) (Optional) NL-means denoise (gần Rician)
  (6) Resize cố định (vd. 320×320)
  (7) Z-score trong mask (mean≈0, std≈1)
  (8) Preview [0..1] để QA

API chính:
- preprocess_slice_record(record, out_size=(320,320), use_n4=False, use_denoise=False)
- preprocess_records(records, out_size=(320,320), slice_keep=(0.3,0.7), use_n4=False, use_denoise=False)

Lưu ý:
- Không đọc file .h5 trong module này.
- Chỉ cho single-coil. Nếu adapter tách real/imag thành (2,H,W), module sẽ ghép lại.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import torch, torch.nn.functional as F

from skimage.filters import threshold_otsu
from skimage.morphology import (
    binary_opening,
    binary_closing,
    remove_small_objects,
    disk,
)
from skimage.restoration import denoise_nl_means, estimate_sigma


# ==============================================================
# 0) Helpers chung
# ==============================================================


def _to_float32(arr: np.ndarray) -> np.ndarray:
    """Cast an toàn sang float32, squeeze kênh dư."""
    x = np.squeeze(arr)
    return x.astype(np.float32, copy=False)


def _ensure_2d(x: np.ndarray, name: str) -> np.ndarray:
    """Đảm bảo (H, W)."""
    if x.ndim != 2:
        raise ValueError(f"{name} phải có shape (H, W), hiện là {x.shape}")
    return x


def _percentile_clip(
    img: np.ndarray, pmin: float = 1.0, pmax: float = 99.5
) -> np.ndarray:
    """Clip theo percentile để bỏ outlier cường độ."""
    lo, hi = np.percentile(img, pmin), np.percentile(img, pmax)
    return np.clip(img, lo, hi)


def _resize_np(img: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    """Resize bằng PyTorch (bilinear), trả về float32 (H, W)."""
    t = torch.from_numpy(img)[None, None].float()
    t = F.interpolate(t, size=out_hw, mode="bilinear", align_corners=False)
    return t[0, 0].numpy().astype(np.float32)


def _body_mask(img: np.ndarray) -> np.ndarray:
    """
    Sinh mask cơ thể: Otsu + morphology + remove small.
    Trả về nhị phân uint8 {0,1}.
    """
    v = img - img.min()
    vmax = v.max()
    if vmax > 0:
        v = v / vmax
    th = threshold_otsu(v)
    m = (v > th).astype(np.uint8)
    se = disk(2)
    m = binary_opening(m, se)
    m = binary_closing(m, se)
    m = remove_small_objects(m.astype(bool), min_size=256)
    return m.astype(np.uint8)


def _zscore_in_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Z-score trong vùng mask; fallback global nếu mask quá ít."""
    vals = img[mask > 0]
    if vals.size < 10:
        mean, std = img.mean(), img.std()
    else:
        mean, std = vals.mean(), vals.std()
    std = std if std > 1e-6 else 1.0
    return ((img - mean) / std).astype(np.float32)


def _preview_01(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Tạo ảnh xem trước [0..1] theo min/max trong mask (fallback global)."""
    vals = img[mask > 0]
    if vals.size > 0:
        lo, hi = float(vals.min()), float(vals.max())
    else:
        lo, hi = float(img.min()), float(img.max())
    return ((img - lo) / (hi - lo + 1e-6)).astype(np.float32)


# ==============================================================
# 1) FFT cho single-coil
# ==============================================================


def _fftshift2d(x: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(x, axes=(-2, -1))


def _ifftshift2d(x: np.ndarray) -> np.ndarray:
    return np.fft.ifftshift(x, axes=(-2, -1))


def ifft2c_single(kspace_2d: np.ndarray) -> np.ndarray:
    """
    iFFT 2D "centered" cho single-coil.
    Input:  kspace_2d complex (H, W)
    Output: magnitude float32 (H, W)
    """
    kspace_2d = _ensure_2d(kspace_2d, "kspace")
    x = _ifftshift2d(kspace_2d)
    x = np.fft.ifft2(x, norm="ortho")
    x = _fftshift2d(x)
    return np.abs(x).astype(np.float32)


# ==============================================================
# 2) Tuỳ chọn: N4 bias correction & NL-means denoise
# ==============================================================


def n4_bias_correction(
    slice_img: np.ndarray, mask: Optional[np.ndarray] = None, shrink: int = 2
) -> np.ndarray:
    """
    N4 bias correction (SimpleITK). Nếu không có SITK, trả ảnh gốc.
    slice_img: float32 (H, W)
    mask     : uint8 {0,1} (H, W) hoặc None
    """
    try:
        import SimpleITK as sitk
    except Exception:
        return slice_img

    img = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-8)
    itk_img = sitk.GetImageFromArray(img.astype(np.float32))
    itk_mk = (
        sitk.GetImageFromArray(mask.astype(np.uint8))
        if mask is not None
        else sitk.OtsuThreshold(itk_img, 0, 1, 128)
    )

    f = sitk.N4BiasFieldCorrectionImageFilter()
    f.SetMaximumNumberOfIterations([50, 50, 30, 20])
    f.SetShrinkFactor(shrink)

    out = f.Execute(itk_img, itk_mk)
    out = sitk.GetArrayFromImage(out).astype(np.float32)

    # scale về dynamic gốc
    return out * (slice_img.max() - slice_img.min() + 1e-8) + slice_img.min()


def rician_denoise(slice_img: np.ndarray) -> np.ndarray:
    """
    Khử nhiễu nhẹ bằng NL-means (gần đúng Rician cho magnitude).
    Giữ texture; dùng khi ảnh nhiễu nặng.
    """
    sigma = float(np.mean(estimate_sigma(slice_img, channel_axis=None)))
    h = 0.8 * sigma if sigma > 0 else 0.01
    den = denoise_nl_means(
        slice_img,
        h=h,
        patch_size=3,
        patch_distance=5,
        fast_mode=True,
        channel_axis=None,
    )
    return den.astype(np.float32)


# ==============================================================
# 3) Chuẩn hoá đầu vào record từ Adapter
# ==============================================================

_RECON_KEYS = (
    "image",
    "target",
    "reconstruction",
    "reconstruction_rss",
    "reconstruction_esc",
)


def normalize_record_input(
    record: Dict[str, Any]
) -> Tuple[np.ndarray, str, Dict[str, Any]]:
    """
    Chuẩn hoá 1 record từ Adapter để sẵn sàng preprocess.

    Trả về:
        img_or_kspace : np.ndarray
            - Nếu source in {'image','target'} → magnitude float32 (H,W)
            - Nếu source == 'kspace'          → complex (H,W)
        source        : 'image' | 'target' | 'kspace'
        meta          : dict (nguyên trạng record['meta'] nếu có)
    """
    meta = record.get("meta", {})

    # 1) Ưu tiên 'image'
    if "image" in record and record["image"] is not None:
        img = _ensure_2d(_to_float32(record["image"]), "image")
        return img, "image", meta

    # 2) Fallback sang các khoá 'target' / 'reconstruction*'
    for k in _RECON_KEYS[1:]:
        if record.get(k, None) is not None:
            rec = _ensure_2d(_to_float32(record[k]), k)
            return rec, "target", meta

    # 3) Cuối cùng dùng kspace
    ksp = record.get("kspace", None)
    if ksp is None:
        raise ValueError("Record không có image/target/kspace hợp lệ.")

    ksp = np.squeeze(ksp)
    # Adapter có thể tách real/imag thành (2,H,W) → ghép lại complex
    if ksp.ndim == 3 and ksp.shape[0] == 2 and not np.iscomplexobj(ksp):
        ksp = ksp[0].astype(np.float32) + 1j * ksp[1].astype(np.float32)

    if not np.iscomplexobj(ksp):
        raise ValueError("kspace phải là complex (single-coil).")

    ksp = _ensure_2d(ksp, "kspace")
    return ksp, "kspace", meta


# ==============================================================
# 4) Preprocess 1 record (1 lát)
# ==============================================================


def preprocess_slice_record(
    record: Dict[str, Any],
    out_size: Tuple[int, int] = (320, 320),
    use_n4: bool = False,
    use_denoise: bool = False,
    clip_percentiles: Tuple[float, float] = (1.0, 99.5),
) -> Dict[str, Any]:
    """
    Tiền xử lý 1 lát (single-coil) từ record Adapter.

    Output:
        {
          'img_z'  : float32 (H,W) — z-score trong mask (đầu vào cho model)
          'img_01' : float32 (H,W) — preview [0..1] để QA
          'mask'   : uint8   (H,W) — body mask (không phải ACL)
          'meta'   : dict    — passthrough
          'source' : str     — 'image' | 'target' | 'kspace'
        }
    """
    x, src, meta = normalize_record_input(record)

    # Recon nếu là kspace
    if src == "kspace":
        img = ifft2c_single(x)
    else:
        img = x  # magnitude đã sẵn

    # 1) Clip
    img = _percentile_clip(img, *clip_percentiles)

    # 2) Mask
    mk = _body_mask(img)

    # 3) Optional N4
    if use_n4:
        img = n4_bias_correction(img, mk)

    # 4) Optional denoise
    if use_denoise:
        img = rician_denoise(img)

    # 5) Resize
    img_r = _resize_np(img, out_size)
    mk_r = (_resize_np(mk.astype(np.float32), out_size) > 0.5).astype(np.uint8)

    # 6) Z-score trong mask
    img_z = _zscore_in_mask(img_r, mk_r)

    # 7) Preview
    img_01 = _preview_01(img_r, mk_r)

    return {
        "img_z": img_z.astype(np.float32),
        "img_01": img_01.astype(np.float32),
        "mask": mk_r,
        "meta": meta,
        "source": src,
    }


# ==============================================================
# 5) Preprocess nhiều record (subset lát giữa volume)
# ==============================================================


def preprocess_records(
    records: List[Dict[str, Any]],
    out_size: Tuple[int, int] = (320, 320),
    slice_keep: Tuple[float, float] = (0.3, 0.7),
    use_n4: bool = False,
    use_denoise: bool = False,
    clip_percentiles: Tuple[float, float] = (1.0, 99.5),
) -> Dict[str, Any]:
    """
    Tiền xử lý nhiều lát (ví dụ một volume) từ list record Adapter.

    slice_keep: giữ dải lát giữa volume để tập trung vùng ACL (vd. 0.3..0.7).
    Output:
        {
          'tensor' : torch.FloatTensor (S,1,H,W) — img_z stack
          'preview': np.ndarray        (S,H,W)   — [0..1] để QA/hiển thị
          'mask'   : np.ndarray        (S,H,W)   — body mask
          'indices': List[int]                    — slice_idx (nếu meta có)
          'sources': List[str]                    — nguồn mỗi lát ('image'|'target'|'kspace')
          'metas'  : List[dict]                   — passthrough meta
        }
    """
    ns = len(records)
    s0 = max(0, int(ns * slice_keep[0]))
    s1 = min(ns, int(ns * slice_keep[1]))

    imgs, prevs, masks, idxs, sources, metas = [], [], [], [], [], []

    for i in range(s0, s1):
        out = preprocess_slice_record(
            records[i],
            out_size=out_size,
            use_n4=use_n4,
            use_denoise=use_denoise,
            clip_percentiles=clip_percentiles,
        )
        imgs.append(out["img_z"][None, ...])  # (1,H,W)
        prevs.append(out["img_01"])
        masks.append(out["mask"])
        meta = out.get("meta", {})
        idxs.append(meta.get("slice_idx", i))
        sources.append(out["source"])
        metas.append(meta)

    vol = np.stack(imgs, axis=0).astype(np.float32)  # (S,1,H,W)
    prv = np.stack(prevs, axis=0).astype(np.float32)  # (S,H,W)
    msk = np.stack(masks, axis=0).astype(np.uint8)  # (S,H,W)

    return {
        "tensor": torch.from_numpy(vol),
        "preview": prv,
        "mask": msk,
        "indices": idxs,
        "sources": sources,
        "metas": metas,
    }


# ==============================================================
# 6) Public interface
# ==============================================================

__all__ = [
    "ifft2c_single",
    "n4_bias_correction",
    "rician_denoise",
    "normalize_record_input",
    "preprocess_slice_record",
    "preprocess_records",
]
