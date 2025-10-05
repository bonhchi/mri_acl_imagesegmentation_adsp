"""
MRI Knee Preprocessor (fastMRI, single-coil)
- Không đọc .h5 (adapter lo I/O)
- Ăn được record: {'image'| 'target' | 'kspace', 'meta': {...}}
- Pipeline: Recon(if kspace) → Clip → BodyMask → (N4?) → (Denoise?) → Resize → Z-Score(in-mask)
- API chính:
    - MRIKneePreprocessor.preprocess_record(record) -> dict
    - MRIKneePreprocessor.preprocess_records(records) -> dict
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from skimage.filters import threshold_otsu
from skimage.morphology import binary_opening, binary_closing, remove_small_objects, disk
from skimage.restoration import denoise_nl_means, estimate_sigma


class MRIKneePreprocessor:
    """
    Tiền xử lý MRI gối (single-coil).
    """

    # --------- ctor / config ----------
    def __init__(
        self,
        out_size: Tuple[int, int] = (320, 320),
        slice_keep: Tuple[float, float] = (0.3, 0.7),
        clip_percentiles: Tuple[float, float] = (1.0, 99.5),
        use_n4: bool = False,
        use_denoise: bool = False,
    ) -> None:
        self.out_size = out_size
        self.slice_keep = slice_keep
        self.clip_percentiles = clip_percentiles
        self.use_n4 = use_n4
        self.use_denoise = use_denoise
        self._validate()

    # --------- public API ----------
    def preprocess_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Xử lý 1 record từ Adapter.
        Output:
          {
            'img_z': float32 (H,W),       # chuẩn hóa z-score trong mask
            'img_01': float32 (H,W),      # preview [0..1]
            'mask': uint8 (H,W),          # body mask (không phải ACL)
            'meta': dict,                 # passthrough
            'source': 'image'|'target'|'kspace'
          }
        """
        x, src, meta = self._normalize_record_input(record)

        # Recon nếu là kspace
        img = self.ifft2c_single(x) if src == "kspace" else x

        # Clip
        img = self._percentile_clip(img, *self.clip_percentiles)

        # Body mask (tune if ACL needed)
        mk = self._body_mask(img)

        # Optional N4
        if self.use_n4:
            img = self._n4_bias_correction(img, mk)

        # Optional denoise
        if self.use_denoise:
            img = self._rician_denoise(img)

        # Resize
        img_r = self._resize_np(img, self.out_size)
        # Mask resize for downstream steps
        mk_r = (self._resize_np(mk.astype(np.float32), self.out_size) > 0.5).astype(np.uint8)

        # Z-score (in-mask)
        img_z = self._zscore_in_mask(img_r, mk_r)

        # Preview [0..1]
        img_01 = self._preview_01(img_r, mk_r)

        return {
            "img_z": img_z.astype(np.float32),
            "img_01": img_01.astype(np.float32),
            "mask": mk_r,
            "meta": meta,
            "source": src,
        }

    def preprocess_records(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Xử lý nhiều record (ví dụ một volume). Tự cắt dải lát giữa theo slice_keep.
        Output:
          {
            'tensor' : torch.FloatTensor (S,1,H,W),
            'preview': np.ndarray (S,H,W),
            'mask'   : np.ndarray (S,H,W),
            'indices': List[int],
            'sources': List[str],
            'metas'  : List[dict],
          }
        """
        ns = len(records)
        if ns == 0:
            raise ValueError("No records provided to preprocess_records.")

        s0 = max(0, int(ns * self.slice_keep[0]))
        s1 = min(ns, int(ns * self.slice_keep[1]))
        s1 = max(s1, s0 + 1)
        if s1 > ns:
            s1 = ns
        if s0 >= s1:
            s0, s1 = 0, ns
        if s0 >= s1:
            raise ValueError("slice_keep produced an empty selection.")

        imgs, prevs, masks, idxs, sources, metas = [], [], [], [], [], []
        for i in range(s0, s1):
            out = self.preprocess_record(records[i])
            imgs.append(out["img_z"][None, ...])   # (1,H,W)
            prevs.append(out["img_01"])
            masks.append(out["mask"])
            m = out.get("meta", {})
            idxs.append(m.get("slice_idx", i))
            sources.append(out["source"])
            metas.append(m)

        if not imgs:
            raise RuntimeError("No slices were processed from the provided records.")

        vol = np.stack(imgs, axis=0).astype(np.float32)  # (S,1,H,W)
        prv = np.stack(prevs, axis=0).astype(np.float32) # (S,H,W)
        msk = np.stack(masks, axis=0).astype(np.uint8)   # (S,H,W)

        return {
            "tensor": torch.from_numpy(vol),
            "preview": prv,
            "mask": msk,
            "indices": idxs,
            "sources": sources,
            "metas": metas,
        }

    # --------- static / utility you may reuse ----------
    @staticmethod
    def ifft2c_single(kspace_2d: np.ndarray) -> np.ndarray:
        """
        iFFT 2D 'centered' (single-coil).
        Input:  complex (H,W)  |  or (2,H,W) real/imag (will error unless combined earlier)
        Output: float32 (H,W) magnitude
        """
        MRIKneePreprocessor._ensure_2d(kspace_2d, "kspace")
        x = np.fft.ifftshift(kspace_2d, axes=(-2, -1))
        x = np.fft.ifft2(x, norm="ortho")
        x = np.fft.fftshift(x, axes=(-2, -1))
        return np.abs(x).astype(np.float32)

    # --------- private helpers ----------
    def _validate(self) -> None:
        lo, hi = self.slice_keep
        if not (0.0 <= lo < hi <= 1.0):
            raise ValueError("slice_keep phải thoả 0.0 <= lo < hi <= 1.0")
        pmin, pmax = self.clip_percentiles
        if not (0.0 <= pmin < pmax <= 100.0):
            raise ValueError("clip_percentiles phải trong [0,100] và pmin < pmax")

    @staticmethod
    def _to_float32(arr: np.ndarray) -> np.ndarray:
        x = np.squeeze(arr)
        return x.astype(np.float32, copy=False)

    @staticmethod
    def _ensure_2d(x: np.ndarray, name: str) -> np.ndarray:
        if x.ndim != 2:
            raise ValueError(f"{name} phải có shape (H,W), hiện là {x.shape}")
        return x

    @staticmethod
    def _percentile_clip(img: np.ndarray, pmin: float, pmax: float) -> np.ndarray:
        lo, hi = np.percentile(img, pmin), np.percentile(img, pmax)
        return np.clip(img, lo, hi)

    @staticmethod
    def _resize_np(img: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
        t = torch.from_numpy(img)[None, None].float()
        t = F.interpolate(t, size=out_hw, mode="bilinear", align_corners=False)
        return t[0, 0].numpy().astype(np.float32)

    # Mask logic lives here (ACL tweaks)
    @staticmethod
    def _body_mask(img: np.ndarray) -> np.ndarray:
        v = img - img.min()
        vmax = v.max()
        if vmax <= 0:
            return np.zeros_like(img, dtype=np.uint8)
        v = v / vmax
        try:
            th = threshold_otsu(v)
        except ValueError:
            th = float(v.mean())
        if not np.isfinite(th):
            th = 0.5
        m = (v > th).astype(np.uint8)
        if m.sum() == 0:
            return m
        se = disk(2)
        m = binary_opening(m, se)
        m = binary_closing(m, se)
        m = remove_small_objects(m.astype(bool), min_size=256)
        return m.astype(np.uint8)

    @staticmethod
    def _zscore_in_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        vals = img[mask > 0]
        if vals.size < 10:
            mean, std = img.mean(), img.std()
        else:
            mean, std = vals.mean(), vals.std()
        std = std if std > 1e-6 else 1.0
        return ((img - mean) / std).astype(np.float32)

    @staticmethod
    def _preview_01(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        vals = img[mask > 0]
        if vals.size > 0:
            lo, hi = float(vals.min()), float(vals.max())
        else:
            lo, hi = float(img.min()), float(img.max())
        return ((img - lo) / (hi - lo + 1e-6)).astype(np.float32)

    @staticmethod
    def _n4_bias_correction(slice_img: np.ndarray, mask: Optional[np.ndarray] = None, shrink: int = 2) -> np.ndarray:
        try:
            import SimpleITK as sitk
        except Exception:
            return slice_img
        img = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-8)
        itk_img = sitk.GetImageFromArray(img.astype(np.float32))
        itk_mk = sitk.GetImageFromArray(mask.astype(np.uint8)) if mask is not None else sitk.OtsuThreshold(itk_img, 0, 1, 128)
        f = sitk.N4BiasFieldCorrectionImageFilter()
        f.SetMaximumNumberOfIterations([50, 50, 30, 20])
        f.SetShrinkFactor(shrink)
        out = f.Execute(itk_img, itk_mk)
        out = sitk.GetArrayFromImage(out).astype(np.float32)
        return out * (slice_img.max() - slice_img.min() + 1e-8) + slice_img.min()

    @staticmethod
    def _rician_denoise(slice_img: np.ndarray) -> np.ndarray:
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

    # --------- adapter contract normalization ----------
    @staticmethod
    def _normalize_record_input(record: Dict[str, Any]) -> Tuple[np.ndarray, str, Dict[str, Any]]:
        """
        Ưu tiên: image -> target(reconstruction*) -> kspace
        - image/target: float32 (H,W)
        - kspace: complex (H,W) hoặc (2,H,W) real/imag (sẽ error nếu chưa ghép)
        """
        meta = record.get("meta", {})

        if record.get("image", None) is not None:
            img = MRIKneePreprocessor._ensure_2d(MRIKneePreprocessor._to_float32(record["image"]), "image")
            return img, "image", meta

        for k in ("target", "reconstruction", "reconstruction_rss", "reconstruction_esc"):
            if record.get(k, None) is not None:
                rec = MRIKneePreprocessor._ensure_2d(MRIKneePreprocessor._to_float32(record[k]), k)
                return rec, "target", meta

        ksp = record.get("kspace", None)
        if ksp is None:
            raise ValueError("Record không có image/target/kspace hợp lệ.")
        ksp = np.squeeze(ksp)
        # nếu adapter tách real/imag → (2,H,W), bạn nên GHÉP lại trước khi truyền vào class
        if not np.iscomplexobj(ksp):
            if ksp.ndim == 3 and ksp.shape[0] == 2:
                # Nếu bạn muốn ghép ở đây, mở comment 2 dòng dưới:
                # ksp = ksp[0].astype(np.float32) + 1j * ksp[1].astype(np.float32)
                # pass
                raise ValueError("kspace không phải complex. Hãy ghép (real,imag) -> complex trước khi preprocess.")
        ksp = MRIKneePreprocessor._ensure_2d(ksp, "kspace")
        return ksp, "kspace", meta


# --------- convenience API ----------
def _resolve_preprocessor(preprocessor=None, **kwargs):
    """Return an existing preprocessor or build a new one from kwargs."""
    if preprocessor is not None and kwargs:
        raise ValueError('Provide either an existing preprocessor or keyword overrides, not both.')
    return preprocessor or MRIKneePreprocessor(**kwargs)


def preprocess_record(record, *, preprocessor=None, **kwargs):
    """Convenience wrapper around MRIKneePreprocessor.preprocess_record."""
    pre = _resolve_preprocessor(preprocessor, **kwargs)
    return pre.preprocess_record(record)


def preprocess_records(records, *, preprocessor=None, **kwargs):
    """Convenience wrapper around MRIKneePreprocessor.preprocess_records."""
    pre = _resolve_preprocessor(preprocessor, **kwargs)
    return pre.preprocess_records(records)


__all__ = ['MRIKneePreprocessor', 'preprocess_record', 'preprocess_records']
