import h5py
import numpy as np
from typing import Dict, Any, List, Optional
from .base_adapter import BaseAdapter  # như bạn đã có
from src.utils.kspace import ifft2c, complex_abs, center_crop_or_pad

class FastMRISinglecoilAdapter(BaseAdapter):
    """
    Chuẩn hoá mỗi slice thành dict:
      {
        "image": np.ndarray float32 [H, W] (magnitude, đã normalize),
        "target": np.ndarray | None (nếu có reconstruction_*),
        "meta": {...}  (filename, slice_idx, patient/series nếu đọc được),
        "kspace": Optional[np.ndarray complex64 [H, W]] (tuỳ bạn có muốn trả ra)
      }
    """
    def __init__(
        self,
        target_size: Optional[tuple] = (320, 320),  # knee phổ biến 320x320
        keep_kspace: bool = False,
        normalize: str = "minmax"  # "minmax" | "zscore" | None
    ):
        self.target_size = target_size
        self.keep_kspace = keep_kspace
        self.normalize = normalize

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        img = img.astype(np.float32)
        if self.normalize == "minmax":
            vmin, vmax = img.min(), img.max()
            if vmax > vmin:
                img = (img - vmin) / (vmax - vmin)
        elif self.normalize == "zscore":
            mu, sd = img.mean(), img.std()
            img = (img - mu) / (sd + 1e-8)
        return img

    def load(self, h5_path: str) -> Dict[str, Any]:
        """
        Đọc toàn bộ file: trả về dict với danh sách slice-level records (chưa preprocess).
        """
        slices = []
        with h5py.File(h5_path, "r") as hf:
            # Bắt buộc có 'kspace'
            kspace = hf["kspace"][:]  # shape [S, H, W], complex64
            S = kspace.shape[0]

            # target (nếu có)
            target = None
            target_key = None
            for cand in ["reconstruction_rss", "reconstruction_esc", "reconstruction_espirit", "reconstruction"]:
                if cand in hf:
                    target = hf[cand][:]
                    target_key = cand
                    break

            # meta
            ismrmrd_header = hf["ismrmrd_header"][()].decode("utf-8") if "ismrmrd_header" in hf else None

            for s in range(S):
                record = {
                    "filepath": h5_path,
                    "slice_idx": s,
                    "kspace": kspace[s],                # complex2d
                    "target": None if target is None else target[s],
                    "target_key": target_key,
                    "ismrmrd_header": ismrmrd_header,
                }
                slices.append(record)

        return {"records": slices}

    def preprocess(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Từ 1 slice record → ảnh magnitude chuẩn hoá + meta.
        """
        k = record["kspace"]  # complex2d
        img_cplx = ifft2c(k)  # complex image domain
        mag = complex_abs(img_cplx)  # [H, W], float

        if self.target_size is not None:
            mag = center_crop_or_pad(mag, self.target_size[0], self.target_size[1])

        img = self._normalize(mag)

        # target nếu có (chuẩn hoá cùng cách để so sánh MSE/SSIM nếu bạn muốn)
        tgt = record["target"]
        if tgt is not None:
            if self.target_size is not None:
                tgt = center_crop_or_pad(tgt, self.target_size[0], self.target_size[1])
            tgt = self._normalize(tgt.astype(np.float32))

        example = {
            "image": img.astype(np.float32),                 # [H, W]
            "target": None if tgt is None else tgt.astype(np.float32),
            "meta": {
                "filepath": record["filepath"],
                "slice_idx": record["slice_idx"],
                "target_key": record["target_key"],
            }
        }
        if self.keep_kspace:
            example["kspace"] = k  # giữ nguyên complex64

        return example