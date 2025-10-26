# src/adapters/oai_zib_adapter.py
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base_adapter import BaseAdapter


class OAIZibVolumeAdapter(BaseAdapter):
    """
    Adapter for the OAI-ZIB knee MRI dataset stored as paired numpy volumes.

    The dataset structure is expected to be:
        root/
            numpy_mri/*.npz          # image volumes, key 'data' -> (H, W, S)
            numpy_mask/*.npz         # mask volumes, key 'data' -> (H, W, S)
            oai_zib_label_path.csv   # mapping between MRI and mask file names

    Each record returned corresponds to a single 2D slice.
    """

    def __init__(
        self,
        root_dir: Optional[str] = None,
        mapping_csv: str = "oai_zib_label_path.csv",
        mri_subdir: str = "numpy_mri",
        mask_subdir: str = "numpy_mask",
        cache_volumes: bool = True,
    ) -> None:
        resolved = root_dir or os.getenv("OAI_ZIB_ROOT")
        if not resolved:
            raise ValueError("Must provide root_dir or set OAI_ZIB_ROOT")
        super().__init__(str(Path(resolved).resolve()))
        self.mapping_csv = mapping_csv
        self.mri_subdir = mri_subdir
        self.mask_subdir = mask_subdir
        self.cache_volumes = cache_volumes
        self._mri_cache: Dict[str, np.ndarray] = {}
        self._mask_cache: Dict[str, np.ndarray] = {}
        self._depth_cache: Dict[str, int] = {}
        self._mapping_cache: Optional[List[Tuple[str, str]]] = None

    # ------------- discovery -------------------------------------------------
    def discover_records(self, root_dir: Optional[str] = None) -> List[Dict[str, object]]:
        root = Path(root_dir or self.root_dir or ".").resolve()
        mapping = self._load_mapping(root)
        records: List[Dict[str, object]] = []
        missing: List[str] = []

        for mri_name, mask_name in mapping:
            try:
                mri_path = self._resolve_volume_path(root, self.mri_subdir, mri_name)
                mask_path = self._resolve_volume_path(root, self.mask_subdir, mask_name)
            except FileNotFoundError as exc:
                missing.append(str(exc))
                continue

            depth = self._volume_depth(mri_path)
            subject_id, visit_id, side = self._parse_filename(Path(mri_name).stem)
            for slice_idx in range(depth):
                records.append(
                    {
                        "filepath": str(mri_path),
                        "mask_path": str(mask_path),
                        "slice_idx": slice_idx,
                        "subject_id": subject_id,
                        "visit_id": visit_id,
                        "side": side,
                    }
                )

        if missing:
            preview = ", ".join(missing[:5])
            if len(missing) > 5:
                preview += f", ... (+{len(missing) - 5} more)"
            print(f"[warn] OAI-ZIB adapter skipped {len(missing)} pair(s): {preview}")
        return records

    # ------------- loading ---------------------------------------------------
    def load_record(self, record: Dict[str, object]) -> Dict[str, object]:
        slice_idx = int(record["slice_idx"])
        mri_path = Path(record["filepath"])
        mask_path = Path(record["mask_path"])

        mri_vol = self._load_mri_volume(mri_path)
        mask_vol = self._load_mask_volume(mask_path)

        if slice_idx < 0 or slice_idx >= mri_vol.shape[0]:
            raise IndexError(f"Slice index {slice_idx} out of range for {mri_path.name}")
        if mask_vol.shape[0] != mri_vol.shape[0]:
            raise ValueError(
                f"Mask depth {mask_vol.shape[0]} does not match MRI depth {mri_vol.shape[0]} for {mri_path.name}"
            )

        img_slice = mri_vol[slice_idx].astype(np.float32, copy=False)
        mask_slice = mask_vol[slice_idx].astype(np.uint8, copy=False)

        meta = {
            "filepath": str(mri_path),
            "mask_path": str(mask_path),
            "slice_idx": slice_idx,
            "dataset": "oaizib",
            "subject_id": record.get("subject_id"),
            "visit_id": record.get("visit_id"),
            "side": record.get("side"),
        }
        return {
            "image": img_slice,
            "mask": mask_slice,
            "label": None,
            "meta": meta,
        }

    # ------------- helpers ---------------------------------------------------
    def _load_mapping(self, root: Path) -> List[Tuple[str, str]]:
        if self._mapping_cache is not None:
            return self._mapping_cache

        csv_path = self._resolve_mapping_path(root)
        if not csv_path.exists():
            raise FileNotFoundError(f"Mapping CSV not found: {csv_path}")

        pairs: List[Tuple[str, str]] = []
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                mri = row.get("mri_path")
                mask = row.get("label_path")
                if not mri or not mask:
                    continue
                pairs.append((mri.strip(), mask.strip()))

        self._mapping_cache = pairs
        return pairs

    def _resolve_mapping_path(self, root: Path) -> Path:
        candidate = Path(self.mapping_csv)
        return candidate if candidate.is_absolute() else root / candidate

    def _resolve_volume_path(self, root: Path, subdir: str, filename: str) -> Path:
        candidate = Path(filename)
        if not candidate.is_absolute():
            candidate = root / subdir / filename
        if not candidate.exists():
            raise FileNotFoundError(f"{filename} not found under {root / subdir}")
        return candidate.resolve()

    def _volume_depth(self, mri_path: Path) -> int:
        key = str(mri_path)
        if key in self._depth_cache:
            return self._depth_cache[key]
        vol = self._load_mri_volume(mri_path)
        depth = int(vol.shape[0])
        self._depth_cache[key] = depth
        return depth

    def _load_mri_volume(self, path: Path) -> np.ndarray:
        key = str(path)
        if self.cache_volumes and key in self._mri_cache:
            return self._mri_cache[key]
        arr = self._load_npz_array(path)
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D MRI volume, got shape {arr.shape} for {path.name}")
        vol = self._to_depth_first(arr).astype(np.float32, copy=False)
        if self.cache_volumes:
            self._mri_cache[key] = vol
        return vol

    def _load_mask_volume(self, path: Path) -> np.ndarray:
        key = str(path)
        if self.cache_volumes and key in self._mask_cache:
            return self._mask_cache[key]
        arr = self._load_npz_array(path)
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D mask volume, got shape {arr.shape} for {path.name}")
        vol = self._to_depth_first(arr).astype(np.uint8, copy=False)
        if self.cache_volumes:
            self._mask_cache[key] = vol
        return vol

    @staticmethod
    def _load_npz_array(path: Path) -> np.ndarray:
        with np.load(path) as data:
            if "data" not in data:
                raise KeyError(f"'data' key not found in {path.name}")
            arr = data["data"]
        return np.asarray(arr)

    @staticmethod
    def _to_depth_first(arr: np.ndarray) -> np.ndarray:
        # Files are typically stored as (H, W, S); detect and convert to (S, H, W).
        if arr.ndim != 3:
            return arr
        last_dim = arr.shape[-1]
        if last_dim <= arr.shape[0] and last_dim <= arr.shape[1]:
            return np.moveaxis(arr, -1, 0)
        return arr

    @staticmethod
    def _parse_filename(stem: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        parts = stem.split("_")
        subject = parts[0] if parts else None
        visit = parts[1] if len(parts) > 1 else None
        side = parts[2] if len(parts) > 2 else None
        return subject, visit, side


__all__ = ["OAIZibVolumeAdapter"]
