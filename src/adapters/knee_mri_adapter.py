# src/adapters/knee_mri_adapter.py
import csv
import glob
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base_adapter import BaseAdapter

def _coerce_img(x):
    """Đưa về np.ndarray (H,W) hoặc (H,W,C)."""
    if isinstance(x, np.ndarray):
        return x
    try:
        # PIL Image?
        from PIL import Image
        if isinstance(x, Image.Image):
            return np.array(x)
    except Exception:
        pass
    # fallback: cố ép
    return np.array(x)

class KneePckAdapter(BaseAdapter):
    """
    Đọc dataset pickle (.pck/.pickle) cho Knee MRI classification.

    discover_records: trả list các {'pck_path', 'item_idx'}
    load_record: trả {'image', 'mask': None, 'label', 'meta'}
    """

    def __init__(self, pck_pattern=("*.pck", "*.pickle")):
        self.pck_pattern = pck_pattern

    def _list_pck_files(self, root_dir):
        files = []
        for pat in self.pck_pattern:
            files += glob.glob(os.path.join(root_dir, "**", pat), recursive=True)
        files = sorted(set(files))
        return files

    def _probe_length(self, pck_path):
        """Mở pickle đọc 1 lần để biết số phần tử."""
        with open(pck_path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict):
            for key in ["data", "images", "x", "X"]:
                if key in obj:
                    return len(obj[key])
            # nếu có 'labels' thì dùng len(labels)
            for key in ["labels", "y", "Y", "target", "targets"]:
                if key in obj:
                    return len(obj[key])
            # dict không chuẩn → cố tìm list đầu tiên
            for v in obj.values():
                if isinstance(v, (list, tuple, np.ndarray)):
                    return len(v)
            raise ValueError(f"Unrecognized dict layout in {pck_path}")
        elif isinstance(obj, (list, tuple)):
            return len(obj)
        else:
            raise ValueError(f"Unsupported pickle root type: {type(obj)} in {pck_path}")

    def discover_records(self, root_dir):
        records = []
        for pck in self._list_pck_files(root_dir):
            n = self._probe_length(pck)
            for i in range(n):
                records.append({"pck_path": pck, "item_idx": i})
        return records

    def _read_item(self, pck_path, idx):
        """Đọc đúng phần tử idx từ pickle (mở file mỗi lần — đơn giản & an toàn RAM)."""
        with open(pck_path, "rb") as f:
            obj = pickle.load(f)

        if isinstance(obj, dict):
            # tìm ảnh
            for key in ["data", "images", "x", "X"]:
                if key in obj:
                    img = _coerce_img(obj[key][idx])
                    break
            else:
                # dạng dict các list/ndarray: lấy list/ndarray đầu tiên làm ảnh
                arr_keys = [k for k, v in obj.items() if isinstance(v, (list, tuple, np.ndarray))]
                if not arr_keys:
                    raise ValueError(f"No array-like found in dict of {pck_path}")
                img = _coerce_img(obj[arr_keys[0]][idx])

            # tìm label
            label = None
            for lk in ["labels", "y", "Y", "target", "targets"]:
                if lk in obj:
                    label = int(obj[lk][idx])
                    break
            # Nếu không có label, để None
            return img, label

        elif isinstance(obj, (list, tuple)):
            item = obj[idx]
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                img = _coerce_img(item[0])
                label = int(item[1]) if item[1] is not None else None
            else:
                img = _coerce_img(item)
                label = None
            return img, label

        else:
            raise ValueError(f"Unsupported pickle root type while read: {type(obj)}")

    def load_record(self, record):
        pck_path, idx = record["pck_path"], record["item_idx"]
        img, label = self._read_item(pck_path, idx)

        return {
            "image": img,          # np.ndarray, raw
            "mask": None,
            "label": label,
            "meta": {
                "filepath": pck_path,
                "item_idx": idx,
                "dataset": "kaggle-knee",
            }
        }


class KneeMRIVolumeAdapter(BaseAdapter):
    """
    Adapter for the knee MRI pickle dataset (dataset/kneemri).
    Each pickle file stores a 3D volume (depth x height x width) with ROI metadata
    provided in metadata.csv. We expose per-slice records with bounding-box masks.
    """

    def __init__(
        self,
        root_dir: Optional[str] = None,
        metadata_filename: str = "metadata.csv",
        cache_volumes: bool = True,
    ) -> None:
        resolved = root_dir or os.getenv("KNEE_MRI_ROOT")
        if not resolved:
            raise ValueError("Must provide root_dir or set KNEE_MRI_ROOT")
        super().__init__(resolved)
        self.metadata_filename = metadata_filename
        self.cache_volumes = cache_volumes
        self._meta_rows: Optional[List[Dict[str, int]]] = None
        self._meta_root: Optional[Path] = None
        self._volume_cache: Dict[str, np.ndarray] = {}
        self._path_cache: Dict[Tuple[str, str], Path] = {}

    def discover_records(self, root_dir: Optional[str] = None) -> List[Dict[str, int]]:
        root = Path(root_dir or self.root_dir or ".").resolve()
        meta_rows = self._load_metadata(root)
        records: List[Dict[str, int]] = []
        missing: List[str] = []
        for row in meta_rows:
            try:
                vol_path = self._resolve_volume_path(root, row["volume_filename"])
            except FileNotFoundError as exc:
                missing.append(row["volume_filename"])
                continue
            volume = self._load_volume(vol_path)
            depth = int(volume.shape[0])
            bbox = (
                row["roi_x"],
                row["roi_y"],
                row["roi_z"],
                row["roi_x1"],
                row["roi_y1"],
                row["roi_z1"],
            )
            for slice_idx in range(depth):
                records.append(
                    {
                        "filepath": str(vol_path),
                        "slice_idx": slice_idx,
                        "exam_id": row["exam_id"],
                        "series_no": row["series_no"],
                        "acl_diagnosis": row["acl_diagnosis"],
                        "knee_lr": row["knee_lr"],
                        "bbox": bbox,
                        "volume_filename": row["volume_filename"],
                    }
                )
        if missing:
            preview = ", ".join(missing[:5])
            if len(missing) > 5:
                preview += f", ... (+{len(missing) - 5} more)"
            print(f"[warn] Missing {len(missing)} volume file(s): {preview}")
        return records

    def load_record(self, record: Dict[str, int]) -> Dict[str, object]:
        path = Path(record["filepath"])
        volume = self._load_volume(path)
        slice_idx = int(record["slice_idx"])
        if slice_idx < 0 or slice_idx >= volume.shape[0]:
            raise IndexError(f"Slice index {slice_idx} out of range for {path.name}")

        img_slice = volume[slice_idx].astype(np.float32, copy=False)
        mask_slice = np.zeros_like(img_slice, dtype=np.uint8)

        x0, y0, z0, x1, y1, z1 = record["bbox"]
        depth, height, width = volume.shape[0], volume.shape[1], volume.shape[2]
        x0 = max(0, min(x0, height))
        y0 = max(0, min(y0, width))
        z0 = max(0, min(z0, depth))
        x1 = max(x0, min(x1, height))
        y1 = max(y0, min(y1, width))
        z1 = max(z0, min(z1, depth))

        if z0 <= slice_idx < z1:
            mask_slice[x0:x1, y0:y1] = 1

        meta = {
            "filepath": str(path),
            "slice_idx": slice_idx,
            "dataset": "kneemri",
            "exam_id": record["exam_id"],
            "series_no": record["series_no"],
            "acl_diagnosis": record["acl_diagnosis"],
            "knee_lr": record["knee_lr"],
            "roi": {
                "x": x0,
                "y": y0,
                "z": z0,
                "x1": x1,
                "y1": y1,
                "z1": z1,
            },
        }
        return {
            "image": img_slice,
            "mask": mask_slice,
            "label": int(record["acl_diagnosis"]),
            "meta": meta,
        }

    def _load_metadata(self, root: Path) -> List[Dict[str, int]]:
        if self._meta_rows is not None and self._meta_root == root:
            return self._meta_rows

        csv_path = root / self.metadata_filename
        if not csv_path.exists():
            raise FileNotFoundError(f"metadata.csv not found under {root}")

        rows: List[Dict[str, int]] = []
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for raw in reader:
                roi_x = int(raw["roiX"])
                roi_y = int(raw["roiY"])
                roi_z = int(raw["roiZ"])
                roi_h = int(raw["roiHeight"])
                roi_w = int(raw["roiWidth"])
                roi_d = int(raw["roiDepth"])
                row = {
                    "exam_id": int(raw["examId"]),
                    "series_no": int(raw["seriesNo"]),
                    "acl_diagnosis": int(raw["aclDiagnosis"]),
                    "knee_lr": int(raw["kneeLR"]),
                    "roi_x": roi_x,
                    "roi_y": roi_y,
                    "roi_z": roi_z,
                    "roi_h": roi_h,
                    "roi_w": roi_w,
                    "roi_d": roi_d,
                    "roi_x1": roi_x + roi_h,
                    "roi_y1": roi_y + roi_w,
                    "roi_z1": roi_z + roi_d,
                    "volume_filename": raw["volumeFilename"],
                }
                rows.append(row)

        self._meta_rows = rows
        self._meta_root = root
        return rows

    def _resolve_volume_path(self, root: Path, filename: str) -> Path:
        cache_key = (str(root), filename)
        if cache_key in self._path_cache:
            return self._path_cache[cache_key]

        direct = root / filename
        if direct.exists():
            path = direct
        else:
            matches = list(root.rglob(filename))
            if not matches:
                raise FileNotFoundError(f"Volume {filename} not found under {root}")
            path = matches[0]
        self._path_cache[cache_key] = path
        return path

    def _load_volume(self, path: Path) -> np.ndarray:
        key = str(path)
        if self.cache_volumes and key in self._volume_cache:
            return self._volume_cache[key]
        with path.open("rb") as fh:
            arr = pickle.load(fh)
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"Expected numpy array in {path.name}, got {type(arr)}")
        if arr.ndim != 3:
            raise ValueError(f"Volume {path.name} has invalid shape {arr.shape}")
        volume = np.asarray(arr)
        if self.cache_volumes:
            self._volume_cache[key] = volume
        return volume


__all__ = ["KneePckAdapter", "KneeMRIVolumeAdapter"]
