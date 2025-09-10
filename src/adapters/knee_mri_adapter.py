# src/adapters/kaggle_knee_pck_adapter.py
import os, glob, pickle
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
