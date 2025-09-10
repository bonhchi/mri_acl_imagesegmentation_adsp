import h5py
import numpy as np
from typing import Dict, Any, List, Optional
from .base_adapter import BaseAdapter  # như bạn đã có
from src.utils.kspace import ifft2c, complex_abs, center_crop_or_pad

class FastMRISinglecoilAdapter(BaseAdapter):
    def discover_records(self, root_dir):
        records = []
        files = sorted(glob.glob(os.path.join(root_dir, "*.h5")))
        for fp in files:
            with h5py.File(fp, "r") as hf:
                num_slices = hf["kspace"].shape[0]
            for s in range(num_slices):
                records.append({"filepath": fp, "slice_idx": s})
        return records

    def load_record(self, record):
        fp, s = record["filepath"], record["slice_idx"]
        with h5py.File(fp, "r") as hf:
            kspace = hf["kspace"][s]
            target = None
            for cand in ["reconstruction_rss", "reconstruction_esc", "reconstruction"]:
                if cand in hf:
                    target = hf[cand][s]
                    break
        return {
            "kspace": kspace,
            "target": target,
            "meta": 
            {"filepath": fp, "slice_idx": s, "dataset": "fastmri"}
        }