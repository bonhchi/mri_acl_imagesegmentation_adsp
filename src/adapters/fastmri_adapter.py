import os, glob, h5py, numpy as np
from .base_adapter import BaseAdapter

class FastMRISinglecoilAdapter(BaseAdapter):
    def __init__(self, root_dir=None, env_key="FASTMRI_ROOT"):
        # Nếu không truyền root_dir thì lấy từ biến môi trường
        self.root_dir = root_dir or os.getenv(env_key)
        if not self.root_dir:
            raise ValueError(f"Must provide root_dir or set env {env_key}")
        
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
        target = None
        target_key = None

        with h5py.File(fp, "r") as hf:
            kspace = np.asarray(hf["kspace"][s])
            target = None
            target_key = None
            
            for cand in ["reconstruction_rss", "reconstruction_esc", "reconstruction"]:
                if cand in hf:
                    target = np.asarray(hf[cand][s])
                    target_key = cand
                    break
        return {
            "image": None,
            "mask": None,
            "label": None,
            "kspace": kspace,
            "target": target,
            "meta": {
                "filepath": fp, 
                "slice_idx": s, 
                "dataset": "fastmri",
                "target_key": target_key,
                "adapter": "fastmri_singlecoil-h5"}
        }
    