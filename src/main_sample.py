import os, json, argparse, pathlib
from typing import List, Dict
import numpy as np
import torch
from pprint import pprint

# ---- adapters & dataset wrapper
from adapters.fastmri_adapter import FastMRISinglecoilAdapter
from adapters.oai_zib_adapter import OaiZibAdapter
from adapters.kaggle_knee_pck_adapter import KaggleKneePckAdapter
from datasets.trainer_dataset import TrainerDataset
from src.preprocess.mri_preprocess import preprocess_records

# (tùy) nếu bạn đã có preprocessor, import vào đây
try:
    from preprocess.mri_preprocess import Preprocessor  # hoặc common_preprocessor.Preprocessor
except Exception:
    Preprocessor = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **k: x

try:
    from src.utils.seed import set_seed  # type: ignore
except Exception:
    def set_seed(*args, **kwargs): pass
try:
    from src.utils.logger import build_logger  # type: ignore
except Exception:
    def build_logger(*args, **kwargs): return None

import imageio.v2 as iio

#Adapter
def build_adapter(name: str, args) -> TrainerDataset:
    """
    Tạo adapter + TrainerDataset theo tên dataset.
    - Mặc định đọc root từ ENV; có thể override bằng --root.
    """
    name = name.lower()

    if name == "fastmri":
        # root: ENV FASTMRI_ROOT hoặc --root
        root = args.root or os.getenv("FASTMRI_ROOT")
        if not root:
            raise ValueError("Missing root for fastMRI. Set FASTMRI_ROOT or pass --root")

        adapter = FastMRISinglecoilAdapter(root_dir=root)  # adapter đã đọc env trong __init__ nếu bạn cấu hình như trước
        pre = None
        if args.with_preproc and Preprocessor:
            pre = Preprocessor(target_size=(320, 320), normalize="zscore")
        return TrainerDataset(adapter, preprocessor=pre)

    elif name == "oai":
        # 2 cách: dùng index npz (ENV OAI_ZIB_INDEX_NPZ) hoặc pattern glob (ENV OAI_ZIB_IMG_GLOB / OAI_ZIB_MSK_GLOB)
        index_npz = args.index or os.getenv("OAI_ZIB_INDEX_NPZ")
        if index_npz:
            adapter = OaiZibAdapter(index_npz=index_npz, slice_axis=args.slice_axis)
        else:
            img_glob = args.img_glob or os.getenv("OAI_ZIB_IMG_GLOB")
            msk_glob = args.msk_glob or os.getenv("OAI_ZIB_MSK_GLOB")
            if not (img_glob and msk_glob):
                raise ValueError("OAI-ZIB needs --img-glob & --mask-glob or set env OAI_ZIB_IMG_GLOB/OAI_ZIB_MSK_GLOB")
            adapter = OaiZibAdapter(image_glob=img_glob, mask_glob=msk_glob, slice_axis=args.slice_axis)

        pre = None
        if args.with_preproc and Preprocessor:
            pre = Preprocessor(target_size=(320, 320), normalize="zscore")
        return TrainerDataset(adapter, preprocessor=pre)

    elif name in ("kaggle", "kneepck", "kneemri"):
        # root: ENV KAGGLE_KNEE_PCK_ROOT hoặc --root
        root = args.root or os.getenv("KAGGLE_KNEE_PCK_ROOT")
        if not root:
            raise ValueError("Missing root for Kaggle .pck. Set KAGGLE_KNEE_PCK_ROOT or pass --root")

        adapter = KaggleKneePckAdapter()
        pre = None
        if args.with_preproc and Preprocessor:
            pre = Preprocessor(target_size=(256, 256), normalize="minmax")  # ví dụ khác fastMRI
        return TrainerDataset(adapter, root_dir=root, preprocessor=pre)

    else:
        raise ValueError(f"Unknown dataset name: {name}")


def preview(ds, n=3):
    print(f"Dataset size: {len(ds)}")
    for i in range(min(n, len(ds))):
        sample = ds[i]
        meta = sample.get("meta", {})
        # cố gắng in kích thước ảnh/mask nếu có
        img = sample.get("image")
        msk = sample.get("mask")
        shp_img = getattr(img, "shape", None)
        shp_msk = getattr(msk, "shape", None)
        print(f"[{i}] adapter={meta.get('adapter')} img={shp_img} mask={shp_msk} label={sample.get('label')}")
        # in meta rút gọn
        pprint({k: v for k, v in meta.items() if k not in ("adapter",)})


def parse_args_adapter():
    p = argparse.ArgumentParser(description="Adapter demo entrypoint")
    p.add_argument("--dataset", required=True, choices=["fastmri", "oai", "kaggle"],
                   help="Chọn adapter: fastmri | oai | kaggle")
    p.add_argument("--root", default=None, help="Override root dir (ưu tiên hơn ENV)")
    p.add_argument("--with-preproc", action="store_true", help="Gắn preprocessor nếu module có")
    # OAI-ZIB options
    p.add_argument("--index", default=None, help="Path tới index .npz (nếu dùng)")
    p.add_argument("--img-glob", default=None, help="Glob ảnh NIfTI (nếu không dùng index)")
    p.add_argument("--mask-glob", default=None, help="Glob mask NIfTI (nếu không dùng index)")
    p.add_argument("--slice-axis", type=int, default=2, help="Trục cắt slice cho OAI-ZIB (0/1/2)")
    return p.parse_args()

#End Adapter

#Preprocess
def group_records_by_file(records: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Gom các record theo filepath để xử lý từng volume (.h5).
    """
    buckets = {}
    for r in records:
        fp = r["filepath"]
        buckets.setdefault(fp, []).append(r)
    # đảm bảo thứ tự lát ổn định
    for fp in buckets:
        buckets[fp] = sorted(buckets[fp], key=lambda x: x["slice_idx"])
    return buckets

def save_pack(out_dir: str, pack: Dict, preview_max: int = 8):
    """
    Lưu các artefact tiền xử lý: tensor.pt, mask.npy, indices.json, metas.json, stats.json và một vài preview PNG.
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) tensor cho huấn luyện
    tensor: torch.Tensor = pack["tensor"]            # (S,1,H,W)
    torch.save(tensor, os.path.join(out_dir, "tensor.pt"))

    # 2) mask, indices, metas
    np.save(os.path.join(out_dir, "mask.npy"), pack["mask"])  # (S,H,W)
    with open(os.path.join(out_dir, "indices.json"), "w", encoding="utf-8") as f:
        json.dump(pack.get("indices", []), f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "metas.json"), "w", encoding="utf-8") as f:
        json.dump(pack.get("metas", []), f, ensure_ascii=False, indent=2)

    # 3) một số preview PNG để QA
    prev = pack["preview"]  # (S,H,W) ∈ [0,1]
    pv_dir = os.path.join(out_dir, "preview")
    os.makedirs(pv_dir, exist_ok=True)
    S = prev.shape[0]
    take = min(preview_max, S)
    for i in range(take):
        fname = f"slice_{pack['indices'][i]:03d}.png"
        iio.imwrite(os.path.join(pv_dir, fname), (prev[i] * 255).astype(np.uint8))

    # 4) thống kê QC nhanh (mean/std trong mask)
    img_z = tensor[:, 0].numpy()     # (S,H,W)
    mk = pack["mask"]                # (S,H,W)
    means, stds = [], []
    for s in range(img_z.shape[0]):
        vals = img_z[s][mk[s] > 0]
        if vals.size == 0:
            means.append(float("nan")); stds.append(float("nan"))
        else:
            means.append(float(vals.mean())); stds.append(float(vals.std()))
    stats = {
        "count_slices": int(S),
        "mean_in_mask_mean": float(np.nanmean(means)),
        "mean_in_mask_std": float(np.nanmean(stds)),
        "per_slice_mean": means[:50],  # log 50 lát đầu để gọn
        "per_slice_std": stds[:50],
    }
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

def parse_args_preprocess():
    ap = argparse.ArgumentParser(
        description="Preprocess fastMRI knee (single-coil) → tensor/preview/meta"
    )
    ap.add_argument("--root_dir", required=True, help="Thư mục chứa .h5 (single-coil)")
    ap.add_argument("--out_dir", required=True, help="Thư mục lưu artefact tiền xử lý")
    ap.add_argument("--height", type=int, default=320, help="Chiều cao đầu ra (H)")
    ap.add_argument("--width", type=int, default=320, help="Chiều rộng đầu ra (W)")
    ap.add_argument("--slice_keep", type=str, default="0.3,0.7",
                    help="Giữ dải lát giữa volume, ví dụ '0.3,0.7'")
    ap.add_argument("--use_n4", action="store_true", help="Bật N4 bias correction")
    ap.add_argument("--use_denoise", action="store_true", help="Bật NL-means denoise")
    ap.add_argument("--clip", type=str, default="1.0,99.5",
                    help="Percentile clip, ví dụ '1.0,99.5'")
    ap.add_argument("--preview_max", type=int, default=8,
                    help="Số preview PNG mỗi volume")
    return ap.parse_args()

# def build_preprocess(name: str, args)
# End Preprocess

# Training

# End TrainS



def main():
    args = parse_args_adapter()
    ds, adapter = build_adapter(args.dataset, args)
    preview(ds, n=3)  # xem nhanh vài mẫu; train loop để file train/*.py xử lý

    pargs = parse_args_preprocess()
    preprocess = build_preprocess(pargs, adapter=ds.adapter)
    print(preprocess)


if __name__ == "__main__":
    main()

