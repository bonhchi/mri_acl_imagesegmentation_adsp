import os, json, argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
from pprint import pprint

# ---- adapters & dataset wrapper
from adapters.fastmri_adapter import FastMRISinglecoilAdapter
from adapters.base_adapter import BaseAdapter
from datasets.trainer_dataset import TrainerDataset
from src.preprocess.mri_preprocess import MRIKneePreprocessor, preprocess_records

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
def build_adapter(name: str, args) -> Tuple[TrainerDataset, BaseAdapter]:
    """
    FastMRI-only adapter factory used during the demo.
    - Root is taken from FASTMRI_ROOT env or --root CLI override.
    """
    if name.lower() != "fastmri":
        raise ValueError("Demo only supports the fastMRI single-coil dataset.")

    root = args.root or os.getenv("FASTMRI_ROOT")
    if not root:
        raise ValueError("Missing root for fastMRI. Set FASTMRI_ROOT or pass --root")

    adapter = FastMRISinglecoilAdapter(root_dir=root)
    pre = None
    if args.with_preproc and Preprocessor:
        pre = Preprocessor(target_size=(320, 320), normalize="zscore")
    dataset = TrainerDataset(adapter, preprocessor=pre)
    return dataset, adapter



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


def parse_args_adapter(argv: Optional[Sequence[str]] = None):
    p = argparse.ArgumentParser(description="Adapter demo entrypoint (fastMRI only)")
    p.add_argument("--dataset", required=True, choices=["fastmri"],
                   help="Chỉ hỗ trợ fastMRI single-coil")
    p.add_argument("--root", default=None, help="Override root dir (ưu tiên hơn FASTMRI_ROOT)")
    p.add_argument("--with-preproc", action="store_true", help="Gắn preprocessor nếu module có")
    args, remaining = p.parse_known_args(argv)
    return args, remaining

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

def parse_args_preprocess(argv: Optional[Sequence[str]] = None):
    if not argv:
        return None
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
    return ap.parse_args(argv)


def _parse_pair(value: str, name: str) -> Tuple[float, float]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"{name} phải ở dạng 'lo,hi'")
    try:
        lo, hi = map(float, parts)
    except ValueError as exc:
        raise ValueError(f"{name} phải gồm 2 số thực, nhận {value!r}") from exc
    return lo, hi

def build_preprocess(args, adapter: BaseAdapter):
    """Run MRI preprocessing for all volumes discovered by the adapter."""
    slice_keep = _parse_pair(args.slice_keep, "slice_keep")
    clip = _parse_pair(args.clip, "clip")
    preprocessor = MRIKneePreprocessor(
        out_size=(args.height, args.width),
        slice_keep=slice_keep,
        clip_percentiles=clip,
        use_n4=args.use_n4,
        use_denoise=args.use_denoise,
    )
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    try:
        records = adapter.discover_records(args.root_dir)
    except TypeError:
        records = adapter.discover_records()
    if not records:
        return []
    grouped = group_records_by_file(records)
    volume_items = list(grouped.items())
    summary = []
    for filepath, record_defs in tqdm(volume_items, desc="Preprocess volumes"):
        loaded = [adapter.load_record(rec) for rec in record_defs]
        pack = preprocess_records(loaded, preprocessor=preprocessor)
        out_dir = out_root / Path(filepath).stem
        save_pack(str(out_dir), pack, preview_max=args.preview_max)
        summary.append({
            "filepath": filepath,
            "output_dir": str(out_dir),
            "num_slices": int(pack["tensor"].shape[0]),
        })
    return summary

# End Preprocess

# Training

# End TrainS



def main():
    """Pipeline demo cho fastMRI: adapter preview -> preprocess artefact."""
def main():
    """FastMRI demo pipeline: adapter preview -> preprocessing artefacts."""
    # Step 1: parse adapter arguments (fastMRI only) and build the dataset wrapper
    adapter_args, remaining = parse_args_adapter()
    dataset, adapter = build_adapter(adapter_args.dataset, adapter_args)
    # Step 2: preview a few slices to sanity-check the adapter output
    preview(dataset, n=3)
    # Step 3: parse preprocessing arguments (if extra CLI flags were provided)
    preprocess_args = parse_args_preprocess(remaining)
    if preprocess_args is None:
        # No preprocessing args -> stop after preview
        return
    # Step 4: run preprocessing and persist tensor/preview/meta artefacts
    results = build_preprocess(preprocess_args, adapter=adapter)
    if not results:
        print("No volume matched the preprocessing filters.")
        return
    print(f"Preprocess finished for {len(results)} volume(s), stored at {preprocess_args.out_dir}")
    # Follow up: run src/train/train_unet.py to train and collect experiment metrics
if __name__ == "__main__":
    main()

