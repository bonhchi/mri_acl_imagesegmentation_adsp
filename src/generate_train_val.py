# -*- coding: utf-8 -*-
"""
Generate train/val split lists from preprocessed volumes.
Usage:
    python generate_train_val.py
"""

import pathlib
import random

from src.utils.listing import format_list_entries, infer_dataset_name

ARTIFACT_DIR = pathlib.Path(r"D:\Master\ImageSegmentation\Demo\artifacts\fastmri_knee")
OUTPUT_DIR = pathlib.Path("lists")
SPLIT_RATIO = 0.8
ALL_FILE = pathlib.Path("all.txt")

print("[1/3] Scanning for volume.npz files...")
npz_files = list(ARTIFACT_DIR.rglob("volume.npz"))
print(f"  Found {len(npz_files)} files")

if not npz_files:
    print("[ERROR] No volume.npz found. Run preprocess first.")
    raise SystemExit(1)

label = infer_dataset_name(ARTIFACT_DIR)

ALL_FILE.write_text("\n".join(format_list_entries(npz_files, label)), encoding="utf-8")

print(f"[2/3] Splitting {SPLIT_RATIO:.0%} into train/val...")
paths = list(npz_files)
random.seed(42)
random.shuffle(paths)
cutoff = int(len(paths) * SPLIT_RATIO)
OUTPUT_DIR.mkdir(exist_ok=True)
train_lines = format_list_entries(paths[:cutoff], label)
val_lines = format_list_entries(paths[cutoff:], label)
(OUTPUT_DIR / "train.txt").write_text("\n".join(train_lines), encoding="utf-8")
(OUTPUT_DIR / "val.txt").write_text("\n".join(val_lines), encoding="utf-8")
print(f"[OK] Train: {len(train_lines)} files | Val: {len(val_lines)} files")

print(f"[3/3] Lists saved in: {OUTPUT_DIR.resolve()}")
print("Done.")
