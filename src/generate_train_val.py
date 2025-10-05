"""
Generate train/val split lists from preprocessed fastMRI volumes.
Author: Phu Tran
Usage:
    python generate_train_val.py
"""

import pathlib
import random

# === CẤU HÌNH ===
ARTIFACT_DIR = pathlib.Path(r"D:\Master\ImageSegmentation\Demo\artifacts\fastmri_knee")  # Thư mục chứa các volume.npz
OUTPUT_DIR = pathlib.Path("lists")                     # Nơi lưu train.txt & val.txt
SPLIT_RATIO = 0.8                                      # 80% train / 20% val
ALL_FILE = pathlib.Path("all.txt")                     # File tạm danh sách tổng

# === TÌM CÁC FILE .NPZ ===
print("[1/3] Scanning for volume.npz files...")
npz_files = list(ARTIFACT_DIR.rglob("volume.npz"))
print(f"  → Found {len(npz_files)} files")

if not npz_files:
    print("[ERROR] No volume.npz found. Run preprocess step first.")
    raise SystemExit(1)

# === GHI DANH SÁCH TỔNG ===
ALL_FILE.write_text("\n".join(str(f) for f in npz_files), encoding="utf-8")

# === CHIA TRAIN/VAL ===
print("[2/3] Splitting 80/20 into train/val...")
L = [str(f) for f in npz_files]
random.seed(42)
random.shuffle(L)
k = int(len(L) * SPLIT_RATIO)

OUTPUT_DIR.mkdir(exist_ok=True)
(pathlib.Path(OUTPUT_DIR) / "train.txt").write_text("\n".join(L[:k]), encoding="utf-8")
(pathlib.Path(OUTPUT_DIR) / "val.txt").write_text("\n".join(L[k:]), encoding="utf-8")

print(f"[OK] Train: {k} files | Val: {len(L) - k} files")

# === HOÀN TẤT ===
print(f"[3/3] Lists saved in: {OUTPUT_DIR.resolve()}")
print("Done ✅")
