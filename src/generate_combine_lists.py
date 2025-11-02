#!/usr/bin/env python3
"""
Generate combined train/val list files across multiple MRI datasets.

Example:
    python src/generate_combine_lists.py
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys

# Reuse existing helpers for consistent dataset labelling.
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.append(project_root_str)

from src.utils.listing import format_list_entries  # type: ignore

DEFAULT_ARTIFACTS = {
    "fastmri": PROJECT_ROOT / "artifacts" / "fastmri_knee",
    "kneemri": PROJECT_ROOT / "artifacts" / "kneemri_acl",
    "oaizib": PROJECT_ROOT / "artifacts" / "oaizib_knee",
}


@dataclass
class DatasetSplit:
    name: str
    train_files: List[Path]
    val_files: List[Path]


def parse_overrides(values: Iterable[str]) -> Dict[str, Path]:
    overrides: Dict[str, Path] = {}
    for item in values:
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                f"Invalid override '{item}'. Expected format 'dataset=path'."
            )
        key, raw_path = item.split("=", 1)
        clean_key = key.strip().lower()
        if not clean_key:
            raise argparse.ArgumentTypeError(
                f"Invalid override '{item}'. Dataset name cannot be empty."
            )
        path = Path(raw_path.strip()).expanduser().resolve()
        overrides[clean_key] = path
    return overrides


def discover_volumes(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Artifact directory not found: {root}")
    files = sorted(p for p in root.rglob("volume.npz") if p.is_file())
    if not files:
        raise RuntimeError(f"No volume.npz found under {root}")
    return files


def split_files(files: List[Path], ratio: float, rng: random.Random) -> Tuple[List[Path], List[Path]]:
    pool = list(files)
    rng.shuffle(pool)
    if len(pool) <= 1:
        return pool, []
    cutoff = int(round(len(pool) * ratio))
    cutoff = max(1, min(cutoff, len(pool) - 1))
    return pool[:cutoff], pool[cutoff:]


def build_splits(
    datasets: Iterable[str],
    artifact_map: Dict[str, Path],
    ratio: float,
    seed: int,
) -> List[DatasetSplit]:
    rng = random.Random(seed)
    splits: List[DatasetSplit] = []
    for name in datasets:
        root = artifact_map[name]
        volumes = discover_volumes(root)
        train_files, val_files = split_files(volumes, ratio, rng)
        splits.append(DatasetSplit(name=name, train_files=train_files, val_files=val_files))
    return splits


def write_list(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate combined train/val lists for multiple datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        choices=sorted(DEFAULT_ARTIFACTS.keys()),
        help="Datasets to include. Defaults to all.",
    )
    parser.add_argument(
        "--artifact",
        dest="artifact_overrides",
        action="append",
        default=[],
        help="Override artifact directory using 'dataset=path'. Can be repeated.",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.8,
        help="Train split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for shuffling.",
    )
    parser.add_argument(
        "--train-out",
        type=Path,
        default=Path("lists") / "train_combine.txt",
        help="Output path for combined train list.",
    )
    parser.add_argument(
        "--val-out",
        type=Path,
        default=Path("lists") / "val_combine.txt",
        help="Output path for combined val list.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print dataset counts after generation.",
    )

    args = parser.parse_args(argv)

    if not 0.0 < args.ratio < 1.0:
        parser.error("ratio must be within (0, 1).")

    artifact_map = DEFAULT_ARTIFACTS.copy()
    if args.artifact_overrides:
        artifact_map.update(parse_overrides(args.artifact_overrides))

    selected = [ds.lower() for ds in args.datasets] if args.datasets else sorted(artifact_map.keys())
    missing = [name for name in selected if name not in artifact_map]
    if missing:
        parser.error(f"Unknown dataset keys: {', '.join(missing)}")

    splits = build_splits(selected, artifact_map, args.ratio, args.seed)

    rng = random.Random(args.seed)
    train_lines: List[str] = []
    val_lines: List[str] = []
    summary: List[str] = []

    for split in splits:
        train_lines.extend(format_list_entries(split.train_files, split.name))
        val_lines.extend(format_list_entries(split.val_files, split.name))
        summary.append(
            f"{split.name}: train={len(split.train_files)} val={len(split.val_files)} total={len(split.train_files) + len(split.val_files)}"
        )

    rng.shuffle(train_lines)
    rng.shuffle(val_lines)

    train_out = args.train_out.resolve()
    val_out = args.val_out.resolve()
    write_list(train_out, train_lines)
    write_list(val_out, val_lines)

    if args.summary:
        print("[combine] datasets:", ", ".join(selected))
        for line in summary:
            print("  -", line)
        print(f"[combine] train list -> {train_out}")
        print(f"[combine] val list   -> {val_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
