from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.adapters.fastmri_adapter import FastMRISinglecoilAdapter  # type: ignore
from src.main import build_preprocess  # type: ignore
from src.train.train_unet import UNet2DArgs, UNet2DTrainer  # type: ignore


def _default_dataset_root() -> Optional[Path]:
    for module_name in ("src.configs.config", "configs.config"):
        try:
            module = __import__(module_name, fromlist=("FASTMRI_ROOT",))
            value = getattr(module, "FASTMRI_ROOT", None)
            if value:
                return Path(value)
        except Exception:
            continue
    env = os.getenv("FASTMRI_ROOT")
    return Path(env) if env else None


def _split_ratio(value: str) -> float:
    try:
        ratio = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("split-ratio must be a float") from exc
    if not 0.0 < ratio < 1.0:
        raise argparse.ArgumentTypeError("split-ratio must be within (0, 1)")
    return ratio


def run_preprocess(
    dataset_root: Path,
    out_dir: Path,
    height: int,
    width: int,
    slice_keep: str,
    clip: str,
    use_n4: bool,
    use_denoise: bool,
    preview_max: int,
) -> int:
    adapter = FastMRISinglecoilAdapter(root_dir=str(dataset_root))
    args = SimpleNamespace(
        root_dir=str(dataset_root),
        out_dir=str(out_dir),
        height=height,
        width=width,
        slice_keep=slice_keep,
        clip=clip,
        use_n4=use_n4,
        use_denoise=use_denoise,
        preview_max=preview_max,
    )
    print(f"[step] Preprocess input volumes -> {out_dir}")
    results = build_preprocess(args, adapter=adapter)
    print(f"[done] Preprocess generated {len(results)} volume artefact(s)")
    return len(results)


def collect_npz(artifact_dir: Path) -> List[Path]:
    return sorted(p for p in artifact_dir.rglob("volume.npz") if p.is_file())


def generate_split(
    artifact_dir: Path,
    list_dir: Path,
    ratio: float,
    seed: int,
) -> Tuple[Path, Path]:
    npz_files = collect_npz(artifact_dir)
    if not npz_files:
        raise RuntimeError(f"No volume.npz files found under {artifact_dir}. Run preprocess first.")

    rng = random.Random(seed)
    rng.shuffle(npz_files)

    if len(npz_files) == 1:
        train_files = npz_files
        val_files: List[Path] = []
    else:
        cutoff = int(round(len(npz_files) * ratio))
        cutoff = max(1, min(cutoff, len(npz_files) - 1))
        train_files = npz_files[:cutoff]
        val_files = npz_files[cutoff:]

    list_dir.mkdir(parents=True, exist_ok=True)
    train_path = list_dir / "train.txt"
    val_path = list_dir / "val.txt"

    train_path.write_text("\n".join(str(p) for p in train_files), encoding="utf-8")
    val_path.write_text("\n".join(str(p) for p in val_files), encoding="utf-8")

    print(f"[step] Wrote train list ({len(train_files)} entries) -> {train_path}")
    print(f"[step] Wrote val list ({len(val_files)} entries) -> {val_path}")
    return train_path, val_path


def run_training(train_list: Path, val_list: Path, out_dir: Path, args: argparse.Namespace) -> None:
    train_args = UNet2DArgs(
        train_list=str(train_list),
        val_list=str(val_list),
        out_dir=str(out_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        workers=args.workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        loss=args.loss,
        model=args.model,
        encoder=args.encoder,
        encoder_weights=args.encoder_weights,
        aug=args.aug,
        imagenet_norm=args.imagenet_norm,
        k=args.k,
        classes=args.classes,
        logger=args.logger,
        save_val_probs=args.save_val_probs,
        max_grad_norm=args.max_grad_norm,
        amp=args.amp,
        seed=args.seed,
    )
    trainer = UNet2DTrainer(train_args)
    trainer.run()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess fastMRI volumes and launch U-Net training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=_default_dataset_root(),
        help="Path to raw fastMRI single-coil dataset. Required unless --skip-preprocess is set.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts") / "fastmri_knee",
        help="Directory where preprocess artefacts are stored.",
    )
    parser.add_argument(
        "--list-dir",
        type=Path,
        default=Path("lists"),
        help="Directory for generated train/val list files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs") / "fastmri_unet",
        help="Training output directory.",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip preprocess even if dataset-root is provided.",
    )
    parser.add_argument(
        "--skip-split",
        action="store_true",
        help="Skip generating train/val lists.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip launching U-Net training.",
    )
    parser.add_argument("--height", type=int, default=320, help="Preprocess output height.")
    parser.add_argument("--width", type=int, default=320, help="Preprocess output width.")
    parser.add_argument(
        "--slice-keep",
        default="0.3,0.7",
        help="Slice keep range as 'lo,hi'.",
    )
    parser.add_argument(
        "--clip",
        default="1.0,99.5",
        help="Clip percentiles as 'lo,hi'.",
    )
    parser.add_argument(
        "--preview-max",
        type=int,
        default=6,
        help="Number of preview PNG slices per volume.",
    )
    parser.add_argument("--use-n4", action="store_true", help="Enable N4 bias correction.")
    parser.add_argument("--use-denoise", action="store_true", help="Enable NL-means denoise.")
    parser.add_argument(
        "--split-ratio",
        type=_split_ratio,
        default=0.8,
        help="Train split ratio in (0, 1).",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed for train/val split.",
    )
    parser.add_argument(
        "--train-list",
        type=Path,
        help="Optional existing train list (overrides generated list).",
    )
    parser.add_argument(
        "--val-list",
        type=Path,
        help="Optional existing val list (overrides generated list).",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--workers", type=int, default=4, help="Number of data loader workers.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument(
        "--loss",
        default="dice_bce",
        choices=["dice_bce", "focal", "tversky", "focal_tversky", "dice_ce", "ce"],
        help="Loss function.",
    )
    parser.add_argument(
        "--model",
        default="unet",
        choices=["unet", "unetpp"],
        help="UNet variant.",
    )
    parser.add_argument("--encoder", default="resnet34", help="Encoder backbone identifier.")
    parser.add_argument(
        "--encoder-weights",
        default="none",
        help="Encoder pretrained weights tag (for example 'imagenet' or 'none').",
    )
    parser.add_argument(
        "--aug",
        default="light",
        choices=["none", "light", "medium"],
        help="Augmentation recipe.",
    )
    parser.add_argument(
        "--imagenet-norm",
        action="store_true",
        help="Apply ImageNet normalization to inputs.",
    )
    parser.add_argument("--k", type=int, default=1, help="Number of adjacent slices for 2.5D training.")
    parser.add_argument("--classes", type=int, default=1, help="Number of output classes.")
    parser.add_argument(
        "--logger",
        default="csv",
        choices=["noop", "csv"],
        help="Logger backend.",
    )
    parser.add_argument(
        "--save-val-probs",
        action="store_true",
        help="Persist validation probabilities to disk.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=5.0,
        help="Gradient clipping norm.",
    )
    parser.add_argument("--seed", type=int, default=2024, help="Training seed.")
    parser.add_argument(
        "--no-amp",
        dest="amp",
        action="store_false",
        help="Disable automatic mixed precision.",
    )
    parser.add_argument(
        "--amp",
        dest="amp",
        action="store_true",
        help="Enable automatic mixed precision.",
    )
    parser.set_defaults(amp=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    artifact_dir = Path(args.artifact_dir).resolve()
    list_dir = Path(args.list_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    dataset_root = Path(args.dataset_root).resolve() if args.dataset_root else None

    if not args.skip_preprocess:
        if dataset_root is None:
            parser.error("Provide --dataset-root or set FASTMRI_ROOT unless --skip-preprocess is used.")
        run_preprocess(
            dataset_root,
            artifact_dir,
            args.height,
            args.width,
            args.slice_keep,
            args.clip,
            args.use_n4,
            args.use_denoise,
            args.preview_max,
        )
    else:
        print("[step] Skipping preprocess step.")

    generated_train: Optional[Path] = None
    generated_val: Optional[Path] = None
    if not args.skip_split:
        generated_train, generated_val = generate_split(
            artifact_dir,
            list_dir,
            args.split_ratio,
            args.split_seed,
        )
    else:
        print("[step] Skipping train/val split generation.")

    train_list = (
        Path(args.train_list).resolve()
        if args.train_list
        else (generated_train if generated_train else list_dir / "train.txt")
    )
    val_list = (
        Path(args.val_list).resolve()
        if args.val_list
        else (generated_val if generated_val else list_dir / "val.txt")
    )

    if not train_list.exists():
        parser.error(f"Train list not found: {train_list}")
    if not val_list.exists():
        parser.error(f"Validation list not found: {val_list}")

    if args.skip_train:
        print("[step] Training skipped as requested.")
        return 0

    print(f"[step] Launching U-Net training -> {out_dir}")
    run_training(train_list, val_list, out_dir, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
