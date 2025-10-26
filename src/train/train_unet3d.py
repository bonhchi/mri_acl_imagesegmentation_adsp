# -*- coding: utf-8 -*-
"""
Training entrypoint for 3D U-Net models on preprocessed volume artefacts.

The pipeline consumes the ``volume.npz`` files produced by the preprocessing step.
Each NPZ is expected to expose:
    - ``img`` : ndarray shaped (S, 1, H, W)   (float32)
    - ``msk`` : ndarray shaped (S,   H, W)    (uint8 or int)

We reshape volumes to channel-first tensors (C, D, H, W) and train with random
3D patches for efficiency. Validation uses a deterministic sliding-window grid.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from monai.losses import DiceCELoss
from monai.networks.nets import UNet
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _read_list(txt_path: str) -> List[str]:
    with open(txt_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sanitize_tag(tag: str) -> str:
    return tag.replace(" ", "_").replace("/", "-")


def _compute_starts(dim: int, window: int, step: int) -> List[int]:
    if dim <= window:
        return [0]
    starts = list(range(0, dim - window + 1, max(1, step)))
    final = dim - window
    if starts[-1] != final:
        starts.append(final)
    return starts


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class VolumePatchDataset3D(Dataset):
    """
    Patch-based dataset for 3D volumes stored in preprocessing artefacts.
    """

    def __init__(
        self,
        list_txt: str,
        *,
        patch_size: Tuple[int, int, int],
        classes: int,
        mode: str,
        patches_per_volume: int = 12,
        pos_fraction: float = 0.5,
        eval_stride: Optional[Tuple[int, int, int]] = None,
        cache_volumes: bool = True,
        normalize: str = "volume",
    ) -> None:
        self.files = _read_list(list_txt)
        if not self.files:
            raise ValueError(f"No NPZ files found in {list_txt}")

        self.patch_size = tuple(int(x) for x in patch_size)
        self.classes = max(int(classes), 1)
        self.mode = mode
        self.patches_per_volume = max(1, int(patches_per_volume))
        self.pos_fraction = float(np.clip(pos_fraction, 0.0, 1.0))
        self.eval_stride = (
            tuple(int(max(1, s)) for s in eval_stride) if eval_stride is not None else None
        )
        self.cache_volumes = bool(cache_volumes)
        self.normalize = normalize.lower()

        self._cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self._indices: List[Tuple[int, Tuple[int, int, int]]] = []

        if self.mode != "train":
            self._indices = self._build_eval_index()

    # ------------------------------------------------------------------ #
    # Core dataset protocol
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        if self.mode == "train":
            return len(self.files) * self.patches_per_volume
        return len(self._indices)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "train":
            file_idx = index // self.patches_per_volume
            start = self._sample_train_start(file_idx)
        else:
            file_idx, start = self._indices[index]

        img, mask = self._load_volume(file_idx)
        patch_img, patch_mask = self._extract_patch(img, mask, start)
        x = torch.from_numpy(patch_img.copy()).float()
        if self.classes == 1:
            y = torch.from_numpy((patch_mask > 0).astype(np.float32)).unsqueeze(0)
        else:
            y = torch.from_numpy(patch_mask.astype(np.int64))
        return x, y

    # ------------------------------------------------------------------ #
    # Volume handling
    # ------------------------------------------------------------------ #
    def _load_volume(self, file_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.cache_volumes and file_idx in self._cache:
            return self._cache[file_idx]

        path = Path(self.files[file_idx])
        with np.load(path) as data:
            img = data["img"].astype(np.float32)  # (S,1,H,W)
            mask = data["msk"].astype(np.int16)   # (S,H,W)

        img = np.moveaxis(img, 0, 1)  # -> (1, S, H, W)
        if self.normalize == "volume":
            mu = float(img.mean())
            sigma = float(img.std()) + 1e-6
            img = (img - mu) / sigma
        elif self.normalize == "minmax":
            lo = float(img.min())
            hi = float(img.max())
            img = (img - lo) / (hi - lo + 1e-6)
        # else: "none" -> keep as-is (already roughly z-scored by preprocess)

        if self.cache_volumes:
            self._cache[file_idx] = (img, mask)
        return img, mask

    # ------------------------------------------------------------------ #
    # Patch sampling utilities
    # ------------------------------------------------------------------ #
    def _sample_train_start(self, file_idx: int) -> Tuple[int, int, int]:
        img, mask = self._load_volume(file_idx)
        depth, height, width = mask.shape
        pd, ph, pw = self.patch_size

        if self.pos_fraction > 0.0 and mask.max() > 0 and random.random() < self.pos_fraction:
            coords = np.argwhere(mask > 0)
            center = coords[np.random.randint(len(coords))]
        else:
            center = np.array(
                [
                    np.random.randint(0, max(depth, 1)),
                    np.random.randint(0, max(height, 1)),
                    np.random.randint(0, max(width, 1)),
                ]
            )

        start = center - np.array(self.patch_size) // 2
        start = np.maximum(start, 0)
        max_start = np.maximum(np.array([depth, height, width]) - np.array(self.patch_size), 0)
        start = np.minimum(start, max_start)
        return int(start[0]), int(start[1]), int(start[2])

    def _extract_patch(
        self, img: np.ndarray, mask: np.ndarray, start: Tuple[int, int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        z0, y0, x0 = start
        pd, ph, pw = self.patch_size
        z1 = min(z0 + pd, img.shape[1])
        y1 = min(y0 + ph, img.shape[2])
        x1 = min(x0 + pw, img.shape[3])

        patch_img = np.zeros((img.shape[0], pd, ph, pw), dtype=img.dtype)
        patch_mask = np.zeros((pd, ph, pw), dtype=mask.dtype)

        patch_img[:, : z1 - z0, : y1 - y0, : x1 - x0] = img[:, z0:z1, y0:y1, x0:x1]
        patch_mask[: z1 - z0, : y1 - y0, : x1 - x0] = mask[z0:z1, y0:y1, x0:x1]
        return patch_img, patch_mask

    def _build_eval_index(self) -> List[Tuple[int, Tuple[int, int, int]]]:
        if self.eval_stride is None:
            stride = tuple(max(1, p // 2) for p in self.patch_size)
        else:
            stride = self.eval_stride

        indices: List[Tuple[int, Tuple[int, int, int]]] = []
        for fi in range(len(self.files)):
            img, mask = self._load_volume(fi)
            depth, height, width = mask.shape
            starts_z = _compute_starts(depth, self.patch_size[0], stride[0])
            starts_y = _compute_starts(height, self.patch_size[1], stride[1])
            starts_x = _compute_starts(width, self.patch_size[2], stride[2])
            for z0 in starts_z:
                for y0 in starts_y:
                    for x0 in starts_x:
                        indices.append((fi, (z0, y0, x0)))
        return indices


# ---------------------------------------------------------------------------
# Model helper
# ---------------------------------------------------------------------------


def build_unet3d(
    in_channels: int,
    out_channels: int,
    channels: Sequence[int],
) -> UNet:
    strides = (2,) * (len(channels) - 1)
    return UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=tuple(channels),
        strides=strides,
        num_res_units=2,
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class UNet3DConfig:
    train_list: str
    val_list: str
    out_dir: str = "runs/unet3d"
    run_tag: str = ""
    patch_size: Tuple[int, int, int] = (160, 160, 64)
    patches_per_volume: int = 12
    pos_fraction: float = 0.5
    eval_stride: Tuple[int, int, int] = (80, 80, 32)
    channels: Tuple[int, ...] = (32, 64, 128, 256, 320)
    classes: int = 1
    batch_size: int = 2
    val_batch_size: int = 1
    epochs: int = 80
    lr: float = 1e-3
    weight_decay: float = 1e-4
    workers: int = 4
    amp: bool = True
    seed: int = 2024
    cache_volumes: bool = True
    grad_clip: float = 5.0
    normalize: str = "volume"


def parse_args(argv: Optional[Sequence[str]] = None) -> UNet3DConfig:
    parser = argparse.ArgumentParser("Train 3D U-Net on preprocessed volumes")
    parser.add_argument("--train-list", required=True, help="Path to train.txt listing volume NPZ files.")
    parser.add_argument("--val-list", required=True, help="Path to val.txt listing volume NPZ files.")
    parser.add_argument("--out-dir", default="runs/unet3d", help="Base directory for experiment runs.")
    parser.add_argument("--run-tag", default="", help="Optional suffix appended to the run directory name.")
    parser.add_argument("--patch-size", type=int, nargs=3, default=[160, 160, 64], metavar=("D", "H", "W"))
    parser.add_argument("--patches-per-volume", type=int, default=12, help="Number of random patches sampled per volume each epoch.")
    parser.add_argument("--pos-frac", type=float, default=0.5, help="Probability of sampling a positive patch when lesions exist.")
    parser.add_argument("--eval-overlap", type=float, default=0.5, help="Fractional overlap (0..0.9) used to build validation sliding window.")
    parser.add_argument("--channels", type=int, nargs="+", default=[32, 64, 128, 256, 320], help="Encoder feature widths.")
    parser.add_argument("--classes", type=int, default=1, help="Number of output classes.")
    parser.add_argument("--batch-size", type=int, default=2, help="Training batch size.")
    parser.add_argument("--val-batch-size", type=int, default=1, help="Validation batch size.")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--workers", type=int, default=4, help="Number of DataLoader workers.")
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision.")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed.")
    parser.add_argument("--cache-volumes", dest="cache_volumes", action="store_true", help="Cache volumes in RAM for faster patch sampling.")
    parser.add_argument("--no-cache-volumes", dest="cache_volumes", action="store_false", help="Disable volume caching.")
    parser.add_argument("--normalize", choices=["volume", "minmax", "none"], default="volume", help="Additional per-volume normalisation.")
    parser.add_argument("--grad-clip", type=float, default=5.0, help="Gradient clipping norm (<=0 to disable).")
    parser.set_defaults(amp=True, cache_volumes=True)
    args = parser.parse_args(argv)

    overlap = float(np.clip(args.eval_overlap, 0.0, 0.9))
    patch = tuple(int(p) for p in args.patch_size)
    stride = tuple(max(1, int(round(p * (1.0 - overlap)))) for p in patch)

    return UNet3DConfig(
        train_list=args.train_list,
        val_list=args.val_list,
        out_dir=args.out_dir,
        run_tag=args.run_tag,
        patch_size=patch,
        patches_per_volume=args.patches_per_volume,
        pos_fraction=args.pos_frac,
        eval_stride=stride,
        channels=tuple(args.channels),
        classes=max(1, args.classes),
        batch_size=max(1, args.batch_size),
        val_batch_size=max(1, args.val_batch_size),
        epochs=max(1, args.epochs),
        lr=args.lr,
        weight_decay=args.weight_decay,
        workers=max(0, args.workers),
        amp=bool(args.amp),
        seed=args.seed,
        cache_volumes=bool(args.cache_volumes),
        grad_clip=args.grad_clip,
        normalize=args.normalize,
    )


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class UNet3DTrainer:
    def __init__(self, cfg: UNet3DConfig) -> None:
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._set_seed(cfg.seed)

        self.out_dir = self._prepare_run_directory()
        self.cfg.out_dir = str(self.out_dir)
        self._dump_config()

        self.train_ds = VolumePatchDataset3D(
            cfg.train_list,
            patch_size=cfg.patch_size,
            classes=cfg.classes,
            mode="train",
            patches_per_volume=cfg.patches_per_volume,
            pos_fraction=cfg.pos_fraction,
            cache_volumes=cfg.cache_volumes,
            normalize=cfg.normalize,
        )
        self.val_ds = VolumePatchDataset3D(
            cfg.val_list,
            patch_size=cfg.patch_size,
            classes=cfg.classes,
            mode="val",
            eval_stride=cfg.eval_stride,
            cache_volumes=cfg.cache_volumes,
            normalize=cfg.normalize,
        )

        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.workers,
            pin_memory=True,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=cfg.val_batch_size,
            shuffle=False,
            num_workers=cfg.workers,
            pin_memory=True,
        )

        out_channels = 1 if cfg.classes == 1 else cfg.classes
        self.model = build_unet3d(
            in_channels=1,
            out_channels=out_channels,
            channels=cfg.channels,
        ).to(self.device)

        if cfg.classes == 1:
            self.criterion = DiceCELoss(sigmoid=True, to_onehot_y=False)
        else:
            self.criterion = DiceCELoss(to_onehot_y=True, softmax=True)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            patience=6,
            factor=0.5,
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

        self.best_metric = float("inf")
        self.best_snapshot: Dict[str, Any] = {}
        self.best_ckpt = self.out_dir / "best.pt"
        self.history: List[Dict[str, Any]] = []

        self.log_path = self.out_dir / "history.csv"
        if not self.log_path.exists():
            with self.log_path.open("w", encoding="utf-8") as fh:
                fh.write("epoch,train_loss,train_dice,val_loss,val_dice,lr\n")

    # ------------------------------------------------------------------ #
    # Training utilities
    # ------------------------------------------------------------------ #
    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _prepare_run_directory(self) -> Path:
        base = _ensure_dir(Path(self.cfg.out_dir))
        day_stamp = datetime.now().strftime("%Y-%m-%d")
        run_name = f"{day_stamp}_unet3d"
        if self.cfg.run_tag:
            run_name = f"{run_name}_{_sanitize_tag(self.cfg.run_tag)}"
        run_dir = base / run_name
        suffix = 2
        while run_dir.exists():
            run_dir = base / f"{run_name}_{suffix:02d}"
            suffix += 1
        return _ensure_dir(run_dir)

    def _dump_config(self) -> None:
        with (self.out_dir / "args.json").open("w", encoding="utf-8") as fh:
            json.dump(asdict(self.cfg), fh, indent=2)

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #
    def _batch_dice(self, logits: torch.Tensor, target: torch.Tensor) -> float:
        eps = 1e-6
        if self.cfg.classes == 1:
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            target = target.float()
            dims = tuple(range(1, preds.ndim))
            inter = (preds * target).sum(dim=dims)
            den = preds.sum(dim=dims) + target.sum(dim=dims)
            dice = (2 * inter + eps) / (den + eps)
            return float(dice.mean().item())

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        dices: List[torch.Tensor] = []
        for c in range(1, self.cfg.classes):
            pred_c = (preds == c).float()
            target_c = (target == c).float()
            dims = tuple(range(1, pred_c.ndim))
            inter = (pred_c * target_c).sum(dim=dims)
            den = pred_c.sum(dim=dims) + target_c.sum(dim=dims)
            dices.append((2 * inter + eps) / (den + eps))
        if not dices:
            return 1.0
        stacked = torch.stack(dices, dim=0).mean()
        return float(stacked.mean().item())

    # ------------------------------------------------------------------ #
    # Epoch loops
    # ------------------------------------------------------------------ #
    def _train_one_epoch(self) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        total_dice = 0.0
        for x, y in self.train_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.cfg.amp):
                logits = self.model(x)
                loss = self.criterion(logits, y)
            self.scaler.scale(loss).backward()
            if self.cfg.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += float(loss.item())
            total_dice += self._batch_dice(logits.detach(), y)
        n = max(1, len(self.train_loader))
        return total_loss / n, total_dice / n

    @torch.no_grad()
    def _validate(self) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        for x, y in self.val_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            with torch.cuda.amp.autocast(enabled=self.cfg.amp):
                logits = self.model(x)
                loss = self.criterion(logits, y)
            total_loss += float(loss.item())
            total_dice += self._batch_dice(logits, y)
        n = max(1, len(self.val_loader))
        return total_loss / n, total_dice / n

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def fit(self) -> Dict[str, Any]:
        for epoch in range(1, self.cfg.epochs + 1):
            train_loss, train_dice = self._train_one_epoch()
            val_loss, val_dice = self._validate()
            lr = float(self.optimizer.param_groups[0]["lr"])
            self.scheduler.step(val_loss)

            print(
                f"Epoch {epoch:03d}/{self.cfg.epochs} | "
                f"train {train_loss:.4f} | val {val_loss:.4f} | "
                f"dice {val_dice:.4f} | lr {lr:.2e}"
            )

            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(f"{epoch},{train_loss:.6f},{train_dice:.6f},{val_loss:.6f},{val_dice:.6f},{lr:.6e}\n")

            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_dice": train_dice,
                "val_loss": val_loss,
                "val_dice": val_dice,
                "lr": lr,
            }
            self.history.append(record)

            if val_loss < self.best_metric:
                self.best_metric = val_loss
                self.best_snapshot = record
                torch.save({"model": self.model.state_dict(), "config": asdict(self.cfg)}, self.best_ckpt)
                print("  >> saved new best checkpoint")

        with (self.out_dir / "history.json").open("w", encoding="utf-8") as fh:
            json.dump(self.history, fh, indent=2)

    def summary(self) -> Dict[str, Any]:
        return {
            "best_loss": float(self.best_metric),
            "best_epoch": int(self.best_snapshot.get("epoch", -1)),
            "best_ckpt": str(self.best_ckpt),
            "history_len": len(self.history),
        }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    cfg = parse_args(argv)
    trainer = UNet3DTrainer(cfg)
    trainer.fit()
    info = trainer.summary()
    print("Done. Best checkpoint:", info["best_ckpt"])
    return info


if __name__ == "__main__":
    main()

