# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
from __future__ import annotations
import json, argparse, numpy as np, torch, os
from time import time
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
from torch.utils.data import DataLoader

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.dataio.datasets import KneeNPZ2DSlices
from src.models.unet_factory import build_unet

# Loss: ưu tiên LossManager mới; fallback build_loss cũ nếu bạn chưa đổi
try:
    from src.train.losses import LossManager as _LossManager
    def _build_loss(classes: int, name: str):
        return _LossManager(classes=classes, name=name)
except Exception:
    from src.train.losses import build_loss as _build_loss

# Logger adapters bạn đã có
from src.train.log_adapter import make_logger

# Engine (class) đã refactor
from src.train.engine import Engine


def set_seed(s: int):
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DevicePrefetchLoader:
    """Wrap a DataLoader to pre-stage batches on a target device."""

    def __init__(self, loader: DataLoader, device: torch.device) -> None:
        self.loader = loader
        self.device = device

    def __len__(self) -> int:
        return len(self.loader)

    @property
    def dataset(self):
        return self.loader.dataset

    def __getattr__(self, item):
        return getattr(self.loader, item)

    def __iter__(self):
        if self.device.type != "cuda":
            yield from self.loader
            return

        stream = torch.cuda.Stream()
        iterator = iter(self.loader)

        def _to_device(batch):
            if isinstance(batch, (list, tuple)):
                return type(batch)(
                    item.to(self.device, non_blocking=True) if torch.is_tensor(item) else item
                    for item in batch
                )
            if isinstance(batch, dict):
                return {
                    k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v)
                    for k, v in batch.items()
                }
            if torch.is_tensor(batch):
                return batch.to(self.device, non_blocking=True)
            return batch

        try:
            next_batch = next(iterator)
        except StopIteration:
            return

        with torch.cuda.stream(stream):
            next_batch = _to_device(next_batch)

        while True:
            torch.cuda.current_stream().wait_stream(stream)
            current = next_batch
            try:
                next_batch = next(iterator)
            except StopIteration:
                next_batch = None
            if next_batch is not None:
                with torch.cuda.stream(stream):
                    next_batch = _to_device(next_batch)
            yield current
            if next_batch is None:
                break


@dataclass
class UNet2DArgs:
    # data/model
    train_list: str = ""
    val_list: str = ""
    out_dir: str = "runs/unet2d"
    run_tag: str = ""
    k: int = 1
    aug: str = "light"                       # none|light|medium
    model: str = "unet"                      # unet|unetpp
    encoder: str = "resnet34"
    encoder_weights: str = "none"
    classes: int = 1
    imagenet_norm: bool = False
    cache_mode: str = "none"

    # train
    batch_size: int = 12
    epochs: int = 40
    lr: float = 1e-3
    weight_decay: float = 1e-4
    workers: int = 4
    loss: str = "dice_bce"                   # dice_bce|focal|tversky|focal_tversky|dice_ce|ce
    amp: bool = False
    seed: int = 2024

    # logging/save
    logger: str = "csv+tensorboard"          # noop|csv|tensorboard|csv+tensorboard
    save_val_probs: bool = False

    # misc
    max_grad_norm: float = 5.0
    prefetch_gpu: bool = False
    auto_gpu: bool = False


def parse_args() -> UNet2DArgs:
    p = argparse.ArgumentParser("Train U-Net 2D/2.5D (class runner)")
    # data/model
    p.add_argument("--train-list", required=True)
    p.add_argument("--val-list", required=True)
    p.add_argument("--out-dir", default="runs/unet2d")
    p.add_argument("--run-tag", default="", help="Optional suffix appended to the run directory.")
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--aug", default="light", choices=["none", "light", "medium"])
    p.add_argument("--model", default="unet", choices=["unet", "unetpp"])
    p.add_argument("--encoder", default="resnet34")
    p.add_argument("--encoder-weights", default="none")
    p.add_argument("--classes", type=int, default=1)
    p.add_argument("--imagenet-norm", action="store_true")
    p.add_argument(
        "--cache-mode",
        default="none",
        choices=["none", "cpu"],
        help="Caching strategy for NPZ volumes (none: read every access, cpu: keep full volumes in RAM).",
    )
    # train
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--loss", default="dice_bce",
                   choices=["dice_bce", "focal", "tversky", "focal_tversky", "dice_ce", "ce"])
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=2024)
    # logging/save
    p.add_argument(
        "--logger",
        default="csv+tensorboard",
        help="Logger backend(s). Accepts 'noop', 'csv', 'tensorboard', or combinations like 'csv+tensorboard'.",
    )
    p.add_argument("--save-val-probs", action="store_true")
    # misc
    p.add_argument("--max-grad-norm", type=float, default=5.0)
    p.add_argument(
        "--prefetch-gpu",
        action="store_true",
        help="Stage batches on GPU memory to reduce host RAM pressure.",
    )
    p.add_argument(
        "--auto-gpu",
        action="store_true",
        help="Heuristically increase GPU utilisation (enables AMP, GPU prefetch, larger batch/workers when VRAM is available).",
    )
    a = p.parse_args()
    return UNet2DArgs(**vars(a))


class UNet2DTrainer:
    """
    Runner for U-Net 2D/2.5D training:
      - Prepare DataLoader from KneeNPZ2DSlices
      - Build model (SMP), loss, optimizer, scheduler, scaler
      - Use Engine to train/validate/save_samples
      - Persist the best checkpoint and optional val probabilities
    """

    def __init__(self, args: UNet2DArgs):
        self.args = args
        set_seed(args.seed)
        base_out_dir = Path(args.out_dir)
        base_out_dir.mkdir(parents=True, exist_ok=True)
        day_stamp = datetime.now().strftime("%Y-%m-%d")
        run_name = f"{day_stamp}_{args.model}_{args.encoder}"
        suffix_tag = (args.run_tag or "").strip()
        if suffix_tag:
            safe_tag = suffix_tag.replace(" ", "_")
            run_name = f"{run_name}_{safe_tag}"
        run_dir = base_out_dir / run_name
        suffix = 2
        while run_dir.exists():
            run_dir = base_out_dir / f"{run_name}_{suffix:02d}"
            suffix += 1
        self.out_dir = run_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.args.out_dir = str(self.out_dir)
        start_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.epoch_log_path = self.out_dir / "epoch_log.txt"
        self.aggregate_log_path = self.out_dir.parent / "training_runs.log"
        header = (
            f"# Run {self.out_dir.name} | model={args.model} | encoder={args.encoder} | "
            f"epochs={args.epochs} | started {start_stamp}\n"
        )
        with self.epoch_log_path.open("w", encoding="utf-8") as fh:
            fh.write(header)
        with self.aggregate_log_path.open("a", encoding="utf-8") as fh:
            fh.write("\n" + header)
        self._dump_config()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if getattr(self.args, "auto_gpu", False):
            self._apply_auto_gpu_tweaks()

        self._build_dataloaders()
        self._build_model()
        self._build_optimization_stack()

        self.logger = make_logger(args.logger, str(self.out_dir))
        self.cfg: Dict[str, Any] = {
            "amp": bool(args.amp),
            "classes": int(args.classes),
            "max_grad_norm": float(getattr(args, "max_grad_norm", 5.0)),
            "out_dir": str(self.out_dir),
            "epochs": args.epochs,
            "save_best": True,
        }
        self.engine = Engine(
            model=self.model,
            device=self.device,
            cfg=self.cfg,
            optimizer=self.optimizer,
            loss_obj=self.loss_obj,
            logger=self.logger,
            scaler=self.scaler,
        )

        self.best_metric = float("-inf")
        self.best_ckpt_path = self.out_dir / "best.pt"
        self.best_snapshot: Dict[str, Any] = {}

    def _dump_config(self) -> None:
        with (self.out_dir / "args.json").open("w", encoding="utf-8") as f:
            json.dump(asdict(self.args), f, indent=2)

    def _build_dataloaders(self) -> None:
        common = dict(
            k=self.args.k,
            imagenet_norm=self.args.imagenet_norm,
            encoder_name=self.args.encoder,
            cache_mode=self.args.cache_mode,
        )
        self.train_ds = KneeNPZ2DSlices(self.args.train_list, aug=self.args.aug, **common)
        self.val_ds = KneeNPZ2DSlices(self.args.val_list, aug="none", **common)
        print(f"[data] Train slices by dataset: {self.train_ds.dataset_summary}")
        print(f"[data] Val slices by dataset:   {self.val_ds.dataset_summary}")
        self.train_ld = DataLoader(
            self.train_ds,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.workers,
            pin_memory=True,
            drop_last=True,
        )
        self.val_ld = DataLoader(
            self.val_ds,
            batch_size=max(1, self.args.batch_size // 2),
            shuffle=False,
            num_workers=self.args.workers,
            pin_memory=True,
        )
        if self.args.prefetch_gpu and self.device.type == "cuda":
            self.train_ld = DevicePrefetchLoader(self.train_ld, self.device)
            self.val_ld = DevicePrefetchLoader(self.val_ld, self.device)

    def _apply_auto_gpu_tweaks(self) -> None:
        if self.device.type != "cuda":
            print("[auto-gpu] CUDA unavailable; skipping GPU tuning.")
            return
        if not hasattr(torch.cuda, "mem_get_info"):
            print("[auto-gpu] torch.cuda.mem_get_info not available; skipping GPU tuning.")
            return
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_gb = free_bytes / (1024 ** 3)
        total_gb = total_bytes / (1024 ** 3)
        adjustments: list[str] = []

        new_batch = self.args.batch_size
        for threshold, candidate in ((32, 36), (24, 28), (18, 24), (12, 16)):
            if free_gb >= threshold and candidate > new_batch:
                new_batch = candidate
                break
        if new_batch != self.args.batch_size:
            adjustments.append(f"batch_size {self.args.batch_size} -> {new_batch}")
            self.args.batch_size = new_batch

        target_workers = min(max(os.cpu_count() or 4, 4), 8)
        if free_gb >= 12 and self.args.workers < target_workers:
            adjustments.append(f"workers {self.args.workers} -> {target_workers}")
            self.args.workers = target_workers

        if not self.args.prefetch_gpu:
            self.args.prefetch_gpu = True
            adjustments.append("prefetch_gpu=ON")

        if not self.args.amp:
            self.args.amp = True
            adjustments.append("amp=ON")

        if adjustments:
            summary = ", ".join(adjustments)
            print(f"[auto-gpu] Free {free_gb:.1f} GB / Total {total_gb:.1f} GB | applied: {summary}")
        else:
            print(f"[auto-gpu] Free {free_gb:.1f} GB / Total {total_gb:.1f} GB | no changes needed.")

    def _determine_in_channels(self) -> int:
        if self.args.k == 1 and self.args.imagenet_norm:
            return 3
        return self.args.k

    def _build_model(self) -> None:
        in_ch = self._determine_in_channels()
        self.model = build_unet(
            self.args.model,
            self.args.encoder,
            self.args.encoder_weights,
            in_ch=in_ch,
            classes=self.args.classes,
        )
        self.model.to(self.device)

    def _build_optimization_stack(self) -> None:
        self.loss_obj = _build_loss(self.args.classes, self.args.loss)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=8,
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.args.amp)

    def _record_best(self, epoch: int, train_loss: float, val_loss: float, val_dice: float, val_iou: float, lr: float) -> None:
        self.best_snapshot = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_dice": float(val_dice),
            "val_iou": float(val_iou),
            "lr": float(lr),
        }

    def _metric_key(self, val_loss: float, val_dice: float) -> float:
        return val_dice if self.args.classes == 1 else -val_loss

    def _save_best(self) -> None:
        ckpt_dir = self.out_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.model.state_dict(), "args": asdict(self.args)}, self.best_ckpt_path)

    def _save_val_probs_if_needed(self) -> None:
        if not self.args.save_val_probs:
            return
        self.model.eval()
        probs_list, gt_list = [], []
        with torch.no_grad():
            for x, y in self.val_ld:
                x = x.to(self.device)
                with torch.amp.autocast("cuda", enabled=self.args.amp):
                    logits = self.model(x)
                    if self.args.classes == 1:
                        probs = torch.sigmoid(logits).cpu().numpy()
                        gts = y.cpu().numpy()
                        gts = gts if gts.ndim == 4 else gts[:, None, ...]
                    else:
                        probs = torch.softmax(logits, 1).cpu().numpy()
                        gts = y.cpu().numpy()
                probs_list.append(probs)
                gt_list.append(gts)
        np.savez_compressed(
            self.out_dir / "val_preds.npz",
            probs=np.concatenate(probs_list, 0),
            gts=np.concatenate(gt_list, 0),
        )

    def _append_epoch_log(self, line: str) -> None:
        with self.epoch_log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        with self.aggregate_log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"[{self.out_dir.name}] {line}\n")

    def run(self) -> Dict[str, Any]:
        """Ch?y full training loop v? tr? th?ng tin k?t qu?."""
        t0 = time()
        history = []

        for ep in range(1, self.args.epochs + 1):
            self.cfg["epoch"] = ep

            train_loss = self.engine.train_one_epoch(self.train_ld)
            val_loss, val_dice, val_iou = self.engine.validate(self.val_ld)

            self.scheduler.step(val_loss)
            lr = float(self.optimizer.param_groups[0]["lr"])
            elapsed = time() - t0

            log_line = (
                f"Epoch {ep:03d}/{self.args.epochs} | "
                f"train {train_loss:.4f} | val {val_loss:.4f} | "
                f"dice {val_dice:.4f} | iou {val_iou:.4f} | lr {lr:.2e} | {elapsed:.1f}s"
            )
            print(log_line)
            self._append_epoch_log(log_line)

            if hasattr(self.logger, "log_epoch"):
                self.logger.log_epoch(
                    epoch=ep,
                    time_s=elapsed,
                    train_loss=float(train_loss),
                    val_loss=float(val_loss),
                    val_dice=float(val_dice),
                    val_iou=float(val_iou),
                    lr=lr,
                )
            history.append(
                {
                    "epoch": ep,
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "val_dice": float(val_dice),
                    "val_iou": float(val_iou),
                    "lr": lr,
                }
            )

            metric_key = self._metric_key(val_loss, val_dice)
            if metric_key > self.best_metric:
                self.best_metric = metric_key
                self._record_best(ep, train_loss, val_loss, val_dice, val_iou, lr)
                self._save_best()
                self._save_val_probs_if_needed()
                if hasattr(self.logger, "log_best"):
                    self.logger.log_best(
                        epoch=ep,
                        key=float(metric_key),
                        ckpt_path=str(self.best_ckpt_path),
                    )

            if ep == 1 or ep % 5 == 0:
                self.engine.save_samples(self.val_ld, str(self.out_dir), max_samples=6)

        final_snapshot = history[-1] if history else {}
        if self.best_snapshot:
            summary = {
                "best": self.best_snapshot,
                "final": final_snapshot,
                "best_ckpt": str(self.best_ckpt_path),
                "epochs": int(self.args.epochs),
            }
        else:
            summary = {
                "best": {},
                "final": final_snapshot,
                "best_ckpt": str(self.best_ckpt_path),
                "epochs": int(self.args.epochs),
            }
        with (self.out_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        with (self.out_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        if hasattr(self.logger, "log_meta"):
            self.logger.log_meta(
                {
                    "best_ckpt": str(self.best_ckpt_path),
                    "epochs": self.args.epochs,
                    "batch_size": self.args.batch_size,
                    "lr_init": self.args.lr,
                    "weight_decay": self.args.weight_decay,
                    "scheduler": "ReduceLROnPlateau",
                    "model": self.args.model,
                    "encoder": self.args.encoder,
                    "encoder_weights": self.args.encoder_weights,
                    "classes": self.args.classes,
                    "k_2p5d": self.args.k,
                    "imagenet_norm": bool(self.args.imagenet_norm),
                    "aug": self.args.aug,
                    "seed": self.args.seed,
                }
            )
        if hasattr(self.logger, "close"):
            self.logger.close()

        print("Done. Best ckpt:", self.best_ckpt_path)
        return {"best_ckpt": str(self.best_ckpt_path), "history": history, "summary": summary}
