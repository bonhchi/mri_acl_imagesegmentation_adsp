"""Resume a previously started 2D U-Net training run.

This helper continues training an existing run directory produced by
``train_unet_launcher.py`` / ``UNet2DTrainer``. It restores the model,
optimizer, and logger state as best as possible from the artifacts already
stored in the run folder, then continues training from the requested epoch.

Example usage:

    python src/train/resume_unet.py \\
        --run runs/kneemri_unet/2025-11-02_unet_resnet34 \\
        --start-epoch 67
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from time import time
from typing import Any, Dict, Iterable, List, Optional

import torch
from torch.utils.data import DataLoader

from src.dataio.datasets import KneeNPZ2DSlices
from src.models.unet_factory import build_unet
from src.train.engine import Engine
from src.train.log_adapter import make_logger
from src.train.train_unet import (
    DevicePrefetchLoader,
    UNet2DArgs,
    _build_loss,
    set_seed,
)


def _windows_to_posix(path_str: str) -> Path:
    """Convert a Windows drive path (``E:\\foo``) to WSL style if needed."""
    candidate = Path(path_str)
    if candidate.exists():
        return candidate
    if ":" in path_str:
        drive, rest = path_str.split(":", 1)
        rest = rest.lstrip("\\/").replace("\\", "/")
        translated = Path(f"/mnt/{drive.lower()}/{rest}")
        if translated.exists():
            return translated
    return candidate


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_history_epoch(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                rows.append(
                    {
                        "epoch": int(row["epoch"]),
                        "time_s": float(row["time_s"]),
                        "train_loss": float(row["train_loss"]),
                        "val_loss": float(row["val_loss"]),
                        "val_dice": float(row["val_dice"]),
                        "val_iou": float(row["val_iou"]),
                        "lr": float(row["lr"]),
                    }
                )
            except (KeyError, ValueError):
                continue
    return rows


def _read_last_global_step(path: Path) -> int:
    if not path.exists():
        return -1
    latest: Optional[int] = None
    with path.open("r", newline="") as fh:
        reader = csv.reader(fh)
        next(reader, None)  # skip header
        for row in reader:
            if not row:
                continue
            try:
                latest = int(float(row[0]))
            except (ValueError, IndexError):
                continue
    return latest if latest is not None else -1


def _append_epoch_logs(run_dir: Path, line: str) -> None:
    epoch_log = run_dir / "epoch_log.txt"
    aggregate = run_dir.parent / "training_runs.log"
    with epoch_log.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    with aggregate.open("a", encoding="utf-8") as fh:
        fh.write(f"[{run_dir.name}] {line}\n")


def _determine_in_channels(args: UNet2DArgs) -> int:
    return 3 if args.k == 1 and args.imagenet_norm else args.k


def _replay_scheduler(
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    history: Iterable[Dict[str, Any]],
    until_epoch: int,
) -> None:
    for row in history:
        if row["epoch"] >= until_epoch:
            break
        scheduler.step(row["val_loss"])


def _metric_key(classes: int, val_loss: float, val_dice: float) -> float:
    return val_dice if classes == 1 else -val_loss


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Resume a UNet 2D training run")
    parser.add_argument("--run", required=True, help="Path to the existing run directory.")
    parser.add_argument(
        "--start-epoch",
        type=int,
        required=True,
        help="Epoch index (1-based) to resume from. Must be greater than the last completed epoch.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override the total number of epochs for this run (default: reuse saved config).",
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Checkpoint to load. Defaults to <run>/checkpoints/last_full.pt if present, otherwise best.pt.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device string override (e.g. 'cuda:0' or 'cpu'). Default: auto-detect CUDA.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Optional checkpoint frequency (epochs). Saves full state under checkpoints/epoch_XXX_full.pt.",
    )
    parser.add_argument(
        "--no-save-last",
        action="store_true",
        help="Disable writing checkpoints/last_full.pt after each epoch.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_argument_parser()
    cli = parser.parse_args(argv)

    run_dir = Path(cli.run).expanduser().resolve()
    if not run_dir.exists():
        parser.error(f"Run directory not found: {run_dir}")

    args_path = run_dir / "args.json"
    if not args_path.exists():
        parser.error(f"args.json not found in run directory: {args_path}")

    saved_args = _load_json(args_path)
    run_args = UNet2DArgs(**saved_args)

    # Normalise important paths so they resolve correctly under WSL/Linux.
    run_args.out_dir = str(run_dir)
    run_args.train_list = str(_windows_to_posix(run_args.train_list))
    run_args.val_list = str(_windows_to_posix(run_args.val_list))

    if cli.max_epochs is not None:
        if cli.max_epochs < cli.start_epoch:
            parser.error("--max-epochs must be >= --start-epoch.")
        run_args.epochs = cli.max_epochs

    total_epochs = int(run_args.epochs)
    if cli.start_epoch > total_epochs:
        parser.error(f"--start-epoch ({cli.start_epoch}) exceeds configured max epochs ({total_epochs}).")

    history_epoch = _load_history_epoch(run_dir / "history_epoch.csv")
    last_completed_epoch = history_epoch[-1]["epoch"] if history_epoch else 0
    if cli.start_epoch <= last_completed_epoch:
        parser.error(
            f"--start-epoch must be greater than the last completed epoch ({last_completed_epoch})."
        )

    base_elapsed = history_epoch[-1]["time_s"] if history_epoch else 0.0

    summary_path = run_dir / "summary.json"
    existing_summary = _load_json(summary_path) if summary_path.exists() else {}
    existing_history_path = run_dir / "history.json"
    existing_history: List[Dict[str, Any]] = []
    if existing_history_path.exists():
        try:
            existing_history = json.loads(existing_history_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing_history = []

    set_seed(run_args.seed)

    if cli.device:
        device = torch.device(cli.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but not available.")

    train_ds = KneeNPZ2DSlices(
        run_args.train_list,
        k=run_args.k,
        aug=run_args.aug,
        imagenet_norm=run_args.imagenet_norm,
        encoder_name=run_args.encoder,
        cache_mode=run_args.cache_mode,
    )
    val_ds = KneeNPZ2DSlices(
        run_args.val_list,
        k=run_args.k,
        aug="none",
        imagenet_norm=run_args.imagenet_norm,
        encoder_name=run_args.encoder,
        cache_mode=run_args.cache_mode,
    )

    train_ld = DataLoader(
        train_ds,
        batch_size=run_args.batch_size,
        shuffle=True,
        num_workers=run_args.workers,
        pin_memory=True,
        drop_last=True,
    )
    val_ld = DataLoader(
        val_ds,
        batch_size=max(1, run_args.batch_size // 2),
        shuffle=False,
        num_workers=run_args.workers,
        pin_memory=True,
    )

    if run_args.prefetch_gpu and device.type == "cuda":
        train_ld = DevicePrefetchLoader(train_ld, device)
        val_ld = DevicePrefetchLoader(val_ld, device)

    in_channels = _determine_in_channels(run_args)
    model = build_unet(
        run_args.model,
        run_args.encoder,
        run_args.encoder_weights,
        in_ch=in_channels,
        classes=run_args.classes,
    ).to(device)

    loss_obj = _build_loss(run_args.classes, run_args.loss)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=run_args.lr,
        weight_decay=run_args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=8,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(run_args.amp) and device.type == "cuda")

    logger = make_logger(run_args.logger, str(run_dir))
    cfg = {
        "amp": bool(run_args.amp) and device.type == "cuda",
        "classes": int(run_args.classes),
        "max_grad_norm": float(getattr(run_args, "max_grad_norm", 5.0)),
        "out_dir": str(run_dir),
        "epochs": total_epochs,
        "save_best": True,
    }
    engine = Engine(
        model=model,
        device=device,
        cfg=cfg,
        optimizer=optimizer,
        loss_obj=loss_obj,
        logger=logger,
        scaler=scaler,
    )

    last_global_step = _read_last_global_step(run_dir / "history_step.csv")
    if last_global_step >= 0:
        engine.global_step = last_global_step + 1

    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(cli.ckpt).expanduser() if cli.ckpt else None
    default_ckpts = [
        checkpoints_dir / "last_full.pt",
        run_dir / "best.pt",
    ]
    load_candidates = [ckpt_path] if ckpt_path else default_ckpts

    loaded_state: Optional[Dict[str, Any]] = None
    for candidate in load_candidates:
        if candidate and candidate.exists():
            state = torch.load(candidate, map_location=device)
            if isinstance(state, dict) and "model" in state:
                model.load_state_dict(state["model"])
            else:
                model.load_state_dict(state)
            if isinstance(state, dict):
                if "optimizer" in state:
                    try:
                        optimizer.load_state_dict(state["optimizer"])
                    except Exception:
                        pass
                if "scheduler" in state:
                    try:
                        scheduler.load_state_dict(state["scheduler"])
                    except Exception:
                        pass
                if "scaler" in state and scaler is not None and state["scaler"] is not None:
                    try:
                        scaler.load_state_dict(state["scaler"])
                    except Exception:
                        pass
                loaded_state = state
            print(f"[resume] Loaded checkpoint from {candidate}")
            break
    if loaded_state is None:
        print("[resume] No optimizer/scheduler state available; continuing with freshly initialised optimizers.")

    _replay_scheduler(scheduler, history_epoch, cli.start_epoch)

    if history_epoch:
        recorded_lr = history_epoch[-1]["lr"]
        for group in optimizer.param_groups:
            group["lr"] = recorded_lr

    best_snapshot = existing_summary.get("best", {}) if isinstance(existing_summary, dict) else {}
    if run_args.classes == 1:
        best_metric = float(best_snapshot.get("val_dice", float("-inf")))
    else:
        best_metric = -float(best_snapshot.get("val_loss", float("inf")))
    if best_metric in {float("-inf"), float("inf")} or not history_epoch:
        if run_args.classes == 1 and history_epoch:
            best_metric = max(row["val_dice"] for row in history_epoch)
            best_epoch_row = max(history_epoch, key=lambda r: r["val_dice"])
            best_snapshot = {
                "epoch": best_epoch_row["epoch"],
                "train_loss": best_epoch_row["train_loss"],
                "val_loss": best_epoch_row["val_loss"],
                "val_dice": best_epoch_row["val_dice"],
                "val_iou": best_epoch_row["val_iou"],
                "lr": best_epoch_row["lr"],
            }
        else:
            best_metric = float("-inf")
            best_snapshot = {}

    history_records = list(existing_history)
    samples_every = 5
    start_time = time()
    best_ckpt_path = run_dir / "best.pt"
    print(
        f"[resume] Continuing {run_dir.name} from epoch {cli.start_epoch} "
        f"to {total_epochs} (device={device})"
    )

    try:
        for epoch in range(cli.start_epoch, total_epochs + 1):
            cfg["epoch"] = epoch

            train_loss = engine.train_one_epoch(train_ld)
            val_loss, val_dice, val_iou = engine.validate(val_ld)

            scheduler.step(val_loss)
            current_lr = float(optimizer.param_groups[0]["lr"])
            elapsed = base_elapsed + (time() - start_time)

            log_line = (
                f"Epoch {epoch:03d}/{total_epochs} | "
                f"train {train_loss:.4f} | val {val_loss:.4f} | "
                f"dice {val_dice:.4f} | iou {val_iou:.4f} | "
                f"lr {current_lr:.2e} | {elapsed:.1f}s"
            )
            print(log_line)
            _append_epoch_logs(run_dir, log_line)

            if hasattr(logger, "log_epoch"):
                logger.log_epoch(
                    epoch=epoch,
                    time_s=elapsed,
                    train_loss=float(train_loss),
                    val_loss=float(val_loss),
                    val_dice=float(val_dice),
                    val_iou=float(val_iou),
                    lr=current_lr,
                )

            history_entry = {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_dice": float(val_dice),
                "val_iou": float(val_iou),
                "lr": current_lr,
            }
            history_records.append(history_entry)

            metric_value = _metric_key(run_args.classes, val_loss, val_dice)
            if metric_value > best_metric:
                best_metric = metric_value
                best_snapshot = {
                    "epoch": int(epoch),
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "val_dice": float(val_dice),
                    "val_iou": float(val_iou),
                    "lr": current_lr,
                }
                torch.save({"model": model.state_dict(), "args": asdict(run_args)}, best_ckpt_path)
                if hasattr(logger, "log_best"):
                    logger.log_best(epoch=epoch, key=float(metric_value), ckpt_path=str(best_ckpt_path))

            if epoch == cli.start_epoch or (epoch % samples_every == 0):
                engine.save_samples(val_ld, str(run_dir), max_samples=6)

            if not cli.no_save_last:
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict() if scaler is not None else None,
                        "best_snapshot": best_snapshot,
                    },
                    checkpoints_dir / "last_full.pt",
                )
            if cli.save_every and epoch % cli.save_every == 0:
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict() if scaler is not None else None,
                        "best_snapshot": best_snapshot,
                    },
                    checkpoints_dir / f"epoch_{epoch:03d}_full.pt",
                )
    finally:
        if hasattr(logger, "close"):
            logger.close()

    history_json_path = run_dir / "history.json"
    history_json_path.write_text(json.dumps(history_records, indent=2), encoding="utf-8")

    summary = {
        "best": best_snapshot,
        "final": history_records[-1] if history_records else {},
        "best_ckpt": str(best_ckpt_path),
        "epochs": total_epochs,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if hasattr(logger, "log_meta"):
        logger.log_meta(
            {
                "best_ckpt": str(best_ckpt_path),
                "epochs": total_epochs,
                "batch_size": run_args.batch_size,
                "lr_init": run_args.lr,
                "weight_decay": run_args.weight_decay,
                "scheduler": "ReduceLROnPlateau",
                "model": run_args.model,
                "encoder": run_args.encoder,
                "encoder_weights": run_args.encoder_weights,
                "classes": run_args.classes,
                "k_2p5d": run_args.k,
                "imagenet_norm": bool(run_args.imagenet_norm),
                "aug": run_args.aug,
                "seed": run_args.seed,
            }
        )

    print("[resume] Training complete. Latest checkpoint:", checkpoints_dir / "last_full.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
