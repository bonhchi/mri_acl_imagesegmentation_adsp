"""Backfill TensorBoard event files from legacy CSV logs."""

import argparse
import csv
from pathlib import Path
from typing import Dict


def _as_float(row: Dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except Exception:
        return float("nan")


def export(run_dir: Path, *, overwrite: bool = False) -> Path:
    try:
        from torch.utils.tensorboard import SummaryWriter  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("tensorboard package is required to export CSV logs.") from exc

    run_dir = run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    if overwrite:
        for event_file in run_dir.glob("events.out.tfevents.*"):
            event_file.unlink()

    writer = SummaryWriter(log_dir=str(run_dir))

    step_csv = run_dir / "history_step.csv"
    if step_csv.exists():
        with step_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                step = int(row.get("global_step", 0))
                writer.add_scalar("train/loss_step", _as_float(row, "train_loss_step"), step)
                writer.add_scalar("train/lr", _as_float(row, "lr"), step)
                writer.add_scalar("train/epoch_progress", _as_float(row, "epoch"), step)

    epoch_csv = run_dir / "history_epoch.csv"
    if epoch_csv.exists():
        with epoch_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                epoch = int(row.get("epoch", 0))
                writer.add_scalar("epoch/train_loss", _as_float(row, "train_loss"), epoch)
                writer.add_scalar("epoch/val_loss", _as_float(row, "val_loss"), epoch)
                writer.add_scalar("epoch/val_dice", _as_float(row, "val_dice"), epoch)
                writer.add_scalar("epoch/val_iou", _as_float(row, "val_iou"), epoch)
                writer.add_scalar("epoch/lr", _as_float(row, "lr"), epoch)
                writer.add_scalar("epoch/time_s", _as_float(row, "time_s"), epoch)

    meta_path = run_dir / "summary.json"
    if meta_path.exists():
        try:
            writer.add_text("meta/source", meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    writer.flush()
    writer.close()
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert existing CSV logs in a run directory into TensorBoard event files."
    )
    parser.add_argument("run_dir", type=Path, help="Path to the run directory (contains history_*.csv).")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing TensorBoard event files in the run directory before exporting.",
    )
    args = parser.parse_args()
    out_dir = export(args.run_dir, overwrite=args.overwrite)
    print(f"TensorBoard events written to {out_dir}")


if __name__ == "__main__":
    main()
