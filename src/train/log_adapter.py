"""Logging adapters for training loops."""

import csv
import json
import os
from typing import Iterable, List

from .log_iface import TrainLogger

try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    SummaryWriter = None  # type: ignore


class NoOpLogger(TrainLogger):
    def log_step(self, **kw):
        pass

    def log_epoch(self, **kw):
        pass

    def log_best(self, **kw):
        pass

    def log_meta(self, meta):
        pass

    def close(self):
        pass


class CSVLoggerAdapter(TrainLogger):
    def __init__(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        self.ep = os.path.join(out_dir, "history_epoch.csv")
        self.st = os.path.join(out_dir, "history_step.csv")
        if not os.path.exists(self.ep):
            with open(self.ep, "w", newline="") as f:
                csv.writer(f).writerow(
                    [
                        "epoch",
                        "time_s",
                        "train_loss",
                        "val_loss",
                        "val_dice",
                        "val_iou",
                        "lr",
                    ]
                )
        if not os.path.exists(self.st):
            with open(self.st, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["global_step", "epoch", "lr", "train_loss_step"]
                )
        self.meta = os.path.join(out_dir, "metrics.json")

    def log_step(self, *, global_step: int, epoch: int, lr: float, loss: float) -> None:
        with open(self.st, "a", newline="") as f:
            csv.writer(f).writerow([global_step, epoch, lr, loss])

    def log_epoch(self, **row) -> None:
        with open(self.ep, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    row["epoch"],
                    round(row["time_s"], 2),
                    row["train_loss"],
                    row["val_loss"],
                    row["val_dice"],
                    row["val_iou"],
                    row["lr"],
                ]
            )

    def log_best(self, **kw):
        pass

    def log_meta(self, meta):
        with open(self.meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def close(self):
        pass


class TensorBoardLoggerAdapter(TrainLogger):
    def __init__(self, out_dir: str):
        if SummaryWriter is None:
            raise RuntimeError(
                "Requested TensorBoard logging but torch.utils.tensorboard is unavailable. "
                "Install the 'tensorboard' package to enable this logger."
            )
        os.makedirs(out_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=out_dir)

    def log_step(self, *, global_step: int, epoch: int, lr: float, loss: float) -> None:
        self.writer.add_scalar("train/loss_step", loss, global_step)
        self.writer.add_scalar("train/lr", lr, global_step)
        self.writer.add_scalar("train/epoch_index", epoch, global_step)

    def log_epoch(
        self,
        *,
        epoch: int,
        time_s: float,
        train_loss: float,
        val_loss: float,
        val_dice: float,
        val_iou: float,
        lr: float,
    ) -> None:
        self.writer.add_scalar("epoch/train_loss", train_loss, epoch)
        self.writer.add_scalar("epoch/val_loss", val_loss, epoch)
        self.writer.add_scalar("epoch/val_dice", val_dice, epoch)
        self.writer.add_scalar("epoch/val_iou", val_iou, epoch)
        self.writer.add_scalar("epoch/lr", lr, epoch)
        self.writer.add_scalar("epoch/time_s", time_s, epoch)

    def log_best(self, *, epoch: int, key: float, ckpt_path: str) -> None:
        self.writer.add_scalar("best/metric", key, epoch)
        self.writer.add_text("best/checkpoint_path", ckpt_path, epoch)

    def log_meta(self, meta):
        self.writer.add_text("meta/config", json.dumps(meta, indent=2))

    def close(self):
        self.writer.flush()
        self.writer.close()


class CompositeLogger(TrainLogger):
    def __init__(self, loggers: Iterable[TrainLogger]):
        self._loggers: List[TrainLogger] = [logger for logger in loggers]

    def log_step(self, **kw):
        for logger in self._loggers:
            logger.log_step(**kw)

    def log_epoch(self, **kw):
        for logger in self._loggers:
            logger.log_epoch(**kw)

    def log_best(self, **kw):
        for logger in self._loggers:
            logger.log_best(**kw)

    def log_meta(self, meta):
        for logger in self._loggers:
            logger.log_meta(meta)

    def close(self):
        for logger in self._loggers:
            logger.close()


def _build_logger(kind: str, out_dir: str) -> TrainLogger:
    key = kind.strip().lower()
    if key == "noop":
        return NoOpLogger()
    if key == "csv":
        return CSVLoggerAdapter(out_dir)
    if key in {"tensorboard", "tb"}:
        return TensorBoardLoggerAdapter(out_dir)
    raise ValueError(f"Unsupported logger kind: {kind}")


def make_logger(kind: str, out_dir: str):
    if not kind:
        return NoOpLogger()
    tokens = [token.strip() for token in kind.replace(",", "+").split("+") if token.strip()]
    if not tokens:
        return NoOpLogger()

    concrete = [t for t in tokens if t.lower() != "noop"]
    if not concrete:
        return NoOpLogger()
    if len(concrete) == 1:
        return _build_logger(concrete[0], out_dir)

    loggers = [_build_logger(name, out_dir) for name in concrete]
    return CompositeLogger(loggers)
