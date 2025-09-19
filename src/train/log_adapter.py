# log_adapters.py — CSV/No-op mẫu; bạn có thể viết adapter util riêng theo giao diện trên
import os, csv, json
from .log_iface import TrainLogger


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
