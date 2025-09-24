# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, argparse, numpy as np, torch
from time import time
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple
from torch.utils.data import DataLoader

from src.dataio.datasets import KneeNPZ2DSlices
from src.models.unet_factory import build_unet

# Loss: ưu tiên LossManager mới; fallback build_loss cũ nếu bạn chưa đổi
try:
    from src.train.losses import LossManager as _LossManager
    def _build_loss(classes: int, name: str):
        return _LossManager(classes=classes, name=name)
    def _compute_loss(loss_obj, logits, targets):
        return loss_obj(logits, targets)
except Exception:
    from src.train.losses import build_loss as _build_loss
    def _compute_loss(loss_obj, logits, targets):
        # nn.ModuleList hay single module đều ok
        import torch.nn as nn
        if isinstance(loss_obj, nn.ModuleList):
            return 0.5 * loss_obj[0](logits, targets) + 0.5 * loss_obj[1](logits, targets)
        return loss_obj(logits, targets)

# Logger adapters bạn đã có
from src.train.log_adapters import CSVLoggerAdapter, NoOpLogger

# Engine (class) đã refactor
from src.train.engine import Engine


def make_logger(kind: str, out_dir: str):
    return CSVLoggerAdapter(out_dir) if kind == "csv" else NoOpLogger()


def set_seed(s: int):
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class UNet2DArgs:
    # data/model
    train_list: str = ""
    val_list: str = ""
    out_dir: str = "runs/unet2d"
    k: int = 1
    aug: str = "light"                       # none|light|medium
    model: str = "unet"                      # unet|unetpp
    encoder: str = "resnet34"
    encoder_weights: str = "none"
    classes: int = 1
    imagenet_norm: bool = False

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
    logger: str = "csv"                      # noop|csv
    save_val_probs: bool = False

    # misc
    max_grad_norm: float = 5.0


def parse_args() -> UNet2DArgs:
    p = argparse.ArgumentParser("Train U-Net 2D/2.5D (class runner)")
    # data/model
    p.add_argument("--train-list", required=True)
    p.add_argument("--val-list", required=True)
    p.add_argument("--out-dir", default="runs/unet2d")
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--aug", default="light", choices=["none", "light", "medium"])
    p.add_argument("--model", default="unet", choices=["unet", "unetpp"])
    p.add_argument("--encoder", default="resnet34")
    p.add_argument("--encoder-weights", default="none")
    p.add_argument("--classes", type=int, default=1)
    p.add_argument("--imagenet-norm", action="store_true")
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
    p.add_argument("--logger", default="csv", choices=["noop", "csv"])
    p.add_argument("--save-val-probs", action="store_true")
    # misc
    p.add_argument("--max-grad-norm", type=float, default=5.0)
    a = p.parse_args()
    return UNet2DArgs(**vars(a))


class UNet2DTrainer:
    """
    Runner tổng cho training U-Net 2D/2.5D:
      - chuẩn bị DataLoader từ KneeNPZ2DSlices
      - build model (SMP), loss, optimizer, scheduler, scaler
      - sử dụng Engine (class) để train/validate/save_samples
      - lưu best checkpoint và (tuỳ chọn) probs của val
    """

    def __init__(self, args: UNet2DArgs):
        self.args = args
        set_seed(args.seed)
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "args.json"), "w") as f:
            json.dump(asdict(args), f, indent=2)

        # --- datasets / loaders ---
        self.train_ds = KneeNPZ2DSlices(
            args.train_list,
            k=args.k, aug=args.aug,
            imagenet_norm=args.imagenet_norm,
            encoder_name=args.encoder,
        )
        self.val_ds = KneeNPZ2DSlices(
            args.val_list,
            k=args.k, aug="none",
            imagenet_norm=args.imagenet_norm,
            encoder_name=args.encoder,
        )
        self.train_ld = DataLoader(
            self.train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, drop_last=True,
        )
        self.val_ld = DataLoader(
            self.val_ds, batch_size=max(1, args.batch_size // 2), shuffle=False,
            num_workers=args.workers, pin_memory=True,
        )

        # --- model ---
        in_ch = 3 if (args.imagenet_norm and args.k == 1) else args.k
        self.model = build_unet(
            args.model, args.encoder, args.encoder_weights,
            in_ch=in_ch, classes=args.classes,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # --- loss / optimizer / scheduler / scaler ---
        self.loss_obj = _build_loss(args.classes, args.loss)
        # LossManager trả nn.Module callable; build_loss cũ trả nn.ModuleList/nn.Module — đều tương thích với Engine qua compute_loss nội bộ
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=3)
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

        # --- logger ---
        self.logger = make_logger(args.logger, args.out_dir)

        # --- engine ---
        # cấu hình tối thiểu cho Engine
        self.cfg: Dict[str, Any] = {
            "amp": bool(args.amp),
            "classes": int(args.classes),
            "max_grad_norm": float(getattr(args, "max_grad_norm", 5.0)),
            "out_dir": args.out_dir,
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

        # best tracking
        self.best_key: float = -1.0
        self.best_ckpt_path: str = os.path.join(args.out_dir, "best.pt")

    def _save_best(self):
        os.makedirs(os.path.join(self.args.out_dir, "checkpoints"), exist_ok=True)
        # chấp nhận cả state_dict tối giản (như Engine.fit) hoặc gói thông tin
        torch.save({"model": self.model.state_dict(), "args": asdict(self.args)}, self.best_ckpt_path)

    def _save_val_probs_if_needed(self):
        if not self.args.save_val_probs:
            return
        self.model.eval()
        Ps, Gs = [], []
        with torch.no_grad():
            for x, y in self.val_ld:
                x = x.to(self.device)
                with torch.amp.autocast("cuda", enabled=self.args.amp):
                    lo = self.model(x)
                    if self.args.classes == 1:
                        p = torch.sigmoid(lo).cpu().numpy()
                        g = y.cpu().numpy()
                        g = g if g.ndim == 4 else g[:, None, ...]
                    else:
                        p = torch.softmax(lo, 1).cpu().numpy()
                        g = y.cpu().numpy()
                Ps.append(p); Gs.append(g)
        np.savez_compressed(
            os.path.join(self.args.out_dir, "val_preds.npz"),
            probs=np.concatenate(Ps, 0),
            gts=np.concatenate(Gs, 0),
        )

    def run(self) -> Dict[str, Any]:
        """
        Chạy full training loop (giống main cũ) và trả thông tin kết quả.
        """
        t0 = time()
        best_metric = -1.0  # vd Dice nếu binary, còn multiclass dùng -val_loss
        history = []

        for ep in range(1, self.args.epochs + 1):
            self.cfg["epoch"] = ep

            # --- train / val ---
            tr_loss = self.engine.train_one_epoch(self.train_ld)
            val_loss, val_dice, val_iou = self.engine.validate(self.val_ld)

            # --- sched theo val_loss (như code gốc) ---
            self.scheduler.step(val_loss)
            lr = float(self.optimizer.param_groups[0]["lr"])
            el = time() - t0

            # --- console log ---
            print(f"Epoch {ep:03d}/{self.args.epochs} | "
                  f"train {tr_loss:.4f} | val {val_loss:.4f} | "
                  f"dice {val_dice:.4f} | iou {val_iou:.4f} | lr {lr:.2e} | {el:.1f}s")

            # --- logger epoch ---
            if hasattr(self.logger, "log_epoch"):
                self.logger.log_epoch(
                    epoch=ep, time_s=el,
                    train_loss=float(tr_loss), val_loss=float(val_loss),
                    val_dice=float(val_dice), val_iou=float(val_iou), lr=lr,
                )
            history.append({"epoch": ep, "train_loss": float(tr_loss),
                            "val_loss": float(val_loss), "val_dice": float(val_dice),
                            "val_iou": float(val_iou), "lr": lr})

            # --- chọn best ---
            key = val_dice if self.args.classes == 1 else -val_loss
            if key > best_metric:
                best_metric = key
                self._save_best()
                self._save_val_probs_if_needed()

            # --- lưu samples định kỳ ---
            if ep == 1 or ep % 5 == 0:
                self.engine.save_samples(self.val_ld, self.args.out_dir, max_samples=6)

        # save thêm samples cuối
        self.engine.save_samples(self.val_ld, self.args.out_dir, max_samples=12)

        # meta log & close
        if hasattr(self.logger, "log_meta"):
            self.logger.log_meta({
                "best_ckpt": self.best_ckpt_path,
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
            })
        if hasattr(self.logger, "close"):
            self.logger.close()

        print("Done. Best ckpt:", self.best_ckpt_path)
        return {
            "best_ckpt": self.best_ckpt_path,
            "history": history,
        }
