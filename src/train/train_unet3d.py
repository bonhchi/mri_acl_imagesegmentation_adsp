# src/train/train_unet3d.py
# Train 3D U-Net (patch-based) cho RTX 3060 12GB
# - ROI mặc định: 160x160x64; batch=2 (có AMP)

from __future__ import annotations
import os, argparse, numpy as np, torch
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any, Optional
from torch.utils.data import Dataset, DataLoader
from monai.networks.nets import UNet
from monai.losses import DiceCELoss


# ----------------------- Dataset -----------------------
class KneeNPZ3D(Dataset):
    """Random-crop 3D patch dataset (ưu tiên dương tính ~ pos_neg_ratio)."""
    def __init__(
        self,
        list_txt: str,
        roi: Tuple[int,int,int] = (160,160,64),
        pos_neg_ratio: float = 1.0,
        samples_per_vol: int = 12,
        train: bool = True,
    ):
        self.files = [l.strip() for l in open(list_txt, "r", encoding="utf-8") if l.strip()]
        self.roi = tuple(roi)
        self.r = float(pos_neg_ratio)
        self.n = int(samples_per_vol)
        self.train = bool(train)

    def _load(self, p: str):
        z = np.load(p)
        x = z["img"].astype(np.float32)  # (S,1,H,W)
        y = z["msk"].astype(np.int64)    # (S,H,W)
        mu, sd = float(x.mean()), float(x.std() + 1e-6)
        x = (x - mu) / sd  # z-score per volume
        return x, y

    def _sample(self, x: np.ndarray, y: np.ndarray, positive: bool):
        S, _, H, W = x.shape
        D, Y, X = self.roi[2], self.roi[0], self.roi[1]
        if positive and y.max() > 0:
            zs, ys, xs = np.where(y > 0)
            k = np.random.randint(0, len(zs))
            cz, cy, cx = int(zs[k]), int(ys[k]), int(xs[k])
        else:
            cz, cy, cx = np.random.randint(0, S), np.random.randint(0, H), np.random.randint(0, W)
        z0 = np.clip(cz - D // 2, 0, max(0, S - D))
        y0 = np.clip(cy - Y // 2, 0, max(0, H - Y))
        x0 = np.clip(cx - X // 2, 0, max(0, W - X))
        px = x[z0:z0+D, :, y0:y0+Y, x0:x0+X]  # (D,1,Y,X)
        py = y[z0:z0+D, y0:y0+Y, x0:x0+X]     # (D,Y,X)
        px = np.moveaxis(px, 0, 1)            # -> (1,D,Y,X)
        return torch.from_numpy(px.copy()).float(), torch.from_numpy(py.copy())

    def __len__(self):
        return len(self.files) * (self.n if self.train else 2)

    def __getitem__(self, i):
        fidx = i // (self.n if self.train else 2)
        x, y = self._load(self.files[fidx])
        if self.train:
            pos = np.random.rand() < (self.r / (1 + self.r))
            px, py = self._sample(x, y, pos)
        else:
            px, py = self._sample(x, y, (y.max() > 0))
        if py.max() <= 1:
            py = py.unsqueeze(0).float()  # (1,D,H,W) binary
        else:
            py = py.long()
        return px, py


# ----------------------- Model -----------------------
def build_unet3d(in_ch=1, classes=1, channels=(32, 64, 128, 256, 320)):
    return UNet(
        spatial_dims=3,
        in_channels=in_ch,
        out_channels=(1 if classes == 1 else classes),
        channels=channels,
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )


# ----------------------- Args (dataclass) -----------------------
@dataclass
class UNet3DArgs:
    train_list: str = ""
    val_list: str = ""
    out_dir: str = "runs/unet3d_balanced"
    roi: Tuple[int,int,int] = (160,160,64)
    channels: Tuple[int,...] = (32, 64, 128, 256, 320)
    classes: int = 1
    batch_size: int = 2
    epochs: int = 80
    lr: float = 1e-3
    weight_decay: float = 1e-4
    workers: int = 4
    amp: bool = True
    seed: int = 2024
    pos_neg_ratio: float = 1.0
    samples_per_vol_train: int = 12
    samples_per_vol_val: int = 2

def parse_args() -> UNet3DArgs:
    p = argparse.ArgumentParser("Train 3D U-Net (class runner)")
    p.add_argument("--train-list", required=True)
    p.add_argument("--val-list", required=True)
    p.add_argument("--out-dir", default="runs/unet3d_balanced")
    p.add_argument("--roi", type=int, nargs=3, default=[160,160,64])
    p.add_argument("--channels", type=int, nargs="+", default=[32,64,128,256,320])
    p.add_argument("--classes", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=2024)
    # extra
    p.add_argument("--pos-neg-ratio", type=float, default=1.0)
    p.add_argument("--samples-per-vol-train", type=int, default=12)
    p.add_argument("--samples-per-vol-val", type=int, default=2)
    a = p.parse_args()
    return UNet3DArgs(
        train_list=a.train_list, val_list=a.val_list, out_dir=a.out_dir,
        roi=tuple(a.roi), channels=tuple(a.channels), classes=a.classes,
        batch_size=a.batch_size, epochs=a.epochs, lr=a.lr, weight_decay=a.weight_decay,
        workers=a.workers, amp=a.amp, seed=a.seed,
        pos_neg_ratio=a.pos_neg_ratio,
        samples_per_vol_train=a.samples_per_vol_train,
        samples_per_vol_val=a.samples_per_vol_val,
    )


# ----------------------- Trainer Class -----------------------
class UNet3DTrainer:
    """Runner tổng cho 3D U-Net patch-based (không có hàm main)."""

    def __init__(self, args: UNet3DArgs):
        self.args = args
        self._set_seed(args.seed)
        os.makedirs(args.out_dir, exist_ok=True)

        # datasets / loaders
        self.train_ds = KneeNPZ3D(
            args.train_list, roi=args.roi,
            pos_neg_ratio=args.pos_neg_ratio,
            samples_per_vol=args.samples_per_vol_train,
            train=True,
        )
        self.val_ds = KneeNPZ3D(
            args.val_list, roi=args.roi,
            pos_neg_ratio=args.pos_neg_ratio,
            samples_per_vol=args.samples_per_vol_val,
            train=False,
        )
        self.train_ld = DataLoader(
            self.train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True
        )
        self.val_ld = DataLoader(
            self.val_ds, batch_size=1, shuffle=False,
            num_workers=args.workers, pin_memory=True
        )

        # model / loss / opt
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_unet3d(in_ch=1, classes=args.classes, channels=args.channels).to(self.device)
        self.loss = (
            DiceCELoss(sigmoid=True, to_onehot_y=False)
            if args.classes == 1
            else DiceCELoss(to_onehot_y=True, softmax=True)
        )
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

        # logging / ckpt
        self.best_val = float("inf")
        self.best_path = os.path.join(args.out_dir, "best3d.pt")
        self.log_csv = os.path.join(args.out_dir, "train_log.csv")
        if not os.path.exists(self.log_csv):
            with open(self.log_csv, "w") as f:
                f.write("epoch,train_loss,val_loss,lr\n")

    # --------------- utils ---------------
    @staticmethod
    def _set_seed(s: int):
        import random
        random.seed(s); np.random.seed(s)
        torch.manual_seed(s); torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # --------------- core ---------------
    def train_one_epoch(self) -> float:
        self.model.train()
        run = 0.0
        for x, y in self.train_ld:
            x, y = x.to(self.device), y.to(self.device)
            self.opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self.args.amp):
                loss = self.loss(self.model(x), y)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.opt)
            self.scaler.update()
            run += float(loss.item())
        return run / max(1, len(self.train_ld))

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        run = 0.0
        for x, y in self.val_ld:
            x, y = x.to(self.device), y.to(self.device)
            with torch.amp.autocast("cuda", enabled=self.args.amp):
                run += float(self.loss(self.model(x), y).item())
        return run / max(1, len(self.val_ld))

    def fit(self) -> Dict[str, Any]:
        for ep in range(1, self.args.epochs + 1):
            tr = self.train_one_epoch()
            vl = self.validate()
            lr = self.opt.param_groups[0]["lr"]
            print(f"Epoch {ep:03d}/{self.args.epochs} | train {tr:.4f} | val {vl:.4f} | lr {lr:.2e}")
            with open(self.log_csv, "a") as f:
                f.write(f"{ep},{tr:.6f},{vl:.6f},{lr:.6e}\n")

            if vl < self.best_val:
                self.best_val = vl
                torch.save({"model": self.model.state_dict(), "args": asdict(self.args)}, self.best_path)
                print("  >> saved best")

        print("Done. Best:", self.best_path)
        return {"best_ckpt": self.best_path, "best_val_loss": float(self.best_val)}

