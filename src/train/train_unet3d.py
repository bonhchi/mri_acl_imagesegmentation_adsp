# -*- coding: utf-8 -*-
"""
3D U-Net patch-based (MONAI) cho RTX 3060 12GB.
Yêu cầu: pip install monai nibabel
Input .npz: img (S,1,H,W) float32 | msk (S,H,W) uint8/int
"""
import os, argparse, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from monai.networks.nets import UNet
from monai.losses import DiceCELoss

class KneeNPZ3D(Dataset):
    def __init__(self, list_txt, roi=(160,160,64), pos_neg_ratio=1.0, samples_per_vol=12, train=True):
        self.files = [l.strip() for l in open(list_txt, "r", encoding="utf-8") if l.strip()]
        self.roi = tuple(roi)
        self.r = pos_neg_ratio
        self.n = samples_per_vol
        self.train = train

    def _load(self, path):
        z = np.load(path)
        x = z["img"].astype(np.float32)  # (S,1,H,W)
        y = z["msk"].astype(np.int64)    # (S,H,W)
        # nếu preprocess đã z-score per-volume, có thể bỏ 2 dòng dưới:
        mu, sd = float(x.mean()), float(x.std() + 1e-6)
        x = (x - mu) / sd
        return x, y

    def _sample_patch(self, x, y, positive):
        S, _, H, W = x.shape
        rz, ry, rx = self.roi[2], self.roi[0], self.roi[1]

        if positive and y.max() > 0:
            zs, ys, xs = np.where(y > 0)
            k = np.random.randint(0, len(zs))
            cz, cy, cx = int(zs[k]), int(ys[k]), int(xs[k])
        else:
            cz, cy, cx = np.random.randint(0, S), np.random.randint(0, H), np.random.randint(0, W)

        z0 = np.clip(cz - rz // 2, 0, max(0, S - rz))
        y0 = np.clip(cy - ry // 2, 0, max(0, H - ry))
        x0 = np.clip(cx - rx // 2, 0, max(0, W - rx))
        px = x[z0:z0+rz, :, y0:y0+ry, x0:x0+rx]   # (D,1,H,W)
        py = y[z0:z0+rz,     y0:y0+ry, x0:x0+rx]  # (D,H,W)

        px = np.moveaxis(px, 0, 1)  # (1,D,H,W) cho MONAI
        return torch.from_numpy(px.copy()), torch.from_numpy(py.copy())

    def __len__(self): return len(self.files) * (self.n if self.train else 2)

    def __getitem__(self, i):
        fi = i // (self.n if self.train else 2)
        x, y = self._load(self.files[fi])
        if self.train:
            pos = np.random.rand() < (self.r / (1 + self.r))
            px, py = self._sample_patch(x, y, positive=pos)
        else:
            px, py = self._sample_patch(x, y, positive=(y.max() > 0))
        if py.max() <= 1: py = py.unsqueeze(0).float()  # (1,D,H,W)
        else: py = py.long()
        return px.float(), py

def build_unet3d(in_ch=1, classes=1, channels=(32,64,128,256,320)):
    return UNet(spatial_dims=3, in_channels=in_ch, out_channels=(1 if classes==1 else classes),
                channels=channels, strides=(2,2,2,2), num_res_units=2)

def parse_args():
    p = argparse.ArgumentParser("Train 3D U-Net (MONAI)")
    p.add_argument("--train-list", type=str, required=True)
    p.add_argument("--val-list", type=str, required=True)
    p.add_argument("--out-dir", type=str, default="runs/unet3d_balanced")

    p.add_argument("--roi", type=int, nargs=3, default=[160,160,64], help="H W D")
    p.add_argument("--channels", type=int, nargs="+", default=[32,64,128,256,320])
    p.add_argument("--classes", type=int, default=1)

    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=2024)
    return p.parse_args()

def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def main():
    args = parse_args(); set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    train_ds = KneeNPZ3D(args.train_list, roi=tuple(args.roi), samples_per_vol=12, train=True)
    val_ds   = KneeNPZ3D(args.val_list,   roi=tuple(args.roi), samples_per_vol=2,  train=False)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, pin_memory=True)
    val_ld   = DataLoader(val_ds, batch_size=1, shuffle=False,
                          num_workers=args.workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_unet3d(in_ch=1, classes=args.classes, channels=tuple(args.channels)).to(device)
    loss = DiceCELoss(sigmoid=True, to_onehot_y=False) if args.classes==1 else DiceCELoss(to_onehot_y=True, softmax=True)
    opt  = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best = 1e9
    log_path = os.path.join(args.out_dir, "train_log.csv")
    with open(log_path, "w") as f: f.write("epoch,train_loss,val_loss,lr\n")

    for ep in range(1, args.epochs + 1):
        # train
        model.train(); tr = 0.0
        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(x)   # N,1,D,H,W
                l = loss(logits, y)
            scaler.scale(l).backward(); scaler.step(opt); scaler.update()
            tr += l.item()
        tr /= max(1, len(train_ld))

        # val
        model.eval(); vl = 0.0
        with torch.no_grad():
            for x, y in val_ld:
                x, y = x.to(device), y.to(device)
                with torch.cuda.amp.autocast(enabled=args.amp):
                    logits = model(x)
                    l = loss(logits, y)
                vl += l.item()
        vl /= max(1, len(val_ld))
        lr = opt.param_groups[0]["lr"]
        print(f"Epoch {ep:03d}/{args.epochs} | train {tr:.4f} | val {vl:.4f} | lr {lr:.2e}")
        with open(log_path, "a") as f: f.write(f"{ep},{tr:.6f},{vl:.6f},{lr:.6e}\n")

        if vl < best:
            best = vl
            torch.save({"model": model.state_dict(), "args": vars(args)}, os.path.join(args.out_dir, "best3d.pt"))
            print("  >> saved best")

    print("Done.")

if __name__ == "__main__":
    main()
