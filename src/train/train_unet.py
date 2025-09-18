# -*- coding: utf-8 -*-
import os, json, argparse, numpy as np, torch
from torch.utils.data import DataLoader
from src.dataio.datasets import KneeNPZ2DSlices
from src.models.unet_factory import build_unet
from src.train.losses import build_loss
from src.train.engine import train_one_epoch, validate, save_samples

def parse_args():
    p = argparse.ArgumentParser("Train U-Net 2D/2.5D (SMP)")
    p.add_argument("--train-list", type=str, required=True)
    p.add_argument("--val-list", type=str, required=True)
    p.add_argument("--out-dir", type=str, default="runs/unet2d")

    # data/model
    p.add_argument("--k", type=int, default=1, help="2.5D stack size (odd). 1=2D")
    p.add_argument("--aug", type=str, default="light", choices=["none","light","medium"])
    p.add_argument("--model", type=str, default="unet", choices=["unet","unetpp"])
    p.add_argument("--encoder", type=str, default="resnet34")
    p.add_argument("--encoder-weights", type=str, default="none", help="imagenet|none")
    p.add_argument("--classes", type=int, default=1)
    p.add_argument("--imagenet-norm", action="store_true")

    # train
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--loss", type=str, default="dice_bce",
                   choices=["dice_bce","focal","tversky","focal_tversky","dice_ce","ce"])
    p.add_argument("--patience", type=int, default=10)
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
    with open(os.path.join(args.out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Data
    train_ds = KneeNPZ2DSlices(args.train_list, k=args.k, aug=args.aug,
                               imagenet_norm=args.imagenet_norm, encoder_name=args.encoder)
    val_ds   = KneeNPZ2DSlices(args.val_list,   k=args.k, aug="none",
                               imagenet_norm=args.imagenet_norm, encoder_name=args.encoder)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, pin_memory=True, drop_last=True)
    val_ld   = DataLoader(val_ds, batch_size=max(1, args.batch_size//2), shuffle=False,
                          num_workers=args.workers, pin_memory=True)

    # Model/Loss/Opt
    in_ch = (3 if (args.imagenet_norm and args.k == 1) else args.k)
    model = build_unet(args.model, args.encoder, args.encoder_weights, in_ch=in_ch, classes=args.classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model = model.to(device)
    loss_obj = build_loss(args.classes, args.loss)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_key = -1.0
    best_path = os.path.join(args.out_dir, "best.pt")
    with open(os.path.join(args.out_dir, "train_log.csv"), "w") as f:
        f.write("epoch,train_loss,val_loss,val_dice,val_iou,lr\n")

    for ep in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_ld, opt, scaler, device,
                             {"amp": args.amp, "classes": args.classes}, loss_obj)
        vl, vd, vi = validate(model, val_ld, device,
                              {"amp": args.amp, "classes": args.classes}, loss_obj)
        sch.step(vl)
        lr = opt.param_groups[0]["lr"]
        print(f"Epoch {ep:03d}/{args.epochs} | train {tr:.4f} | val {vl:.4f} | dice {vd:.4f} | iou {vi:.4f} | lr {lr:.2e}")
        with open(os.path.join(args.out_dir, "train_log.csv"), "a") as f:
            f.write(f"{ep},{tr:.6f},{vl:.6f},{vd:.6f},{vi:.6f},{lr:.6e}\n")

        key = vd if args.classes == 1 else -vl
        if key > best_key:
            best_key = key
            torch.save({"model": model.state_dict(), "args": vars(args)}, best_path)
            print(f"  >> Saved best to {best_path}")

        if ep == 1 or ep % 5 == 0:
            save_samples(model, val_ld, device,
                         {"amp": args.amp, "classes": args.classes},
                         args.out_dir, max_samples=6)

    save_samples(model, val_ld, device, {"amp": args.amp, "classes": args.classes},
                 args.out_dir, max_samples=12)
    print("Done. Best:", best_path)

if __name__ == "__main__":
    main()
