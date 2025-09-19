# -*- coding: utf-8 -*-
import os, json, argparse, numpy as np, torch
from time import time
from torch.utils.data import DataLoader
from src.dataio.datasets import KneeNPZ2DSlices
from src.models.unet_factory import build_unet
from src.train.losses import build_loss
from src.train.engine import train_one_epoch, validate, save_samples

# chọn logger backend (noop/csv hoặc adapter riêng của bạn)
from src.train.log_adapters import CSVLoggerAdapter, NoOpLogger
def make_logger(kind, out_dir):
    return CSVLoggerAdapter(out_dir) if kind=="csv" else NoOpLogger()


def parse_args():
    p=argparse.ArgumentParser("Train U-Net 2D/2.5D")
    # data/model
    p.add_argument("--train-list",required=True); p.add_argument("--val-list",required=True)
    p.add_argument("--out-dir",default="runs/unet2d")
    p.add_argument("--k",type=int,default=1); p.add_argument("--aug",default="light",choices=["none","light","medium"])
    p.add_argument("--model",default="unet",choices=["unet","unetpp"])
    p.add_argument("--encoder",default="resnet34"); p.add_argument("--encoder-weights",default="none")
    p.add_argument("--classes",type=int,default=1); p.add_argument("--imagenet-norm",action="store_true")
    # train
    p.add_argument("--batch-size",type=int,default=12); p.add_argument("--epochs",type=int,default=40)
    p.add_argument("--lr",type=float,default=1e-3); p.add_argument("--weight-decay",type=float,default=1e-4)
    p.add_argument("--workers",type=int,default=4); p.add_argument("--loss",default="dice_bce",
              choices=["dice_bce","focal","tversky","focal_tversky","dice_ce","ce"])
    p.add_argument("--amp",action="store_true"); p.add_argument("--seed",type=int,default=2024)
    # logging/save
    p.add_argument("--logger",default="csv",choices=["noop","csv"])
    p.add_argument("--save-val-probs",action="store_true")
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

       # 3) Train loop (log step/epoch, save best, sample grid)
    best_key=-1.0; best_path=os.path.join(args.out_dir,"best.pt")
    global_step=0; t0=time()

    for ep in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_ld, opt, scaler, device,
                             {"amp": args.amp, "classes": args.classes}, loss_obj)
        vl, vd, vi = validate(model, val_ld, device,
                              {"amp": args.amp, "classes": args.classes}, loss_obj)
        sch.step(vl)
        lr = float(opt.param_groups[0]["lr"])
        el = time() - t0
        print(f"Epoch {ep:03d}/{args.epochs} | train {tr:.4f} | val {vl:.4f} | dice {vd:.4f} | iou {vi:.4f} | lr {lr:.2e} | {el:.1f}s")
        logger.log_epoch(epoch=ep,time_s=el,train_loss=float(tr),val_loss=float(vl),
                         val_dice=float(vd),val_iou=float(vi),lr=lr)

        key = vd if args.classes == 1 else -vl
        if key > best_key:
            best_key = key
            torch.save({"model": model.state_dict(), "args": vars(args)}, best_path)
            if args.save_val_probs:  # lưu probs của val để vẽ PR/ROC
                model.eval(); Ps, Gs = [], []
                with torch.no_grad():
                    for x,y in val_ld:
                        x=x.to(device)
                        with torch.amp.autocast("cuda",enabled=args.amp):
                            lo=model(x)
                            if args.classes==1:
                                p=torch.sigmoid(lo).cpu().numpy(); g=y.cpu().numpy(); g=g if g.ndim==4 else g[:,None,...]
                            else:
                                p=torch.softmax(lo,1).cpu().numpy(); g=y.cpu().numpy()
                        Ps.append(p); Gs.append(g)
                np.savez_compressed(os.path.join(args.out_dir,"val_preds.npz"),
                                    probs=np.concatenate(Ps,0), gts=np.concatenate(Gs,0))

        if ep == 1 or ep % 5 == 0:
            save_samples(model, val_ld, device,
                         {"amp": args.amp, "classes": args.classes},
                         args.out_dir, max_samples=6)

    save_samples(model, val_ld, device, {"amp": args.amp, "classes": args.classes},
                 args.out_dir, max_samples=12)
    logger.log_meta({"best_ckpt":best_path,"epochs":args.epochs,"batch_size":args.batch_size,
                     "lr_init":args.lr,"weight_decay":args.weight_decay,"scheduler":"ReduceLROnPlateau",
                     "model":args.model,"encoder":args.encoder,"encoder_weights":args.encoder_weights,
                     "classes":args.classes,"k_2p5d":args.k,"imagenet_norm":bool(args.imagenet_norm),
                     "aug":args.aug,"seed":args.seed})
    logger.close()
    print("Done. Best:", best_path)

if __name__ == "__main__":
    main()
