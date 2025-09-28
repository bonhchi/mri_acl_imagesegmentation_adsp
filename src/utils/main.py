from pathlib import Path

import numpy as np
import torch

from src.adapters.fastmri_adapter import FastMRISinglecoilAdapter
from src.preprocess.mri_preprocess import MRIKneePreprocessor
from src.models.unet_factory import build_unet
from src.train.losses import LossManager


def main():
    # --- Adapter ---
    project_root = Path(__file__).resolve().parents[2]
    singlecoil_root = project_root / "dataset" / "singlecoil_train"
    adapter = FastMRISinglecoilAdapter(root_dir=str(singlecoil_root))
    records = adapter.discover_records(adapter.root_dir)
    sample = adapter.load_record(records[0])

    # --- Preprocess ---
    pre = MRIKneePreprocessor(out_size=(320, 320))
    out = pre.preprocess_record(sample)

    x = out["img_z"]   # numpy (H,W) or (C,H,W)
    y = out["mask"]    # numpy (H,W)

    # --- Tensor ---
    x_t = torch.from_numpy(x).unsqueeze(0).float()  # (1,C,H,W)
    if y.ndim == 2:
        y_t = torch.from_numpy(y).unsqueeze(0).unsqueeze(0).float()  # (1,1,H,W)
    else:
        y_t = torch.from_numpy(y).unsqueeze(0).float()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Model ---
    in_ch = x_t.shape[1]  # C
    model = build_unet(
        model="unet",
        encoder="resnet34",
        encoder_weights="none",
        in_ch=in_ch,
        classes=1,
    ).to(device)

    # --- Loss ---
    loss_fn = LossManager(classes=1, name="dice_bce")

    # --- Forward ---
    model.eval()
    with torch.no_grad():
        logits = model(x_t.to(device))
        loss = loss_fn(logits, y_t.to(device))
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

    print("Input:", x_t.shape)
    print("Logits:", logits.shape)
    print("Loss:", float(loss.item()))
    print("Pred mask:", preds.shape, preds.min().item(), preds.max().item())


if __name__ == "__main__":
    main()
