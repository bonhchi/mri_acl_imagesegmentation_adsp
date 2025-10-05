# -*- coding: utf-8 -*-
"""
Dataset đọc các volume .npz (được export từ preprocess của bạn):
  - img: (S,1,H,W) float32  | msk: (S,H,W) uint8/int
Trả về từng lát 2D (k=1) hoặc 2.5D (k odd: 3,5,...) theo dạng CHW + mask đúng format cho SMP.
Nếu muốn dùng pretrained ImageNet: bật imagenet_norm -> replicate về 3 kênh (k==1) và normalize theo encoder.
"""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except Exception:
    A, ToTensorV2 = None, None


def _read_list(txt_path: str):
    with open(txt_path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def _build_aug(level: str):
    if A is None:
        return None
    if level == "none":
        return A.Compose([ToTensorV2(transpose_mask=True)])
    if level == "light":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.05, rotate_limit=10, p=0.5, border_mode=0),
            ToTensorV2(transpose_mask=True),
        ])
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.ShiftScaleRotate(0.05, 0.10, 15, p=0.7, border_mode=0),
        ToTensorV2(transpose_mask=True),
    ])


class KneeNPZ2DSlices(Dataset):
    """
    list_txt: file txt, mỗi dòng là path tới 1 .npz (1 volume)
    k: 1=2D, 3/5=2.5D (stack láng giềng làm kênh)
    imagenet_norm: nếu True -> replicate 1->3 kênh (k==1) và normalize theo encoder
    """
    def __init__(self, list_txt: str, k: int = 1, aug: str = "light",
                 imagenet_norm: bool = False, encoder_name: str = "resnet34"):
        assert k >= 1 and k % 2 == 1, "k phải là số lẻ (1,3,5,...)"
        self.files = _read_list(list_txt)
        self.k = k
        self.aug = _build_aug(aug)
        self.imagenet_norm = imagenet_norm
        self.encoder_name = encoder_name

        # build index (file_idx, slice_idx)
        self.index = []
        self._sizes = []
        for fi, p in enumerate(self.files):
            with np.load(p) as z:
                S = int(z["img"].shape[0])
            self._sizes.append(S)
            self.index.extend([(fi, s) for s in range(S)])

        if self.imagenet_norm:
            import segmentation_models_pytorch as smp
            params = smp.encoders.get_preprocessing_params(self.encoder_name)
            self.mean = torch.tensor(params["mean"]).view(-1, 1, 1)
            self.std = torch.tensor(params["std"]).view(-1, 1, 1)

    def __len__(self):
        return len(self.index)

    def _load_volume(self, file_idx: int):
        path = self.files[file_idx]
        z = np.load(path)
        x = z["img"].astype(np.float32)  # (S,1,H,W)
        y = z["msk"].astype(np.int64)    # (S,H,W)
        return x, y

    def __getitem__(self, i: int):
        fi, s = self.index[i]
        img_vol, msk_vol = self._load_volume(fi)  # (S,1,H,W), (S,H,W)
        S = img_vol.shape[0]

        # 2D or 2.5D stack
        if self.k == 1:
            x = img_vol[s]                       # (1,H,W)
        else:
            half = self.k // 2
            idxs = [min(max(s + d, 0), S - 1) for d in range(-half, half + 1)]
            x = np.concatenate([img_vol[j] for j in idxs], axis=0)  # (k,H,W)

        y = msk_vol[s]                           # (H,W), int

        # Albumentations expects HWC
        x_hwc = np.moveaxis(x, 0, -1)  # CHW -> HWC
        y_np = y.copy()  # Sao luu mask dang numpy de xu ly
        if self.aug is not None:
            # Dung albumentations (tra ve tensor CHW) de cai thien mask
            out = self.aug(image=x_hwc, mask=y_np)
            img_aug, msk_aug = out["image"], out["mask"]
            if isinstance(img_aug, torch.Tensor):
                # Khi ToTensorV2 tra ve torch tensor -> giu nguyen kich thuoc CHW
                x_tensor = img_aug.float()
            else:
                # Neu pipeline khong dung ToTensorV2 thi can chuyen nguoc HWC -> CHW
                x_tensor = torch.from_numpy(np.moveaxis(img_aug, -1, 0).copy()).float()
            if isinstance(msk_aug, torch.Tensor):
                y_np = msk_aug.cpu().numpy()
            else:
                y_np = np.asarray(msk_aug)
        else:
            # Khong augment -> giu tensor CHW goc
            x_tensor = torch.from_numpy(x.copy()).float()
        y_np = np.ascontiguousarray(y_np)  # Dam bao bo nho lien tuc truoc khi tao tensor

        # Tach hai truong hop nhi phan va da lop de dinh dang mask dung y
        if y_np.max() <= 1:
            y_tensor = torch.from_numpy(y_np.copy()).unsqueeze(0).float()   # (1,H,W) cho nhi phan
        else:
            y_tensor = torch.from_numpy(y_np.copy()).long()                 # (H,W) cho da lop

        # Xu ly rieng truong hop encoder ImageNet can 3 kenh dau vao
        if self.imagenet_norm and x_tensor.shape[0] == 1:
            x_tensor = x_tensor.repeat(3, 1, 1)
        if self.imagenet_norm:  # Chuan hoa tensor theo thong so encoder
            x_tensor = (x_tensor - self.mean) / self.std

        return x_tensor.contiguous(), y_tensor  # Tra ve tensor da sap xep dung thu tu CHW
