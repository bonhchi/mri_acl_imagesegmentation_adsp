# Image Segmentation MRI – Tài liệu hướng dẫn

Tài liệu này được chuyển hóa từ `src/guide.txt` và tóm tắt toàn bộ quy trình làm việc cho dự án.

---

## 1. Chuẩn bị môi trường

```powershell
cd /d D:\Master\ImageSegmentation\Demo
python -m venv .venv
.\.venv\Scripts\activate

python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
# nếu cần thêm
pip install python-dotenv
```

Kiểm tra nhanh:

```powershell
python - <<'PY'
import torch, monai
print("Torch:", torch.__version__, "CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("MONAI:", monai.__version__)
PY
```

## 2. Cấu trúc thư mục

| Thư mục      | Vai trò                                                     |
| ------------ | ----------------------------------------------------------- |
| `artifacts/` | Kết quả preprocess (`volume.npz`, `tensor.pt`, `preview`, `preview_mask`) |
| `dataset/`   | Dữ liệu thô (fastMRI `.h5`, KneeMRI `.pck`, OAI-ZIB `.npz`) |
| `lists/`     | `train.txt`, `val.txt` sử dụng khi train                    |
| `runs/`      | Checkpoint (`best.pt`), lịch sử, ảnh mẫu                    |
| `src/`       | Mã nguồn adapters, preprocess, training, inference          |
| `.env`       | (tùy chọn) Khai báo `FASTMRI_ROOT`, `KNEE_MRI_ROOT`, ...    |

## 3. Tiền xử lý dữ liệu

### fastMRI

```powershell
python src/train_unet_launcher.py `
  --dataset fastmri `
  --dataset-root dataset/singlecoil_train `
  --artifact-dir artifacts/fastmri_knee `
  --preview-max 6
```

### KneeMRI

```powershell
python src/train_unet_launcher.py `
  --dataset kneemri `
  --dataset-root dataset/kneemri `
  --artifact-dir artifacts/kneemri_acl `
  --slice-keep 0.0,1.0 `
  --preview-max 6
```

### OAI-ZIB

```powershell
python src/train_unet_launcher.py `
  --dataset oaizib `
  --dataset-root dataset/OAI-ZIB-framelast `
  --artifact-dir artifacts/artifacts/oaizib_knee `
  --skip-split `
  --skip-train
```

> Tip: khi volume.npz đã tồn tại, thêm `--skip-preprocess`. Dataset "combine" chỉ cần ở bước train.

## 4. Tạo danh sách train/val

### Dùng launcher

```powershell
python src/train_unet_launcher.py --dataset <dataset> --skip-preprocess --skip-train
```

### Thủ công

```powershell
for /r artifacts\fastmri_knee %f in (volume.npz) do @echo %f>>lists\all.txt
python - <<'PY'
import random, pathlib
L = [ln.strip() for ln in pathlib.Path("lists/all.txt").read_text().splitlines() if ln.strip()]
random.seed(42); random.shuffle(L)
k = int(len(L) * 0.8)
pathlib.Path("lists/train.txt").write_text("\n".join(L[:k]), encoding="utf-8")
pathlib.Path("lists/val.txt").write_text("\n".join(L[k:]), encoding="utf-8")
PY
```

> Lưu ý: sau khi tạo, nên thêm tiền tố `dataset|` cho mỗi dòng (ví dụ `fastmri|D:\...\volume.npz`) hoặc dùng `python src/generate_train_val.py` để sinh đúng định dạng mới.

## 5. Huấn luyện U-Net 2D/2.5D

### Pipeline đầy đủ

```powershell
python src/train_unet_launcher.py `
  --dataset fastmri `
  --dataset-root dataset/singlecoil_train `
  --artifact-dir artifacts/fastmri_knee `
  --out-dir runs/fastmri_unet `
  --epochs 80 `
  --batch-size 8 `
  --workers 4 `
  --amp
```

### Huấn luyện OAI-ZIB với danh sách đã chuẩn bị

```powershell
python src/train_unet_launcher.py `
  --dataset oaizib `
  --artifact-dir artifacts/artifacts/oaizib_knee `
  --train-list lists/oaizib_knee/train.txt `
  --val-list lists/oaizib_knee/val.txt `
  --skip-preprocess `
  --skip-split `
  --out-dir runs/oaizib_unet `
  --epochs 80 `
  --batch-size 64 `
  --workers 8 `
  --run-tag oaizib `
  --prefetch-gpu `
  --prefetch-factor 4 `
  --persistent-workers `
  --cache-mode cpu `
  --amp
```

> Ghi chú:
> - `--skip-preprocess` và `--skip-split` chỉ dùng khi `volume.npz` và danh sách train/val đã tạo từ bước 3-4.
> - `--cache-mode cpu` hữu ích khi artifact đặt trên HDD; có thể đổi sang `mmap` hoặc `gpu` nếu phần cứng cho phép.
> - `--prefetch-gpu` tăng tốc độ nạp batch khi VRAM còn trống; đi kèm `--prefetch-factor 4` và `--persistent-workers` để giảm thời gian chờ DataLoader.

Tùy chọn thường dùng: `--skip-preprocess`, `--skip-split`, `--model unetpp`, `--encoder densenet121`, `--prefetch-gpu`, `--prefetch-factor 4`, `--persistent-workers`, `--cache-mode cpu`, `--auto-gpu`, `--run-tag exp1`.
Nếu GPU còn ≥12GB VRAM, `--auto-gpu` sẽ tự bật mixed precision, prefetch và tăng batch size/worker để tận dụng tài nguyên; khi VRAM thấp hơn cấu hình sẽ giữ nguyên. Khi dữ liệu nằm trên ổ đĩa chậm, cân nhắc `--cache-mode cpu` để giữ volume trong RAM (cần thêm RAM trống).

## 6. Huấn luyện U-Net 3D

```powershell
python src/train/train_unet3d.py `
  --train-list lists/fastmri_knee/train.txt `
  --val-list lists/fastmri_knee/val.txt `
  --out-dir runs/unet3d `
  --patch-size 160 160 64 `
  --patches-per-volume 12 `
  --batch-size 2 `
  --epochs 80 `
  --cache-mode gpu `
  --prefetch-gpu `
  --amp
```

Ghi chú: list/\*.txt nên gắn nhãn `dataset|path`. Nếu thiếu, loader vẫn suy ra từ đường dẫn nhưng nên chuẩn hóa để kết hợp nhiều nguồn.

Các tùy chọn đáng lưu ý: `--pos-frac`, `--eval-overlap`, `--channels`, `--normalize`, `--cache-mode {cpu|mmap|none|gpu}`, `--auto-gpu`, `--prefetch-gpu`, `--run-tag`.
Hiệu năng:

- GPU: `--cache-mode gpu` giữ volume trên VRAM, nhanh hơn nhưng tốn bộ nhớ, DataLoader sẽ ép `workers=0`.
- RAM: Dùng `cpu` hoặc `mmap` khi RAM thấp; cân nhắc bỏ `--prefetch-gpu`.
- CPU/disk: `none` phù hợp khi có NVMe nhanh, đổi lại CPU cao hơn.

Khi GPU còn trống ≥28GB VRAM, `--auto-gpu` sẽ tự chuyển `cache_mode=gpu`, tăng batch size + val batch, bật prefetch và giữ worker hợp lý. Nếu VRAM thấp hơn, cấu hình được giữ nguyên để tránh thiếu bộ nhớ.

## 7. Theo dõi & Checkpoints

- Checkpoint tốt nhất: `runs/<dataset>_unet/<run_name>/best.pt`
- Log: `history.csv`, `history.json`, ảnh mẫu trong `samples/`
- TensorBoard (nếu có):

```powershell
tensorboard --logdir runs/<dataset>_unet
```

## 8. Inference nhanh

```powershell
python src/infer.py `
  --ckpt runs/fastmri_unet/2025-10-26_unet_resnet34/best.pt `
  --volume artifacts/fastmri_knee/file1000001/volume.npz `
  --out out/predict
```

## 9. Huấn luyện dữ liệu kết hợp (combine)

Chuẩn bị `train_combine.txt`, `val_combine.txt` gồm nhiều nguồn khác nhau.

```powershell
python src/train_unet_launcher.py `
  --dataset combine `
  --artifact-dir artifacts/combine `
  --train-list lists/train_combine.txt `
  --val-list lists/val_combine.txt `
  --skip-preprocess `
  --skip-split `
  --run-tag combine `
  --epochs 80
```

Thư mục kết quả sẽ có hậu tố `_combine`.
