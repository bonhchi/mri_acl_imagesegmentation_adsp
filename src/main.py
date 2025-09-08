import torch

def main():
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))
        print("CUDA version:", torch.version.cuda)
        
        # test tính toán GPU
        x = torch.rand(10000, 10000, device="cuda")
        y = torch.rand(10000, 10000, device="cuda")
        z = torch.matmul(x, y)
        print("Matrix multiplication done on GPU, shape:", z.shape)
    else:
        print("⚠️ CUDA không khả dụng, đang chạy trên CPU")

if __name__ == "__main__":
    main()