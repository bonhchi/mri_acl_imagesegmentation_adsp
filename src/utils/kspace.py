# src/utils/kspace.py
import numpy as np

def fft2c(x: np.ndarray) -> np.ndarray:
    """Centered 2D FFT (chuẩn L2)."""
    return np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(x, axes=(-2, -1)), norm="ortho"),
        axes=(-2, -1)
    )

def ifft2c(x: np.ndarray) -> np.ndarray:
    """Centered 2D iFFT (chuẩn L2)."""
    return np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(x, axes=(-2, -1)), norm="ortho"),
        axes=(-2, -1)
    )

def complex_abs(x: np.ndarray) -> np.ndarray:
    """|x| cho mảng phức."""
    return np.sqrt((x.real ** 2) + (x.imag ** 2))

def center_crop_or_pad(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Crop hoặc pad ảnh (H,W) vào giữa để ra đúng (out_h,out_w)."""
    h, w = img.shape[-2], img.shape[-1]
    out = np.zeros((*img.shape[:-2], out_h, out_w), dtype=img.dtype)

    hmin, wmin = min(h, out_h), min(w, out_w)
    h0, w0 = (h - hmin) // 2, (w - wmin) // 2         # start in src
    H0, W0 = (out_h - hmin) // 2, (out_w - wmin) // 2 # start in dst
    out[..., H0:H0+hmin, W0:W0+wmin] = img[..., h0:h0+hmin, w0:w0+wmin]
    return out