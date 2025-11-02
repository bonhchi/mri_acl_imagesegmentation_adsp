# src/models/unet_factory.py
"""
Factory helpers for 2D UNet style models.

Tries to use segmentation_models_pytorch (preferred). If the optional dependency
is missing we fall back to MONAI's UNet to keep the training script usable on
setups where installing SMP is inconvenient.
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence, Tuple

try:
    import segmentation_models_pytorch as smp  # type: ignore
    _HAS_SMP = True
except ModuleNotFoundError:  # pragma: no cover - handled at runtime
    smp = None
    _HAS_SMP = False

try:
    from monai.networks.nets import UNet as MonaiUNet  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    MonaiUNet = None


def _normalise_channels(value: Any) -> Tuple[int, ...]:
    if value is None:
        return (16, 32, 64, 128, 256)
    if isinstance(value, int):
        raise TypeError("channels must be an iterable of integers.")
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    raise TypeError(f"Unsupported channels spec: {value!r}")


def _build_monai_unet(
    model: str,
    encoder: str,
    in_ch: int,
    classes: int,
    **kw: Any,
):
    if MonaiUNet is None:
        raise ImportError(
            "segmentation_models_pytorch is not installed and MONAI is unavailable. "
            "Install one of them (pip install segmentation-models-pytorch)."
        )

    channels = _normalise_channels(kw.pop("channels", None))
    strides = tuple(2 for _ in range(len(channels) - 1))
    kernel_size = kw.pop("kernel_size", 3)
    up_kernel_size = kw.pop("up_kernel_size", kernel_size)
    act = kw.pop("act", "PReLU")
    norm = kw.pop("norm", "INSTANCE")
    dropout = kw.pop("dropout", 0.0)

    if model.lower() not in {"unet", "unetpp", "unetplusplus"}:
        raise ValueError(f"Unsupported model '{model}' for MONAI fallback.")

    if kw:
        extra = ", ".join(sorted(kw))
        print(f"[warn] Ignoring unused build_unet kwargs in MONAI fallback: {extra}")

    return MonaiUNet(
        spatial_dims=2,
        in_channels=in_ch,
        out_channels=classes,
        channels=channels,
        strides=strides,
        kernel_size=kernel_size,
        up_kernel_size=up_kernel_size,
        act=act,
        norm=norm,
        dropout=dropout,
    )


def build_unet(
    model: str = "unet",
    encoder: str = "resnet34",
    encoder_weights: str = "none",  # "imagenet" | "ssl" | "none"
    in_ch: int = 1,
    classes: int = 1,
    **kw: Any,
):
    if _HAS_SMP:
        ew = None if str(encoder_weights).lower() in ("none", "null") else encoder_weights
        if model.lower() == "unet":
            return smp.Unet(
                encoder_name=encoder,
                encoder_weights=ew,
                in_channels=in_ch,
                classes=classes,
                **kw,
            )
        if model.lower() in ("unetpp", "unetplusplus"):
            return smp.UnetPlusPlus(
                encoder_name=encoder,
                encoder_weights=ew,
                in_channels=in_ch,
                classes=classes,
                **kw,
            )
        raise ValueError(f"Unsupported model: {model}")

    print(
        "[warn] segmentation_models_pytorch is not installed. "
        "Falling back to MONAI UNet (no pretrained encoders)."
    )
    return _build_monai_unet(model, encoder, in_ch, classes, **kw)
