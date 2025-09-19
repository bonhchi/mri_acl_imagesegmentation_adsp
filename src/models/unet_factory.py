# src/models/unet_factory.py
import segmentation_models_pytorch as smp

def build_unet(
    model="unet",
    encoder="resnet34",
    encoder_weights="none",   # "imagenet" | "ssl" | "none"
    in_ch=1,
    classes=1,
    **kw
):
    ew = None if str(encoder_weights).lower() in ("none", "null") else encoder_weights

    if model.lower() == "unet":
        return smp.Unet(
            encoder_name=encoder,
            encoder_weights=ew,
            in_channels=in_ch,
            classes=classes,
            **kw
        )

    if model.lower() in ("unetpp", "unetplusplus"):
        return smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights=ew,
            in_channels=in_ch,
            classes=classes,
            **kw
        )

    raise ValueError(f"Unsupported model: {model}")
