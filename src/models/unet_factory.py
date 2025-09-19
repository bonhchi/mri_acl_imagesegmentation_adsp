# Factory tạo Unet/Unet++ từ segmentation_models_pytorch
import segmentation_models_pytorch as smp

def build_unet(model="unet", encoder="resnet34", encoder_weights="none",
               in_ch=1, classes=1):
    ew=None if str(encoder_weights).lower()=="none" else encoder_weights
    if model.lower()=="unet":
        return smp.Unet(encoder_name=encoder, encoder_weights=ew,
                        in_channels=in_ch, classes=classes)
    if model.lower() in ["unetpp","unetplusplus"]:
        return smp.UnetPlusPlus(encoder_name=encoder, encoder_weights=ew,
                                in_channels=in_ch, classes=classes)
    raise ValueError("Unsupported model")