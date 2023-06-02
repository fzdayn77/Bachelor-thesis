from .resnet import get_resnet
from .forward_forward import Forward_Forward

def get_encoder(name):
    # All possible encoders
    encoders = ("resnet18", "resnet34", "resnet50", "forward-forward")

    if name not in encoders:
        raise KeyError(f"{name} is not a valid encoder name")
    
    # ResNet18/34/50
    if name == "resnet18" or name == "resnet34" or name == "resnet50":
        return get_resnet(name=name, pretrained=False)
    
    # Forward-Forward
    if name == "forward-forward":
        return Forward_Forward()
