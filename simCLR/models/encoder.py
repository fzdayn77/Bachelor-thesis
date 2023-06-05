from .resnet import get_resnet
from .forward_forward import Unsupervised_FF

def get_encoder(name):
    # All possible encoders
    encoders = ("resnet18", "resnet34", "resnet50", "forward-forward")
    encoder = Unsupervised_FF() # Default

    if name not in encoders:
        raise KeyError(f"{name} is not a valid encoder name")
      
    # ResNet18/34/50
    if name == "resnet18" or name == "resnet34" or name == "resnet50":
        encoder = get_resnet(name=name)
    
    print(f'Encoder ====> {name}')
    return encoder
