from .resnet import get_resnet
from .forward_forward import FF_Net

def get_encoder(model_name: str):
    '''
    Chooses an encoder from the given models.

    Parameters:
        name (String): name of the model. It can be one of four possible
                       models [ResNet18. ResNet34, ResNet50, Forward-Forward].
                       Default name: "forward-forward"
    Returns:
        encoder (nn.Module): the chosen model.
    '''

    # All possible encoders
    encoders = ("resnet18", "resnet34", "resnet50", "forward-forward")
    if model_name not in encoders:
        raise KeyError(f"{model_name} is not a valid encoder name")
      
    # ResNet18/34/50
    if model_name == "resnet18" or model_name == "resnet34" or model_name == "resnet50":
        encoder = get_resnet(name=model_name)
    else:
        encoder = FF_Net
    
    print(f'Encoder ====> {model_name}')

    return encoder
