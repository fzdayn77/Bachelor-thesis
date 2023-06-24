from .resnet import get_resnet
from .forward_forward import FF_Net

def get_encoder(model_name: str, num_layers: int, lr: float, temperature: float, device=None):
    '''
    Chooses an encoder from the given models.

    Parameters:
        name (String): name of the model. It can be one of four possible
                       models [ResNet18. ResNet34, ResNet50, Forward-Forward].
        device: Default is None.

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
        encoder = FF_Net(num_layers, lr, temperature, device=device)
    
    print(f'Encoder ===> {model_name}')

    return encoder
