import torch.nn as nn
from models.projection_head import Projection_Head

class SimCLR(nn.Module):
    """
    Implementation of the simCLR model
    """
    
    def __init__(self, encoder, n_features, projection_dim):
        super(SimCLR, self).__init__()
        
        self.n_features = n_features
        self.projection_dim = projection_dim

        # encoder is either a ResNet or a Forward-Forward Network
        self.encoder = encoder

        # projector is a simple MLP with ReLU as activation function
        self.projector = Projection_Head(input_dim=n_features, hidden_dim=n_features, output_dim=projection_dim).forward

    def forward(self, x_i, x_j):
        # x -> h -> z
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        return h_i, h_j, z_i, z_j