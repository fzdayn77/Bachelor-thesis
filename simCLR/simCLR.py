import torch.nn as nn
from models.projection_head import Projection_Head

class SimCLR(nn.Module): 
    def __init__(self, encoder: nn.Module, n_features: int, projection_head: bool=False):
        super(SimCLR, self).__init__() 
        self.n_features = n_features
        self.projection_head = projection_head

        # encoder is either a ResNet or a Forward-Forward Network
        self.encoder = encoder

        if self.projection_head:
            # projector is a simple MLP with ReLU as activation function
            self.projector = Projection_Head(input_dim=self.n_features, hidden_dim=self.n_features,
                                            output_dim=self.n_features).forward

    def forward(self, x_i, x_j):
        # x -> h
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        if self.projection_head:
            # h -> z
            z_i = self.projector(h_i)
            z_j = self.projector(h_j)
            return h_i, h_j, z_i, z_j

        return h_i, h_j