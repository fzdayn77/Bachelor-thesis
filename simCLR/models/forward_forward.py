import torch
import torch.nn as nn
from utils.utils import goodness_score


class FF_Layer(nn.Linear):
    def __init__(self, in_features: int, out_features: int, n_epochs: int, bias: bool, device=None):
        super().__init__(in_features, out_features, bias=bias)
        
        self.in_features = in_features
        self.out_features = out_features
        self.n_epochs = n_epochs

        # Optimizer
        self.opt = torch.optim.Adam(self.parameters(), lr=0.03)

        # Goodness
        self.goodness = goodness_score

        self.to(device)
        self.ln_layer = nn.LayerNorm(normalized_shape=[1, out_features]).to(device)

    def train_FF_Layer(self, pos_acts, neg_acts):
        """
        Train the layer using positive and negative activations.

        Parameters:
          pos_acts (numpy.ndarray): Numpy array of positive activations.
          neg_acts (numpy.ndarray): Numpy array of negative activations.
        """

        self.opt.zero_grad()
        goodness = self.goodness(pos_acts, neg_acts)
        goodness.backward()
        self.opt.step()

    def forward(self, input):
        input = super().forward(input)
        input = self.ln_layer(input.detach())
        return input