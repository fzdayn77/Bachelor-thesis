import torch
import torch.nn as nn
import numpy as np
from utils.nt_xent_loss import nt_xent_loss


class FF_Layer(nn.Linear):
  def __init__(self, in_features: int, out_features: int, temperature: float, optimizer=None, device=None):
    super(FF_Layer, self).__init__(in_features, out_features, device=device)
    self.in_features = in_features
    self.out_features = out_features
    self.device = device
    self.relu = torch.nn.ReLU()
    self.temperature = temperature

    if optimizer is None:
      # Default optimizer
      self.opt = torch.optim.Adam(self.parameters(), lr=0.1)
    else:
      self.opt = optimizer

  def forward(self, x_1, x_2):
    self.opt.zero_grad()
    loss = nt_xent_loss(x_1, x_2, temperature=self.temperature)

    # TODO : retain_graph must be Fakse !!
    loss.backward(retain_graph=True)
    self.opt.step()

    return loss


class FF_Net(nn.Module):
  def __init__(self, num_layers: int, lr: float, temperature: float, device=None):
    super(FF_Net, self).__init__()
    self.num_layers = num_layers
    self.device = device
    self.lr = lr
    self.temperature = temperature

    # Features Extraction
    self.features = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=3, stride=1),

        torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=3, stride=1),

        torch.nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=3, stride=1)
    )

    # Average Pooling
    self.avgpool = torch.nn.AdaptiveAvgPool2d((4, 4))

    # Forward-Forward Layers
    ff_layers = [
        FF_Layer(in_features=192*4*4 if idx == 0 else 2000,
                 out_features=2000,
                 temperature=self.temperature,
                 device=self.device) for idx in range(self.num_layers)
    ]
    self.ff_layers = ff_layers

  def forward(self, x):
    x_1, x_2 = x[0], x[1]

    if self.device is not None:
      x_1 = x_1.to(self.device)
      x_2 = x_2.to(self.device)

    # x_1 and x_2 shapes ==> torch.Size([128, 3072])
    # 3072 = 32*32*3
    x_1 = self.features(x_1)
    x_1 = self.avgpool(x_1)
    x_1 = torch.flatten(x_1, start_dim=1)

    x_2 = self.features(x_2)
    x_2 = self.avgpool(x_2)
    x_2 = torch.flatten(x_2, start_dim=1)

    # Pass the flattened features through the FF layers
    net_losses = []
    for layer in self.ff_layers:
        layer_loss = layer(x_1, x_2)
        net_losses.append(layer_loss.item())

    return np.asarray(net_losses).mean()