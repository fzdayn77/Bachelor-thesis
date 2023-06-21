import torch
import torch.nn as nn
from utils.nt_xnet_loss import nt_xnet_loss
from utils.utils import generate_sub_batches


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
      self.opt = torch.optim.Adam(self.parameters(), lr=0.01)
    else:
      self.opt = optimizer

  def train_layer(self, x_1, x_2):
    loss_matrix = []

    for _, t_1 in enumerate(x_1):
      if self.device is not None:
        t_1 = t_1.to(self.device)
      #print(t_1)
      
      loss_list = []

      for _, t_2 in enumerate(x_2):
        if self.device is not None:
          t_2 = t_2.to(self.device)
        #print(t_2)
        
        loss = nt_xnet_loss(t_1, t_2, temperature=self.temperature)
        loss_list.append(self.relu(loss))

      loss_matrix.append(loss_list)

    return loss_matrix

  def forward(self, x_1, x_2):
    self.opt.zero_grad()
    loss_matrix = self.train_layer(x_1, x_2)
    loss_matrix[0][0].backward()
    self.opt.step()

    return loss_matrix
  

class FF_Net(nn.Module):
  def __init__(self, num_layers: int, lr: float, device=None):
    super(FF_Net, self).__init__()
    self.num_layers = num_layers
    self.device = device
    self.lr = lr
    self.temperature = 2.0

    # Features Extraction
    self.features = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=3, stride=1),

        torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=3, stride=1),

        torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=3, stride=1),

        torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=3, stride=1)
    )

    # Average Pooling
    self.avgpool = torch.nn.AdaptiveAvgPool2d((2, 2))

    # Forward-Forward Layers
    ff_layers = [
        FF_Layer(in_features=256*2*2 if idx == 0 else 2000,
                 out_features=2000,
                 temperature=self.temperature,
                 device=self.device) for idx in range(self.num_layers)
    ]
    self.ff_layers = ff_layers

  def forward(self, x):
    if self.device is not None:
      x = x.to(self.device)
    
    x = self.features(x)
    x = self.avgpool(x)
    x = torch.flatten(x, start_dim=1)
    sub_1, sub_2 = generate_sub_batches(x)

    # Pass the flattened features through the FF layers
    net_loss_matrices = []
    for layer in self.ff_layers:
        loss_matrix = layer(sub_1, sub_2)
        net_loss_matrices.append(loss_matrix)

    return net_loss_matrices