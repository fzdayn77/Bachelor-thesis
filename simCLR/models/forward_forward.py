import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.utils import goodness_score, get_metrics
from tqdm.auto import tqdm
from utils.prepare_data import get_pos_neg_data, separate_pos_pairs

# Hyperparameters for now !!!
batch_size = 512
augmented_batch_size = batch_size*2
num_epochs = 100
lr = 0.03
input_size = 32*32*3 # Height=Width=32 and channels=3
output_size = 10

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

class FF_Layer(nn.Linear):
    def __init__(self, in_features: int, out_features: int, loss_function=None, optimizer=None, device=None):
        super(FF_Layer, self).__init__(in_features, out_features, device=device)
        self.in_features = in_features
        self.out_features = out_features
        self.opt = torch.optim.Adam(self.parameters(), lr=0.003)
        self.loss = loss_function
        self.device = device
        self.relu = torch.nn.ReLU()
        self.to(self.device)

    def forward(self, x):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        x_direction = torch.flatten(x_direction)
        return self.relu(torch.mm(x_direction, self.weight.T) + self.bias.unsqueeze(0))

    def train_layer(self, x_pos, x_neg):
        self.opt.zero_grad()
        goodness = goodness_score(x_pos, x_neg)
        goodness.backward()
        self.opt.step()
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()
    

class FF_Net(nn.Module):
    def __init__(self, num_layers: int = 4, lr: float = 0.003, device=None):
        super(FF_Net, self).__init__()
        self.num_layers = num_layers
        self.device = device
        self.lr = lr
        self.loss = nn.CrossEntropyLoss(reduction="mean")
        #self.opt = torch.optim.Adam(self.parameters(), lr=self.lr)

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
                    loss_function=self.loss,
                    #optimizer=self.opt,
                    device=self.device) for idx in range(self.num_layers)
        ]
        self.ff_layers = ff_layers

    def forward(self, x):
        # Apply the feature extraction layers
        features = self.features(x)
        pooled_features = self.avgpool(features)
        flattened_features = torch.flatten(pooled_features, start_dim=1)

        # Pass the flattened features through the FF layers
        for layer in self.ff_layers:
            flattened_features = layer(flattened_features)

        return flattened_features


    def train_net(self, x_pos, x_neg):
        h_pos, h_neg = x_pos, x_neg
        h_pos = h_pos.to(self.device)
        h_neg = h_neg.to(self.device)

        # Features Extraction
        h_pos = self.forward(h_pos)
        h_neg = self.forward(h_neg)

        for idx, layer in enumerate(self.ff_layers):
            print(f"Training Layer {idx+1} ...")
            h_pos, h_neg = layer.train_layer(h_pos, h_neg)

        print("Done!")