import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.utils import goodness_score, get_metrics
from tqdm import tqdm
from utils.prepare_data import get_pos_neg_data, separate_pos_pairs

batch_size = 512
augmented_batch_size = batch_size*2
num_epochs = 100
lr = 0.03
input_size = 32*32*3 # Height=Width=32 and channels=3
output_size = 10

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class FF_Layer(nn.Linear):
    def __init__(self, in_features: int, out_features: int, n_epochs: int, bias: bool, device=None):
        super().__init__(in_features, out_features, bias=bias)
        self.n_epochs = n_epochs
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.goodness = goodness_score
        self.to(device)
        self.ln_layer = nn.LayerNorm(normalized_shape=[out_features]).to(device)

    def ff_train(self, pos_acts, neg_acts):
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
    

class Unsupervised_FF(nn.Module):
    def __init__(self, n_layers: int = 4, n_neurons=3072, input_size: int = input_size, n_epochs: int = 100,
                 bias: bool = True, n_classes: int = output_size, n_hid_to_log: int = 3, device=device):
        super().__init__()
        self.n_hid_to_log = n_hid_to_log
        self.n_epochs = n_epochs
        self.device = device

        ff_layers = [
            FF_Layer(in_features=input_size if idx == 0 else n_neurons,
                     out_features=n_neurons,
                     n_epochs=n_epochs,
                     bias=bias,
                     device=device) for idx in range(n_layers)]

        self.ff_layers = nn.ModuleList(ff_layers)
        self.last_layer = nn.Linear(in_features=n_neurons * n_hid_to_log, out_features=n_classes, bias=bias)
        self.to(device)
        self.opt = torch.optim.Adam(self.last_layer.parameters())
        self.loss = torch.nn.CrossEntropyLoss(reduction="mean")

    def train_ff_layers(self, dataloader: DataLoader):
        #outer_tqdm = tqdm(range(self.n_epochs), desc="Training FF Layers", position=0)
        print("Training FF layers .....")
        for epoch in range(self.n_epochs):
            inner_tqdm = tqdm(dataloader, desc=f"Training FF Layers | Epoch {epoch+1}/{self.n_epochs}",
                               leave=False, position=0)
            
            for mini_batch in inner_tqdm:
                pos_pairs_list = separate_pos_pairs(dataloader, len(mini_batch))
                
                for idx in range(len(pos_pairs_list)):
                  pos_data_list, neg_data_list = get_pos_neg_data(pos_pairs_list, idx=idx)

                  for i in range(2):
                    pos_data = pos_data_list[i].to(device)

                    for j in range(len(neg_data_list)):
                      neg_data = neg_data_list[j].to(device)
                      #pos_acts = torch.reshape(pos_data, (len(pos_data), 1, -1)).to(self.device)
                      #neg_acts = torch.reshape(neg_data, (len(neg_data), 1, -1)).to(self.device)

                      for idx, layer in enumerate(self.ff_layers):
                          pos_acts = layer(pos_data)
                          neg_acts = layer(neg_data)
                          layer.ff_train(pos_acts, neg_acts) # type: ignore

    def train_last_layer(self, dataloader: DataLoader):
        print('')
        print("Training last layer .....")
        num_examples = len(dataloader)
        #outer_tqdm = tqdm(range(self.n_epochs), desc="Training Last Layer", position=0)
        loss_list = []
        for epoch in range(self.n_epochs):
            epoch_loss = 0
            inner_tqdm = tqdm(dataloader, desc=f"Training Last Layer | Epoch {epoch+1}/{self.n_epochs}", leave=False, position=0)
            for images, labels in inner_tqdm:
                for i in range(len(images)):
                  images = images[i].to(self.device)
                  images = torch.reshape(images, (images.shape[0], 1, -1))
                  labels = labels.to(self.device)
                  self.opt.zero_grad()
                  preds = self(images)
                  loss = self.loss(preds, labels[i:i+10])
                  epoch_loss += loss
                  loss.backward()
                  self.opt.step()
            loss_list.append(epoch_loss / num_examples)
            # Update progress bar with current loss
        return [l.detach().cpu().numpy() for l in loss_list]

    def forward(self, image: torch.Tensor):
        image = image.to(self.device)
        image = torch.reshape(image, (image.shape[0], 1, -1))
        concat_output = []
        for idx, layer in enumerate(self.ff_layers):
            image = layer(image)
            if idx > len(self.ff_layers) - self.n_hid_to_log - 1:
                concat_output.append(image)
        concat_output = torch.concat(concat_output, 2)
        logits = self.last_layer(concat_output)
        return logits.squeeze()

    def evaluate(self, dataloader: DataLoader, dataset_type: str = "train"):
        self.eval()
        inner_tqdm = tqdm(dataloader, desc=f"Evaluating model", leave=False, position=1)
        all_labels = []
        all_preds = []
        for images, labels in inner_tqdm:
            images = images.to(self.device)
            labels = labels.to(self.device)
            preds = self(images)
            preds = torch.argmax(preds, 1)
            all_labels.append(labels.detach().cpu())
            all_preds.append(preds.detach().cpu())
        all_labels = torch.concat(all_labels, 0).numpy()
        all_preds = torch.concat(all_preds, 0).numpy()
        metrics_dict = get_metrics(all_preds, all_labels)
        print(f"{dataset_type} dataset scores: ", "\n".join([f"{key}: {value}" for key, value in metrics_dict.items()]))