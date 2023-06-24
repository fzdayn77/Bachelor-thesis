import torch
from utils.prepare_data import get_data
from simCLR.models.encoder import get_encoder
from utils.utils import train_model, test_model

# Device
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f'Device ===> {DEVICE}')

# Hyperparameters
BATCH_SIZE = 128
LEARNING_RATE = 0.01
NUM_EPOCHS = 100
TEMPERATURE = 2.0
NUM_LAYERS = 4

# Encoder
encoder = get_encoder(model_name="forward-forward",
                      num_layers=NUM_LAYERS,
                      lr=LEARNING_RATE,
                      temperature=TEMPERATURE,
                      device=DEVICE)

# Data preparation
train_loader, test_loader = get_data(dataset_name="cifar10", batch_size=BATCH_SIZE)

# Executing
# TODO