import torch
from utils.prepare_data import get_data
from simCLR.models.encoder import get_encoder

# Device
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f'Device ===> {DEVICE}')

# Encoder
encoder = get_encoder(model_name="forward-forward")

# Hyperparameters
BATCH_SIZE = 128
LEARNING_RATE = 0.01
NUM_EPOCHS = 100
TEMPERATURE = 2.0

# Data preparation
#train_loader, test_loader = get_data(dataset_name="cifar10", batch_size=BATCH_SIZE)