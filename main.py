import torch

# Device
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f'Device ===> {DEVICE}')

# Hyperparameters
BATCH_SIZE = 128
LEARNING_RATE = 0.01
NUM_EPOCHS = 100

