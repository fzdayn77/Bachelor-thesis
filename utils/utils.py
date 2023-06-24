import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
from tqdm.auto import tqdm


def separate_pos_pairs(dataloader : DataLoader, batch_size : int):
  '''
  Creates a list of positive data pairs. Positive pairs are two augmented images of the same image.

  Parameters:
    dataloader (Dataloader): it can be the trainloader or testloader.
    batch_size (int): is the same batch size used in the Dataloaders .

  Returns:
    List: a list of positive pairs.

  Example:
    dataloader has for example three different Tensors: x, y and z.
    For each one two augmentations will be generated: x --> x_i, x_j
                                                      y --> y_i, y_j
                                                      z --> z_i, z_j
    The returned list will look like this: [ [x_i, x_j] , [y_i, y_j] , [z_i, z_j] ] where each pair
    in the list is a positive pair.
  '''
  
  # Stack all augmentions
  # len(stacked_imgs) = batch_size * 2
  stacked_imgs = torch.stack([img for idx in range(batch_size) for img in dataloader.dataset[idx][0]], dim=0)
  output_list = []

  for idx in range(len(stacked_imgs)):
    # Consider even indices only
    if idx % 2 != 0: continue

    # List of a positive pair
    pair = []
    pair.append(stacked_imgs[idx])
    pair.append(stacked_imgs[idx+1])

    # Add positive pair to the list
    output_list.append(pair)

  return output_list


def get_pos_neg_data(pairs_list, idx : int):
  '''
  Separates positive data from negative data.

  Parameters:
    pairs_list (List of Tensors): contains pairs of Tensors.
    idx (int): index of the positive pair that will nbe separated
               from the rest.

  Returns:
    pos_data (List of Tensors): list of positive data.
    neg_data (List of Tensors): list of negative data.
  '''
  
  # List of two positive Tensors
  pos_data = pairs_list[idx]

  # Negative Data
  neg_data = []
  for i in range(len(pairs_list)):
    # Ignore pos_data
    if i == idx:
      continue
    neg_data.append(pairs_list[i][0])
    neg_data.append(pairs_list[i][1])

  return pos_data, neg_data


def generate_sub_batches(list):
  """
  Generates two sub-batches from the same augmented mini-batch.

  Parameters:
    list (List): the augmented mini-batch (size = original batch_size * 2)

  Returns:
    sub_1 (List): the first sub-batch (size= original batch_size)
    sub_2 (List): the second sub-batch (size= original batch_size)

  Example:
    list = [ x_i, x_j, y_i, y_j, z_i, z_j ] => sub_1 = [ x_i, y_i, z_i ]
                                               sub_2 = [ x_j, y_j, z_j ] 
  """
  sub_1, sub_2 = [], []
  for idx in range(len(list)):
    if idx % 2 != 0: continue
    sub_1.append(list[idx])
    sub_2.append(list[idx+1])

  return sub_1, sub_2



def compute_accuracy(model, data_loader, device):
    """
    Calculates the accuracy of a given data loader.

    Parameters:
        model (nn.Module): the used CNN model
        data_loader (DataLoader): train loader or test loader
        device: the used device
    
    Returns:
        (float): the calculated accuracy
    """
    with torch.no_grad():
        correct_pred, num_examples = 0, 0
        for i, (features, targets) in enumerate(data_loader):

            features = features.to(device)
            targets = targets.float().to(device)

            logits = model(features)
            _, predicted_labels = torch.max(logits, 1)

            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()

    return (correct_pred.float()/num_examples) * 100


def train_model(model: nn.Module, num_epochs: int, train_loader: DataLoader, device=None):
  start_time = time.time()
  minibatch_loss_list, train_acc_list = [], []
  print("Begin training ...")

  for epoch in range(num_epochs):
    # Putting the model in training mode
    model.train()

    inner_tqdm = tqdm(train_loader, desc=f"Training FF Layers | Epoch {epoch+1}/{num_epochs}", leave=True, position=0)

    for i, (mini_batch_imgs, _) in enumerate(inner_tqdm):
      avg_net_loss = model(mini_batch_imgs)
      minibatch_loss_list.append(avg_net_loss)


  # Total time
  elapsed = (time.time() - start_time) / 60
  print(f'Total training time: {elapsed:.2f} min')
  print("Training Done!\n")

  return minibatch_loss_list, train_acc_list


def test_model(model: nn.Module, test_loader: DataLoader, num_epochs, loss_function, optimizer, device=None):
  start_time = time.time()
  test_acc_list = []
  print("Begin testing ...")

  for epoch in range(num_epochs):
    # Putting the model in evaluation mode
    model.eval()

    inner_tqdm = tqdm(test_loader, desc=f"Testing | Epoch {epoch+1}/{num_epochs} ", leave=False, position=0)

    for i, (mini_batch_imgs, _) in enumerate(inner_tqdm):
      avg_net_loss = model(mini_batch_imgs)

      optimizer.zero_grad()
      avg_net_loss.backward()
      optimizer.step()

    with torch.no_grad():
      print("Calculating accuracy ....")
      test_acc = compute_accuracy(model, test_loader, device=device)
      print(f"Accuracy after {epoch+1} epoch(s) ===> {test_acc:.2f} %")
      test_acc_list.append(test_acc.item())

  # Total time
  elapsed = (time.time() - start_time) / 60
  print(f'Total Testing Time: {elapsed:.2f} min')
  print("Testing Done!\n")

  return test_acc_list

