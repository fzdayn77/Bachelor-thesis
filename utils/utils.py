import torch
from torch.utils.data import DataLoader

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


def goodness_score(pos_acts, neg_acts, threshold=2.0):
    """
    Computes the goodness score for a given set of positive and negative activations.

    Parameters:
      pos_acts (torch.Tensor): Numpy array of positive activations.
      neg_acts (torch.Tensor): Numpy array of negative activations.
      threshold (float, optional): Threshold value used to compute the score. Default is 2.0 .

    Returns:
      goodness (torch.Tensor): Goodness score computed as the sum of positive and negative goodness values. Note that this
      score is actually the quantity that is optimized and not the goodness itself. The goodness itself is the same
      quantity but without the threshold subtraction.
    """

    pos_goodness = -torch.sum(torch.pow(pos_acts, 2)) + threshold
    neg_goodness = torch.sum(torch.pow(neg_acts, 2)) - threshold
    return torch.add(pos_goodness, neg_goodness)