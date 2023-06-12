import torch
from data_augmentation import train_data_augmentation, test_data_augmentation, GaussianBlur
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


def get_dataloaders(dataset_name, path, batch_size=128):
    """
    Imports trainset and testset of the chosen Dataset and returns the corresponding trainloader and testloader.
    """

    # Trainset and Testset
    if dataset_name == "cifar10":
        path="/data/CIFAR-10-augmented"
        train_set = datasets.CIFAR10(root=path, train=True, download=True, 
                                    transform=train_data_augmentation(normalize=transforms.Normalize(
                                        mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))))

        test_set = datasets.CIFAR10(root=path, train=False, download=True, 
                                   transform=test_data_augmentation(crop=True))

    elif dataset_name == "cifar100":
        path="/data/CIFAR-100-augmented"
        train_set = datasets.CIFAR100(root=path, train=True, download=True, 
                                    transform=train_data_augmentation(normalize=transforms.Normalize(
                                        mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))))

        test_set = datasets.CIFAR100(root=path, train=False, download=True, 
                                   transform=test_data_augmentation(crop=True))

    elif dataset_name == "imageNet":
        path="/data/ImageNet-augmented"
        train_set = datasets.ImageNet(root=path, train=True, download=True, 
                                    transform=train_data_augmentation(normalize=transforms.Normalize(
                                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))))

        test_set = datasets.ImageNet(root=path, train=False, download=True, 
                                   transform=test_data_augmentation(crop=True))

    else:
        raise KeyError(f"Choose a datset name (cifar10 or cifar100 or imageNet)")

    # Trainloaderand Testloader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_set, train_loader, test_set, test_loader


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