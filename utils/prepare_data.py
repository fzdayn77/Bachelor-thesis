from utils.data_augmentation import train_data_augmentation, test_data_augmentation
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


def get_data(dataset_name: str, batch_size: int, path=None):
    """
    Imports trainset and testset of the chosen Dataset and returns the corresponding trainloader and testloader.

    Parameters:
        dataset_name (str): name of the dataset
        path (str): path to where the data will be downloaded
        batch_size (int): size of the mini-batch

    Returns:
        train_loader (DataLoader): the training loader
        test_loader (DataLoader): the testing loader
    """

    # Trainset and Testset
    if dataset_name == "cifar10":
        if path is None:
            # Default path
            path="./data/CIFAR-10-augmented"
        train_set = datasets.CIFAR10(root=path, train=True, download=True, 
                                    transform=train_data_augmentation(normalize=transforms.Normalize(
                                        mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))))

        test_set = datasets.CIFAR10(root=path, train=False, download=True, 
                                   transform=test_data_augmentation(crop=True))

    elif dataset_name == "cifar100":
        if path is None:
            # Default path
            path="./data/CIFAR-100-augmented"
        train_set = datasets.CIFAR100(root=path, train=True, download=True, 
                                    transform=train_data_augmentation(normalize=transforms.Normalize(
                                        mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))))

        test_set = datasets.CIFAR100(root=path, train=False, download=True, 
                                   transform=test_data_augmentation(crop=True))

    elif dataset_name == "imageNet" and path is None:
        if path is None:
            # Default path
            path="./data/ImageNet-augmented"
        train_set = datasets.ImageNet(root=path, train=True, download=True, 
                                    transform=train_data_augmentation(normalize=transforms.Normalize(
                                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))))

        test_set = datasets.ImageNet(root=path, train=False, download=True, 
                                   transform=test_data_augmentation(crop=True))

    else:
        raise KeyError(f"Choose a datset name (possible dataset names : cifar10 or cifar100 or imageNet)")

    # Trainloaderand Testloader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader