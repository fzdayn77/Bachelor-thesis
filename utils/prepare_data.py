from data_augmentation import train_data_augmentation, test_data_augmentation, GaussianBlur
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


def get_data(dataset_name, path, batch_size=128):
    """
    Imports trainset and testset of the chosen Dataset and returns the corresponding trainloader and testloader.
    """

    # Trainset and Testset
    if dataset_name == "cifar10":
        path="./data/CIFAR-10-augmented"
        train_set = datasets.CIFAR10(root=path, train=True, download=True, 
                                    transform=train_data_augmentation(normalize=transforms.Normalize(
                                        mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))))

        test_set = datasets.CIFAR10(root=path, train=False, download=True, 
                                   transform=test_data_augmentation(crop=True))

    elif dataset_name == "cifar100":
        path="./data/CIFAR-100-augmented"
        train_set = datasets.CIFAR100(root=path, train=True, download=True, 
                                    transform=train_data_augmentation(normalize=transforms.Normalize(
                                        mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))))

        test_set = datasets.CIFAR100(root=path, train=False, download=True, 
                                   transform=test_data_augmentation(crop=True))

    elif dataset_name == "imageNet":
        path="./data/ImageNet-augmented"
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