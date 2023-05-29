from data_augmentation import simCLR_training_data_augmentation, simCLR_eval_data_augmentation, GaussianBlur
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


def get_datasets(dataset_name, path, batch_size=128):
    """
    Imports traiset and testset of the chosen Dataset and returns the corresponding trainloader and testloader
    """

    # Trainset and Testset
    if dataset_name == "cifar10":
        path="/data/CIFAR-10-augmented"
        trainset = datasets.CIFAR10(root=path, train=True, download=True, 
                                    transform=simCLR_training_data_augmentation(normalize=transforms.Normalize(
                                        mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))))

        testset = datasets.CIFAR10(root=path, train=False, download=True, 
                                   transform=simCLR_eval_data_augmentation(crop=True))

    elif dataset_name == "cifar100":
        path="/data/CIFAR-100-augmented"
        trainset = datasets.CIFAR100(root=path, train=True, download=True, 
                                    transform=simCLR_training_data_augmentation(normalize=transforms.Normalize(
                                        mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))))

        testset = datasets.CIFAR100(root=path, train=False, download=True, 
                                   transform=simCLR_eval_data_augmentation(crop=True))

    elif dataset_name == "imageNet":
        path="/data/ImageNet-augmented"
        trainset = datasets.ImageNet(root=path, train=True, download=True, 
                                    transform=simCLR_training_data_augmentation(normalize=transforms.Normalize(
                                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))))

        testset = datasets.ImageNet(root=path, train=False, download=True, 
                                   transform=simCLR_eval_data_augmentation(crop=True))

    else:
        return "Choose a datset name ( cifar10 or cifar100 or imagNet )"

    # Trainloaderand Testloader
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader