from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import Subset

BATCH_SIZE = 128

def load_dataset():
    """Load CIFAR10 dataset."""
    train_dataset = CIFAR10(root="./data", train=True, download=True,transform=ToTensor())
    test_dataset = CIFAR10(root="./data", train=False, download=True,transform=ToTensor())
    test_dataset = Subset(test_dataset, range(0, 256))
    return train_dataset, test_dataset