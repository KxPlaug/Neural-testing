from torchvision.datasets import CIFAR100
from torchvision.transforms import ToTensor

BATCH_SIZE = 128

def load_dataset():
    """Load CIFAR100 dataset."""
    train_dataset = CIFAR100(root="./data", train=True, download=True,transform=ToTensor())
    test_dataset = CIFAR100(root="./data", train=False, download=True,transform=ToTensor())
    return train_dataset, test_dataset