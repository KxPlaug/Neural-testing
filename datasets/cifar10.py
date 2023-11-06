from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

BATCH_SIZE = 128

"""Note: image must in range [0, 1]"""
def load_dataset():
    """Load CIFAR10 dataset."""
    train_dataset = CIFAR10(root="./data", train=True, download=True,transform=ToTensor())
    test_dataset = CIFAR10(root="./data", train=False, download=True,transform=ToTensor())
    return train_dataset, test_dataset