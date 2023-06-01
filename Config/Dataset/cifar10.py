from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

def load_dataset():
    """Load CIFAR10 dataset."""
    dataset = CIFAR10(root="./data", train=False, download=True,transform=ToTensor())
    return dataset