from torchvision.datasets import CIFAR100
from torchvision.transforms import ToTensor

def load_dataset():
    """Load CIFAR100 dataset."""
    dataset = CIFAR100(root="./data", train=False, download=True, transform=ToTensor())
    return dataset