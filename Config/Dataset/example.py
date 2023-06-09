import torch
from torch.utils.data import TensorDataset
from torchvision.transforms import Resize
import os
from utils import check_device
device = check_device()


def load_dataset():
    """Load Custom dataset."""
    ...
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    x_train_batch = torch.load(os.path.join(
        parent_dir, 'imagenet_1k/train/imagenet_x_batch.pt'), map_location=device)
    y_train_batch = torch.load(os.path.join(
        parent_dir, 'imagenet_1k/train/imagenet_y_batch.pt'), map_location=device)
    x_test_batch = torch.load(os.path.join(
        parent_dir, 'imagenet_1k/test/imagenet_x_batch.pt'), map_location=device)
    y_test_batch = torch.load(os.path.join(
        parent_dir, 'imagenet_1k/test/imagenet_y_batch.pt'), map_location=device)
    x_train_batch = Resize((224, 224))(x_train_batch)
    x_test_batch = Resize((224, 224))(x_test_batch)
    train_dataset = TensorDataset(x_train_batch, y_train_batch)
    test_dataset = TensorDataset(x_test_batch, y_test_batch)
    return train_dataset, test_dataset


if __name__ == "__main__":
    load_dataset()
