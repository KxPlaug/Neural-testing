import torch
from torch.utils.data import TensorDataset
from torchvision.transforms import Resize
import os
from utils import check_device

BATCH_SIZE = 32

def load_dataset():
    """Load Custom dataset."""
    ...
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    x_train_batch = torch.load(os.path.join(
        parent_dir, 'imagenet_1k/train/img_batch.pt'), map_location="cpu")
    y_train_batch = torch.load(os.path.join(
        parent_dir, 'imagenet_1k/train/label_batch.pt'), map_location="cpu")
    x_test_batch = torch.load(os.path.join(
        parent_dir, 'imagenet_1k/test/img_batch.pt'), map_location="cpu")
    y_test_batch = torch.load(os.path.join(
        parent_dir, 'imagenet_1k/test/label_batch.pt'), map_location="cpu")
    train_dataset = TensorDataset(x_train_batch, y_train_batch)
    test_dataset = TensorDataset(x_test_batch, y_test_batch)
    return train_dataset, test_dataset


if __name__ == "__main__":
    load_dataset()
