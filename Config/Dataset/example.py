import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import Resize
import os


def load_dataset():
    """Load Custom dataset."""
    ...
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    x_batch = torch.load(os.path.join(
        parent_dir, 'imagenet_1k/imagenet_x_batch.pt'), map_location=torch.device('cpu'))[:20]
    y_batch = torch.load(os.path.join(
        parent_dir, 'imagenet_1k/imagenet_y_batch.pt'), map_location=torch.device('cpu'))[:20]
    x_batch = Resize((224, 224))(x_batch)
    dataset = TensorDataset(x_batch, y_batch)
    return dataset


if __name__ == "__main__":
    load_dataset()
