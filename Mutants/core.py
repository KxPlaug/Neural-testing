import torch
import random

class LabelError:
    def __init__(self, p=0.5, num_classes=10):
        self.p = p
        self.num_classes = num_classes

    def __call__(self, batch):
        x, y = zip(*batch)
        x = torch.stack(x)
        y = torch.Tensor(y).long()
        for i in range(len(y)):
            if random.random() < self.p:
                random_label = random.randint(0, self.num_classes - 1)
                while random_label == y[i]:
                    random_label = random.randint(0, self.num_classes - 1)
                y[i] = random_label
        return x, y
    
class DataMissing:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, batch):
        x, y = zip(*batch)
        x = torch.stack(x)
        y = torch.Tensor(y).long()
        for i in range(len(y)):
            x[i] = x[i] * (torch.rand(x[i].shape) > self.p).float()
        return x, y
    
class DataShuffle:
    def __call__(self, batch):
        x, y = zip(*batch)
        x = torch.stack(x)
        y = torch.Tensor(y).long()
        random_index = torch.randperm(len(y))
        return x[random_index], y[random_index]
    
class NoisePerturb:
    def __init__(self, std=0.1):
        self.std = std

    def __call__(self, batch):
        x, y = zip(*batch)
        x = torch.stack(x)
        y = torch.Tensor(y).long()
        for i in range(len(y)):
            x[i] = x[i] + torch.randn(x[i].shape) * self.std
        x = torch.clamp(x, 0, 1)
        return x, y


class ContrastRatio:
    def __init__(self, min_ratio=0.8, max_ratio=1.2):
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def __call__(self, batch):
        x, y = zip(*batch)
        x = torch.stack(x)
        y = torch.Tensor(y).long()
        ratio = random.uniform(self.min_ratio, self.max_ratio)
        x = torch.clamp(x * ratio, 0, 1)
        return x, y

class Brightness:
    def __init__(self, min_shift=-0.1, max_shift=0.1):
        self.min_shift = min_shift
        self.max_shift = max_shift

    def __call__(self, batch):
        x, y = zip(*batch)
        x = torch.stack(x)
        y = torch.Tensor(y).long()
        shift = random.uniform(self.min_shift, self.max_shift)
        x = torch.clamp(x + shift, 0, 1)
        return x, y