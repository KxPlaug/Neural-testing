import torch
import random
from torchvision import transforms

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
    
class RandomCrop:
    def __init__(self, padding=4):
        self.padding = padding

    def __call__(self, batch):
        x, y = zip(*batch)
        x = torch.stack(x)
        y = torch.Tensor(y).long()
        x = transforms.RandomCrop((x.shape[-2], x.shape[-1]), padding=self.padding,pad_if_needed=True)(x)
        # x = torch.nn.functional.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='constant', value=0)
        # x = torch.nn.functional.interpolate(x, size=(x.shape[-2], x.shape[-1]), mode='bilinear', align_corners=False)
        return x, y
    
class DataMissingOD:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, batch):
        batch = list(zip(*batch))
        batch[0] = list(batch[0])
        for i in range(len(batch[0])):
            batch[0][i] = batch[0][i] * (torch.rand(batch[0][i].shape) > self.p).float()
        batch[0] = tuple(batch[0])
        batch = tuple(batch)
        return batch
    
class DataShuffleOD:
    
    def __call__(self, batch):
        batch = list(zip(*batch))
        batch[0] = list(batch[0])
        batch[1] = list(batch[1])
        state = random.getstate()
        random.shuffle(batch[0])
        random.setstate(state)
        random.shuffle(batch[1])
        batch[0] = tuple(batch[0])
        batch[1] = tuple(batch[1])
        batch = tuple(batch)
        return batch
    
class NoisePerturbOD:
    def __init__(self, std=0.1):
        self.std = std

    def __call__(self, batch):
        batch = list(zip(*batch))
        batch[0] = list(batch[0])
        for i in range(len(batch[0])):
            batch[0][i] = batch[0][i] + torch.randn(batch[0][i].shape) * self.std
        batch[0] = tuple(batch[0])
        batch = tuple(batch)
        return batch
    
class ContrastRatioOD:
    def __init__(self, min_ratio=0.8, max_ratio=1.2):
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def __call__(self, batch):
        batch = list(zip(*batch))
        batch[0] = list(batch[0])
        ratio = random.uniform(self.min_ratio, self.max_ratio)
        for i in range(len(batch[0])):
            batch[0][i] = torch.clamp(batch[0][i] * ratio, 0, 1)
        batch[0] = tuple(batch[0])
        batch = tuple(batch)
        return batch
    
class BrightnessOD:
    def __init__(self, min_shift=-0.1, max_shift=0.1):
        self.min_shift = min_shift
        self.max_shift = max_shift

    def __call__(self, batch):
        batch = list(zip(*batch))
        batch[0] = list(batch[0])
        shift = random.uniform(self.min_shift, self.max_shift)
        for i in range(len(batch[0])):
            batch[0][i] = torch.clamp(batch[0][i] + shift, 0, 1)
        batch[0] = tuple(batch[0])
        batch = tuple(batch)
        return batch
    
class RandomCropOD:
    def __init__(self, padding=4):
        self.padding = padding
        
    def __call__(self, batch):
        batch = list(zip(*batch))
        batch[0] = list(batch[0])
        for i in range(len(batch[0])):
            batch[0][i] = transforms.RandomCrop((batch[0][i].shape[-2], batch[0][i].shape[-1]), padding=self.padding,pad_if_needed=True)(batch[0][i])
        batch[0] = tuple(batch[0])
        batch = tuple(batch)
        return batch