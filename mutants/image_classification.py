import torch
import random
from torchvision import transforms

class LabelError:
    r"""
    LabelError
    Arguments:
        p (float): Probability of label error. (Default: 0.5)
        num_classes (int): Number of classes. (Default: 10)
        
    Example::
        >>> collect_fn = LabelError(p=0.5, num_classes=10)
        >>> dataloader = DataLoader(dataset, batch_size=32, collate_fn=collect_fn)
    """
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
    r"""
    DataMissing
    Arguments:
        p (float): Probability of data missing. (Default: 0.5)
        
    Example::
        >>> collect_fn = DataMissing(p=0.5)
        >>> dataloader = DataLoader(dataset, batch_size=32, collate_fn=collect_fn)
    """
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
    r"""
    DataShuffle
    Arguments:
        p (float): Probability of data shuffle. (Default: 0.5)
        
    Example::
        >>> collect_fn = DataShuffle(p=0.5)
        >>> dataloader = DataLoader(dataset, batch_size=32, collate_fn=collect_fn)
    """
    def __call__(self, batch):
        x, y = zip(*batch)
        x = torch.stack(x)
        y = torch.Tensor(y).long()
        random_index = torch.randperm(len(y))
        return x[random_index], y[random_index]
    
class NoisePerturb:
    r"""
    NoisePerturb
    Arguments:
        std (float): Standard deviation of noise. (Default: 0.1)
        
    Example::
        >>> collect_fn = NoisePerturb(std=0.1)
        >>> dataloader = DataLoader(dataset, batch_size=32, collate_fn=collect_fn)
    """
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
    r"""
    ContrastRatio
    Arguments:
        min_ratio (float): Minimum contrast ratio. (Default: 0.8)
        max_ratio (float): Maximum contrast ratio. (Default: 1.2)
        
    Example::
        >>> collect_fn = ContrastRatio(min_ratio=0.8, max_ratio=1.2)
        >>> dataloader = DataLoader(dataset, batch_size=32, collate_fn=collect_fn)
    """
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
    r"""
    Brightness
    Arguments:
        min_shift (float): Minimum brightness shift. (Default: -0.1)
        max_shift (float): Maximum brightness shift. (Default: 0.1)
        
    Example::
        >>> collect_fn = Brightness(min_shift=-0.1, max_shift=0.1)
        >>> dataloader = DataLoader(dataset, batch_size=32, collate_fn=collect_fn)
    """
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
    r"""
    RandomCrop
    Arguments:
        padding (int): Padding size. (Default: 4)
        
    Example::
        >>> collect_fn = RandomCrop(padding=4)
        >>> dataloader = DataLoader(dataset, batch_size=32, collate_fn=collect_fn)
    """
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
    
