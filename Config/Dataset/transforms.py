import random
import torch

from torchvision.transforms import functional as F
from typing import Dict, List, Optional, Tuple, Union

import PIL
import torch
import torchvision
from torch import nn, Tensor
from torchvision import ops
from torchvision.transforms import functional as F, InterpolationMode, transforms as T


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def forward(
            self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = image.clone()
        if torch.rand(1) < self.p:
            boxes = target['boxes']
            for box in boxes:
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                image[:, :, y1:y2, x1:x2] = torch.flip(
                    image[:, :, y1:y2, x1:x2], [3])
            return image, target, True
        else:
            return image, target, False


class RandomVerticalFlip(T.RandomVerticalFlip):
    def forward(
            self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = image.clone()
        if torch.rand(1) < self.p:
            boxes = target['boxes']
            for box in boxes:
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                image[:, :, y1:y2, x1:x2] = torch.flip(
                    image[:, :, y1:y2, x1:x2], [2])
            return image, target, True
        else:
            return image, target, False


class RandomAdjustContrast(nn.Module):
    def __init__(self, p, factor) -> None:
        super().__init__()
        self.factor = factor
        self.p = p

    def forward(
            self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = image.clone()
        if torch.rand(1) < self.p:
            image = F.adjust_contrast(image, self.factor)
            return image, target, True
        else:
            return image, target, False


class RandomAdjustBrightness(nn.Module):
    def __init__(self, p, delta) -> None:
        super().__init__()
        self.delta = delta
        self.p = p

    def forward(
            self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = image.clone()
        if torch.rand(1) < self.p:
            image = F.adjust_brightness(image, self.delta)
            return image, target, True
        else:
            return image, target, False


class RandomAdjustColor(nn.Module):
    def __init__(self, p, factor) -> None:
        super().__init__()
        self.factor = factor
        self.p = p

    def forward(
            self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = image.clone()
        if torch.rand(1) < self.p:
            image = F.adjust_hue(image, self.factor)
            return image, target, True
        else:
            return image, target, False
