import os
from torch.autograd import Variable as V
from torchvision import transforms as T
from tqdm import tqdm
import numpy as np
from PIL import Image
from dct import *
import numpy as np
import scipy.stats as st
import torch.nn.functional as F
import torch
import torch.nn as nn


class Normalize(nn.Module):

    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, input):
        size = input.size()
        x = input.clone()
        for i in range(size[1]):
            x[:, i] = (x[:, i] - self.mean[i]) / self.std[i]
        return x


"""Translation-Invariant https://arxiv.org/abs/1904.02884"""


def gkern(kernlen=15, nsig=3):
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    kernel = kernel.astype(np.float32)
    gaussian_kernel = np.stack([kernel, kernel, kernel])  # 5*5*3
    gaussian_kernel = np.expand_dims(gaussian_kernel, 1)  # 1*5*5*3
    gaussian_kernel = torch.from_numpy(
        gaussian_kernel).cuda()  # tensor and cuda
    return gaussian_kernel


"""Input diversity: https://arxiv.org/abs/1803.06978"""


def DI(x, resize_rate=1.15, diversity_prob=0.5):
    assert resize_rate >= 1.0
    assert diversity_prob >= 0.0 and diversity_prob <= 1.0
    img_size = x.shape[-1]
    img_resize = int(img_size * resize_rate)
    rnd = torch.randint(low=img_size, high=img_resize,
                        size=(1,), dtype=torch.int32)
    rescaled = F.interpolate(
        x, size=[rnd, rnd], mode='bilinear', align_corners=False)
    h_rem = img_resize - rnd
    w_rem = img_resize - rnd
    pad_top = torch.randint(low=0, high=h_rem.item(),
                            size=(1,), dtype=torch.int32)
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(low=0, high=w_rem.item(),
                             size=(1,), dtype=torch.int32)
    pad_right = w_rem - pad_left
    padded = F.pad(rescaled, [pad_left.item(), pad_right.item(
    ), pad_top.item(), pad_bottom.item()], value=0)
    ret = padded if torch.rand(1) < diversity_prob else x
    return ret


def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + \
        (result > t_max).float() * t_max
    return result


T_kernel = gkern(7, 3)


class SSA:
    def __init__(self, model, min=0, max=1, num_iter=10, eps=16.0/255.0, momentum=1.0, N=20, rho=0.5, sigma=16.0/255.0):
        self.model = model
        self.min = min
        self.max = max
        self.num_iter = num_iter
        self.eps = eps
        self.momentum = momentum
        self.N = N
        self.rho = rho
        self.sigma = sigma
        self.alpha = self.eps / self.num_iter

    def __call__(self, images, labels):
        images_min = clip_by_tensor(images - self.eps, self.min, self.max)
        images_max = clip_by_tensor(images + self.eps, self.min, self.max)
        x = images.clone()
        for _ in range(self.num_iter):
            noise = 0
            for _ in range(self.N):
                gauss = torch.randn_like(x) * self.sigma
                x_dct = dct_2d(x + gauss)
                mask = torch.rand_like(x) * 2 * self.rho + 1 - self.rho
                x_idct = idct_2d(x_dct * mask)
                x_idct = V(x_idct, requires_grad=True)

                # DI-FGSM https://arxiv.org/abs/1803.06978
                # output_v3 = model(DI(x_idct))
                loss = self.model.get_loss(x_idct, labels)
                loss.backward()
                noise += x_idct.grad.data
            noise = noise / self.N
            x = x + self.alpha * torch.sign(noise)
            x = clip_by_tensor(x, images_min, images_max)

            # TI-FGSM https://arxiv.org/pdf/1904.02884.pdf
            # noise = F.conv2d(noise, T_kernel, bias=None, stride=1, padding=(3, 3), groups=3)

            # MI-FGSM https://arxiv.org/pdf/1710.06081.pdf
            # noise = noise / torch.abs(noise).mean([1, 2, 3], keepdim=True)
            # noise = momentum * grad + noise
            # grad = noise
        return x.detach()
