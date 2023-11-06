import re
from typing import Any
import numpy as np
from pruning.sparse_utils import *
import math
from scipy.special import erfinv
import argparse
import copy
import torch.optim as optim
import pickle as pkl
from torch.autograd import Variable
from utils import check_device
import torch
device = check_device()


def parse_param(param):
    reg = re.compile("\.\d+\.")
    finded = reg.findall(param)
    if len(finded) == 0:
        return param
    else:
        for f in finded:
            f = f[1:-1]
            param = param.replace(f".{f}.", f"[{f}].")
    return parse_param(param)


def prune_model(model, param_remove, all_param_names):
    model = copy.deepcopy(model)
    with torch.no_grad():
        for param in all_param_names:
            param_ = parse_param(param)
            try:
                exec("model." + param_ + "[~param_remove[param]] = 0", {
                     "model": model, "param_remove": param_remove, "param": param, "param_": param_})
            except:
                exec("model." + param_ + "[~param_remove[param],:] = 0", {
                     "model": model, "param_remove": param_remove, "param": param, "param_": param_})
    return model


def get_all_param_names(model):
    parameters = list(model.named_parameters())
    all_param_names = list()
    i = 0
    while i < len(parameters):
        if len(parameters[i][1].shape) == 1 and "weight" in parameters[i][0]:
            i += 2
            continue
        else:
            all_param_names.append(parameters[i][0])
            i += 1
    all_param_names = all_param_names[1:-1]
    return all_param_names


def calculate_pruned(model, removed):
    leave, all_num = 0, 0
    for name, param in model.named_parameters():
        param = param.cpu().detach().numpy()
        if name in removed:
            leave += removed[name].sum()
        else:
            leave += param.size
        all_num += param.size
    return float(1 - (leave / all_num))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ASL:
    r"""
    ASL pruning algorithm
    Arguments:
        lv (float): lambda, the weight of the sparsity regularization term (Default: 10.0)
        epochs (int): number of epochs to train (Default: 1000)
        lr (float): learning rate (Default: 0.1)
        sparsity (str): 'adaptive' or 'fixed' (Default: 'adaptive')
        pthres (int): number of epochs before the pruning ratio is reached (Default: 1000)
        pruning_ratio (float): pruning ratio (Default: 0.65)

    __call__:
        Arguments:
            model (nn.Sequential): model to be pruned
            train_dataloader (torch.utils.data.DataLoader): dataloader for training
            save_path (str): path to save the pruned model mask
        Returns:
            model_pruned (nn.Sequential): pruned model

    Example::
        >>> pruner = ASL()
        >>> model = pruner(model, train_dataloader, save_path)
    """
    def __init__(self, lv=10.0, epochs=1000, lr=0.1, sparsity='adaptive', pthres=1000, pruning_ratio=0.65):
        
        self.lv = lv
        self.epochs = epochs
        self.lr = lr
        self.sparsity = sparsity
        self.pthres = pthres
        self.pruning_ratio = pruning_ratio

    def _init_model(self, model):
        self.model = model
        self.original_model = copy.deepcopy(model)
        self.all_param_names = get_all_param_names(model)
        self.all_param_masks = dict()
        for name, parameter in model.named_parameters():
            if name in self.all_param_names:
                self.all_param_masks[name] = torch.ones_like(parameter)
        iter_sparsify(self.model, erfinv(.6 * self.starget)
                      * math.sqrt(2), True, self.pthres)
        parameters, sparameters = [], []
        for name, p in self.model.named_parameters():
            if ".r" in name:
                sparameters += [p]
            else:
                parameters += [p]
        # ensuring convergence: slower lr on sparsity controlling pruning parameter (w/o weight decay)!
        self.optimizer = optim.SGD([{"params": parameters}, {"params": sparameters, "lr": self.lr/100.0, "weight_decay": 0}],
                                   lr=self.lr, momentum=.9, weight_decay=5e-4)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, self.epochs//2, T_mult=1)
        self.model_size = count_parameters(self.model)
        print('number of parameters: ' + str(self.model_size) + ' !!!!\n')
        sparsity(self.model, False)

    def train(self, epoch):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_dataloader):
            data, target = data.to(device), target.to(device)
            data, target = Variable(data), Variable(target)
            self.optimizer.zero_grad()
            _,loss = self.model.get_loss(data, target)
            eloss = 0
            if self.sparsity == 'fixed':
                eloss = self.lv * \
                    ((adaptive_loss(self.model, reduce=False) - 1 + self.starget)**2).mean()
            elif self.sparsity == 'adaptive':
                eloss = self.lv * (adaptive_loss(self.model, True)
                                   [0] - 1 + self.starget)**2
            loss = loss + eloss
            loss.backward()
            self.optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx *
                    len(data), len(self.train_dataloader.dataset),
                    100. * batch_idx / len(self.train_dataloader), loss.item()))

    def __call__(self, model, train_dataloader, save_path=None):
        self.train_dataloader = train_dataloader
        self.starget = self.pruning_ratio
        self._init_model(model)
        for epoch in range(1, self.epochs + 1):
            self.train(epoch)
            self.scheduler.step()
            sp, nz = sparsity(self.model, True)
            print("overall sparsity : " + str(sp) + " with " +
                  str(self.model_size-int(nz)) + " nonzero elements")
            model_clone = copy.deepcopy(self.model)
            iter_desparsify(model_clone)
            for name, param in model_clone.named_parameters():
                if name in self.all_param_names:
                    self.all_param_masks[name] = (param != 0).float()
            pruned_ratio = calculate_pruned(model_clone, self.all_param_masks)
            if ((self.starget - pruned_ratio) < 0.01 and (self.starget - pruned_ratio) > 0) or pruned_ratio > self.starget:
                print("pruning ratio reached")
                mask = copy.deepcopy(self.all_param_masks)
                for key in mask.keys():
                    mask[key] = mask[key].bool()
                if save_path is not None:
                    pkl.dump(mask, open(save_path, "wb"))
                model_pruned = prune_model(
                    self.original_model, mask, self.all_param_names)
                break
        return model_pruned
