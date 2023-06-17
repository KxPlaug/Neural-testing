import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from utils import check_device
import copy
import pickle
import re
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


class Taylor:
    def __init__(self, alpha=0.000001, pruning_ratio=0.65):
        self.alpha = alpha
        self.pruning_ratio = pruning_ratio

    def _get_grad(self, param):
        param = parse_param(param)
        grad = np.array(eval(
            "net." + param + ".grad.cpu().detach().numpy()", {"net": self.net}))
        return grad

    def _get_total(self):
        loss_func = torch.nn.CrossEntropyLoss(reduction='sum')
        self.all_param_names = get_all_param_names(self.net)
        total_loss = 0
        num = 0
        for x, y in tqdm(self.train_dataloader):
            x, y = x.to(device), y.to(device)
            outputs = self.net(x)
            loss = loss_func(outputs, y)
            total_loss = loss.item() + total_loss
            loss.backward()
            num += x.shape[0]
        totals = dict()
        for param in self.all_param_names:
            grad = self._get_grad(param)
            totals[param] = (grad / num) ** 2
        self.totals = totals

    def _calculate_param_remove(self):
        self.param_remove = dict()
        totals = [self.totals[param] for param in self.all_param_names]
        param_weights = [eval("net." + parse_param(param) + ".cpu().detach().numpy()", {"net": self.net})
                         for param in self.all_param_names]
        combine = [np.abs(total * weight**2)
                   for total, weight in zip(totals, param_weights)]
        # combine = np.array(combine)
        combine_flatten = np.concatenate(
            [combine_.flatten() for combine_ in combine], axis=0)
        percentile = 100 - self.thre * 100
        threshold = np.percentile(combine_flatten, percentile)
        for idx, param in enumerate(self.all_param_names):
            t = combine[idx] > threshold
            self.param_remove[param] = t

    def _prune_model(self):
        with torch.no_grad():
            for param in tqdm(self.all_param_names):
                param_ = parse_param(param)
                try:
                    exec("model." + param_ + "[~param_remove[param]] = 0", {
                        "model": self.original_net, "param_remove": self.param_remove, "param": param, "param_": param_})
                except:
                    exec("model." + param_ + "[~param_remove[param],:] = 0", {
                        "model": self.original_net, "param_remove": self.param_remove, "param": param, "param_": param_})
        self.pruned_net = self.original_net

    def __call__(self, net, train_dataloader, save_path):
        self.net = net
        self.original_net = copy.deepcopy(net)
        self.train_dataloader = train_dataloader
        self.thre = 1 - self.pruning_ratio
        self._get_total()
        self._calculate_param_remove()
        self._prune_model()
        removed = dict()
        for name in self.param_remove:
            removed[name] = torch.Tensor(self.param_remove[name]).to(device)
        with open(save_path, "wb") as f:
            pickle.dump(removed, f)
        return self.pruned_net
