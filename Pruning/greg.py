'''
    This code is based on the official PyTorch ImageNet training example 'main.py'. Commit ID: 69d2798, 04/23/2020.
    URL: https://github.com/pytorch/examples/tree/master/imagenet
    Major modified parts will be indicated by '@mst' mark.
    Questions to @mingsun-tse (wang.huan@northeastern.edu).
'''

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import pickle
from importlib import import_module
from Pruning.utils import get_n_params, get_n_flops, get_n_params_, get_n_flops_, parse_prune_ratio_vgg, strlist_to_list
from Pruning.utils import add_noise_to_model, compute_jacobian
from Pruning.reg_pruner import Pruner

from utils import check_device
device = check_device()


def apply_mask_forward(model, mask):
    for name, m in model.named_modules():
        if name in mask:
            m.weight.data.mul_(mask[name])
            

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def validate(val_loader, model):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    train_state = model.training

    # switch to evaluate mode
    model.eval()

    # @mst: add noise to model
    model_ensemble = []
    model_ensemble.append(model)

    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images, target = images.to(device), target.to(device)
            output,loss = model.get_loss(images, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
    if train_state:
        model.train()
    return top1.avg.item(), top5.avg.item(), losses.avg

class GReg:
    def __init__(self, img_size=224, num_classes=1000, num_chennels=3, momentum=0.9, weight_decay=1e-4,
                 reg_granularity_recover=1e-4,reg_granularity_prune=1e-4,reg_granularity_pick=1e-5,
                 reg_upper_limit_pick=1e-2,reg_upper_limit=1.0,lr_prune=0.001,
                 test_interval=2000,update_reg_interval=5,stabilize_reg_interval=40000):
        self.img_size = img_size
        self.num_classes = num_classes
        self.num_channels = num_chennels
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.reg_granularity_recover = reg_granularity_recover
        self.reg_granularity_prune = reg_granularity_prune
        self.reg_granularity_pick = reg_granularity_pick
        self.reg_upper_limit_pick = reg_upper_limit_pick
        self.reg_upper_limit = reg_upper_limit
        self.lr_prune = lr_prune
        self.test_interval = test_interval
        self.update_reg_interval = update_reg_interval
        self.stabilize_reg_interval = stabilize_reg_interval

    def __call__(self, model, train_loader_prune, test_loader,save_path,n_conv_within_block=3,res=True,
                    stage_pr="[0,0.675,0.675,0.675,0.675,0.675]",skip_layers="",pick_pruned="min"):
        pruner = None
        class passer:
            pass
        passer.test = validate
        passer.train_loader = train_loader_prune
        passer.test_loader = test_loader
        passer.pruner = pruner
        passer.n_conv_within_block = n_conv_within_block
        passer.res = res
        passer.is_single_branch = model.is_single_branch
        if model.is_single_branch:
            passer.stage_pr = parse_prune_ratio_vgg(stage_pr, num_layers=model.num_layers) # example: [0-4:0.5, 5:0.6, 8-10:0.2]
            passer.skip_layers = strlist_to_list(skip_layers, str) # example: [0, 2, 6]
        else: # e.g., resnet
            passer.stage_pr = strlist_to_list(stage_pr, float) # example: [0, 0.4, 0.5, 0]
            passer.skip_layers = strlist_to_list(skip_layers, str) 
        passer.pick_pruned = pick_pruned
        pruner = Pruner(model,passer,reg_granularity_recover=self.reg_granularity_recover,reg_granularity_prune=self.reg_granularity_prune,reg_granularity_pick=self.reg_granularity_pick,
                 reg_upper_limit_pick=self.reg_upper_limit_pick,reg_upper_limit=self.reg_upper_limit,lr_prune=self.lr_prune,momentum=self.momentum,weight_decay=self.weight_decay,
                 test_interval=self.test_interval,update_reg_interval=self.update_reg_interval,stabilize_reg_interval=self.stabilize_reg_interval)
        pruner.prune()
        mask = pruner.mask
        pickle.dump(mask,open(save_path,"wb"))
        pruned_model = pruner.original_model
        pruned_model = apply_mask_forward(pruned_model,mask)
        return pruned_model

