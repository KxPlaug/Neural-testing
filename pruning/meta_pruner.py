import torch
import torch.nn as nn
import copy
import time
import numpy as np
from math import ceil
from collections import OrderedDict
from pruning.utils import strdict_to_dict
from tqdm import tqdm
from utils import check_device
device = check_device()

class Layer:
    def __init__(self, name, size, layer_index, res=False):
        self.name = name
        self.size = []
        for x in size:
            self.size.append(x)
        self.layer_index = layer_index
        self.is_shortcut = True if "downsample" in name else False
        if res:
            self.stage, self.seq_index, self.block_index = self._get_various_index_by_name(name)
    
    def _get_various_index_by_name(self, name):
        '''Get the indeces including stage, seq_ix, blk_ix.
            Same stage means the same feature map size.
        '''
        global lastest_stage # an awkward impel, just for now
        name = name[6:]
        if name.startswith('module.'):
            name = name[7:] # remove the prefix caused by pytorch data parallel
        if "conv1" == name: # TODO: this might not be so safe
            lastest_stage = 0
            return 0, None, None
        if "linear" in name or 'fc' in name: # Note: this can be risky. Check it fully. TODO: @mingsun-tse
            return lastest_stage + 1, None, None # fc layer should always be the last layer
        else:
            # try:
            stage  = int(name.split(".")[0][-1]) # ONLY work for standard resnets. name example: layer2.2.conv1, layer4.0.downsample.0
            seq_ix = int(name.split(".")[1])
            if 'conv' in name.split(".")[-1]:
                blk_ix = int(name[-1]) - 1
            else:
                blk_ix = -1 # shortcut layer  
            lastest_stage = stage
            return stage, seq_ix, blk_ix
            # except:
            #     print('!Parsing the layer name failed: %s. Please check.' % name)
                
class MetaPruner:
    def __init__(self, model, passer):
        self.model = model
        self.test = lambda net: passer.test(passer.test_loader, net)
        self.train_loader = passer.train_loader
        self.n_conv_within_block = passer.n_conv_within_block
        self.res = passer.res
        self.stage_pr = passer.stage_pr
        self.skip_layers = passer.skip_layers
        self.pick_pruned = passer.pick_pruned
        self.is_single_branch = passer.is_single_branch
        self.layers = OrderedDict()
        self._register_layers()
        self.kept_wg = {}
        self.pruned_wg = {}
        self.get_pr() # set up pr for each layer
        
    def _pick_pruned(self, w_abs, pr, mode="minimum"):
        if pr == 0:
            return []
        w_abs_list = w_abs.flatten()
        n_wg = len(w_abs_list)
        n_pruned = min(ceil(pr * n_wg), n_wg - 1) # do not prune all
        if mode == "rand":
            out = np.random.permutation(n_wg)[:n_pruned]
        elif mode == "minimum":
            out = w_abs_list.sort()[1][:n_pruned]
        elif mode == "maximum":
            out = w_abs_list.sort()[1][-n_pruned:]
        return out

    def _register_layers(self):
        '''
            This will maintain a data structure that can return some useful 
            information by the name of a layer.
        '''
        ix = -1 # layer index, starts from 0
        max_len_name = 0
        layer_shape = {}
        for name, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if "downsample" not in name:
                    ix += 1
                layer_shape[name] = [ix, m.weight.size()]
                max_len_name = max(max_len_name, len(name))
                
                size = m.weight.size()
                self.layers[name] = Layer(name, size, ix, self.res)
        
        max_len_ix = len("%s" % ix)
        print("Register layer index and kernel shape:")
        format_str = "[%{}d] %{}s -- kernel_shape: %s".format(max_len_ix, max_len_name)
        for name, (ix, ks) in layer_shape.items():
            print(format_str % (ix, name, ks))

    def _next_conv(self, model, name, mm):
        if hasattr(self.layers[name], 'block_index'):
            block_index = self.layers[name].block_index
            if block_index == self.n_conv_within_block - 1:
                return None
        ix_conv = 0
        ix_mm = -1
        for n, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                ix_conv += 1
                if m == mm:
                    ix_mm = ix_conv
                if ix_mm != -1 and ix_conv == ix_mm + 1:
                    return n
        return None
    
    def _prev_conv(self, model, name, mm):
        if hasattr(self.layers[name], 'block_index'):
            block_index = self.layers[name].block_index
            if block_index in [None, 0, -1]: # 1st conv, 1st conv in a block, 1x1 shortcut layer
                return None
        for n, _ in model.named_modules():
            if n in self.layers:
                ix = self.layers[n].layer_index
                if ix + 1 == self.layers[name].layer_index:
                    return n
        return None

    def _next_bn(self, model, mm):
        just_passed_mm = False
        for m in model.modules():
            if m == mm:
                just_passed_mm = True
            if just_passed_mm and isinstance(m, nn.BatchNorm2d):
                return m
        return None
   
    def _replace_module(self, model, name, new_m):
        '''
            Replace the module <name> in <model> with <new_m>
            E.g., 'module.layer1.0.conv1'
            ==> model.__getattr__('module').__getattr__("layer1").__getitem__(0).__setattr__('conv1', new_m)
        '''
        obj = model
        segs = name.split(".")
        for ix in range(len(segs)):
            s = segs[ix]
            if ix == len(segs) - 1: # the last one
                if s.isdigit():
                    obj.__setitem__(int(s), new_m)
                else:
                    obj.__setattr__(s, new_m)
                return
            if s.isdigit():
                obj = obj.__getitem__(int(s))
            else:
                obj = obj.__getattr__(s)
    
    def _get_n_filter(self, model):
        '''
            Do not consider the downsample 1x1 shortcuts.
        '''
        n_filter = []
        for name, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                if not self.layers[name].is_shortcut:
                    n_filter.append(m.weight.size(0))
        return n_filter
    
    def _get_layer_pr_vgg(self, name):
        '''Example: '[0-4:0.5, 5:0.6, 8-10:0.2]'
                    6, 7 not mentioned, default value is 0
        '''
        layer_index = self.layers[name].layer_index
        pr = self.stage_pr[layer_index]
        if str(layer_index) in self.skip_layers:
            pr = 0
        return pr

    def _get_layer_pr_resnet(self, name):
        '''
            This function will determine the prune_ratio (pr) for each specific layer
            by a set of rules.
        '''
        wg = 'weight'
        stage = self.layers[name].stage
        pr = self.stage_pr[stage]
        return pr
    
    def get_pr(self):
        if self.is_single_branch:
            get_layer_pr = self._get_layer_pr_vgg
        else:
            get_layer_pr = self._get_layer_pr_resnet

        self.pr = {}
        for name, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                self.pr[name] = get_layer_pr(name)
        print(self.pr)

    def _get_kept_wg_L1(self):
        wg = 'weight'
        for name, m in tqdm(self.model.named_modules(),total=len(list(self.model.named_modules()))):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                score = m.weight.abs().flatten()
                self.pruned_wg[name] = self._pick_pruned(score, self.pr[name], self.pick_pruned)
                self.kept_wg[name] = [i for i in range(len(score)) if i not in self.pruned_wg[name]]

    def _prune_and_build_new_model(self):
        self._get_masks()
        return
    
    def _get_masks(self):
        '''Get masks for unstructured pruning
        '''
        self.mask = {}
        for name, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                mask = torch.ones_like(m.weight.data).to(device).flatten()
                pruned = self.pruned_wg[name]
                mask[pruned] = 0
                self.mask[name] = mask.view_as(m.weight.data)
