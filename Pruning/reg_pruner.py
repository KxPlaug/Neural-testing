import torch
import torch.nn as nn
import torch.optim as optim
import os, copy, time, pickle, numpy as np, math
from Pruning.meta_pruner import MetaPruner
from Pruning.utils import plot_weights_heatmap
import matplotlib.pyplot as plt
pjoin = os.path.join

class Pruner(MetaPruner):
    def __init__(self, model, passer,reg_granularity_recover=1e-4,reg_granularity_prune=1e-4,reg_granularity_pick=1e-5,
                 reg_upper_limit_pick=1e-2,reg_upper_limit=1.0,lr_prune=0.001,momentum=0.9,weight_decay=1e-4,
                 test_interval=2000,update_reg_interval=5,stabilize_reg_interval=40000):
        super(Pruner, self).__init__(model, passer)

        # Reg related variables
        self.reg = {}
        self.delta_reg = {}
        self.hist_mag_ratio = {}
        self.n_update_reg = {}
        self.iter_update_reg_finished = {}
        self.iter_finish_pick = {}
        self.iter_stabilize_reg = math.inf
        self.original_w_mag = {}
        self.original_kept_w_mag = {}
        self.ranking = {}
        self.pruned_wg_L1 = {}
        self.all_layer_finish_pick = False
        self.w_abs = {}
        self.original_model = copy.deepcopy(self.model)
        self.reg_granularity_recover = reg_granularity_recover
        self.reg_granularity_prune = reg_granularity_prune
        self.reg_granularity_pick = reg_granularity_pick
        self.reg_upper_limit_pick = reg_upper_limit_pick
        self.reg_upper_limit = reg_upper_limit
        self.lr_prune = lr_prune
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.test_interval = test_interval
        self.update_reg_interval = update_reg_interval
        self.stabilize_reg_interval = stabilize_reg_interval
        
        self._get_kept_wg_L1()
        for k, v in self.pruned_wg.items():
            self.pruned_wg_L1[k] = v
        self.kept_wg = {}
        self.pruned_wg = {}

        self.prune_state = "update_reg"
        for name, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):

                self.reg[name] = torch.zeros_like(m.weight.data).flatten().cuda()
                
                w_abs = self._get_score(m)
                n_wg = len(w_abs)
                self.ranking[name] = []
                for _ in range(n_wg):
                    self.ranking[name].append([])
                self.original_w_mag[name] = m.weight.abs().mean().item()
                kept_wg_L1 = [i for i in range(n_wg) if i not in self.pruned_wg_L1[name]]
                self.original_kept_w_mag[name] = w_abs[kept_wg_L1].mean().item()

    def _pick_pruned_wg(self, w, pr):
        if pr == 0:
            return []
        elif pr > 0:
            w = w.flatten()
            n_pruned = min(math.ceil(pr * w.size(0)), w.size(0) - 1) # do not prune all
            return w.sort()[1][:n_pruned]
        elif pr == -1: # automatically decide lr by each layer itself
            tmp = w.flatten().sort()[0]
            n_not_consider = int(len(tmp) * 0.02)
            w = tmp[n_not_consider:-n_not_consider]

            sorted_w, sorted_index = w.flatten().sort()
            max_gap = 0
            max_index = 0
            for i in range(len(sorted_w) - 1):
                # gap = sorted_w[i+1:].mean() - sorted_w[:i+1].mean()
                gap = sorted_w[i+1] - sorted_w[i]
                if gap > max_gap:
                    max_gap = gap
                    max_index = i
            max_index += n_not_consider
            return sorted_index[:max_index + 1]
        else:
            exit(1)
    
    def _update_mag_ratio(self, m, name, w_abs, pruned=None):
        if type(pruned) == type(None):
            pruned = self.pruned_wg[name]
        kept = [i for i in range(len(w_abs)) if i not in pruned]
        ave_mag_pruned = w_abs[pruned].mean()
        ave_mag_kept = w_abs[kept].mean()
        if len(pruned):
            mag_ratio = ave_mag_kept / ave_mag_pruned 
            if name in self.hist_mag_ratio:
                self.hist_mag_ratio[name] = self.hist_mag_ratio[name]* 0.9 + mag_ratio * 0.1
            else:
                self.hist_mag_ratio[name] = mag_ratio
        else:
            mag_ratio = math.inf
            self.hist_mag_ratio[name] = math.inf
        
        # print
        mag_ratio_now_before = ave_mag_kept / self.original_kept_w_mag[name]

        return mag_ratio_now_before

    def _get_score(self, m):
        w_abs = m.weight.abs().flatten()
        return w_abs

        
    def _greg_2(self, m, name):
        layer_index = self.layers[name].layer_index
        w_abs = self.w_abs[name]
        n_wg = len(w_abs)
        pr = self.pr[name]
        if pr == 0:
            self.kept_wg[name] = range(n_wg)
            self.pruned_wg[name] = []
            self.iter_finish_pick[name] = self.total_iter
            return True
        
        if name in self.iter_finish_pick:
            recover_reg = self.reg_granularity_recover
            # for pruned weights, push them more
            self.reg[name][self.pruned_wg[name]] += self.reg_granularity_prune
            self.reg[name][self.kept_wg[name]] = recover_reg
        else:
            self.reg[name] += self.reg_granularity_pick

        
        finish_pick_cond = self.reg[name].max() >= self.reg_upper_limit_pick
        if name not in self.iter_finish_pick and finish_pick_cond:
            self.iter_finish_pick[name] = self.total_iter
            pruned_wg = self._pick_pruned_wg(w_abs, pr)
            kept_wg = [i for i in range(n_wg) if i not in pruned_wg]
            self.kept_wg[name] = kept_wg
            self.pruned_wg[name] = pruned_wg
            self.all_layer_finish_pick = True
            for k in self.reg:
                if self.pr[k] > 0 and (k not in self.iter_finish_pick):
                    self.all_layer_finish_pick = False
                    break
        

        cond0 = name in self.iter_finish_pick # finsihed picking
        cond1 = self.reg[name].max() > self.reg_upper_limit
        finish_update_reg = cond0 and cond1
        return finish_update_reg

    def _update_reg(self):
        for name, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                cnt_m = self.layers[name].layer_index
                pr = self.pr[name]
                
                if name in self.iter_update_reg_finished.keys():
                    continue

                # get the importance score (L1-norm in this case)
                self.w_abs[name] = self._get_score(m)
                
                # update reg functions, two things: 
                # (1) update reg of this layer (2) determine if it is time to stop update reg
                finish_update_reg = self._greg_2(m, name)

                # check prune state
                if finish_update_reg:
                    # after 'update_reg' stage, keep the reg to stabilize weight magnitude
                    self.iter_update_reg_finished[name] = self.total_iter
                    print("==> [%d] Just finished 'update_reg'. Iter = %d" % (cnt_m, self.total_iter))

                    # check if all layers finish 'update_reg'
                    self.prune_state = "stabilize_reg"
                    for n, mm in self.model.named_modules():
                        if isinstance(mm, nn.Conv2d) or isinstance(mm, nn.Linear):
                            if n not in self.iter_update_reg_finished:
                                self.prune_state = "update_reg"
                                break
                    if self.prune_state == "stabilize_reg":
                        self.iter_stabilize_reg = self.total_iter
                    

    def _apply_reg(self):
        for name, m in self.model.named_modules():
            if name in self.reg:
                reg = self.reg[name] # [N, C]
                reg = reg.view_as(m.weight.data) # [N, C, H, W]
                l2_grad = reg * m.weight
                m.weight.grad += l2_grad



    def prune(self):
        self.model = self.model.train()
        self.optimizer = optim.SGD(self.model.parameters(), 
                                lr=self.lr_prune,
                                momentum=self.momentum,
                                weight_decay=self.weight_decay)
        
        # resume model, optimzer, prune_status
        self.total_iter = -1

        acc1 = acc5 = 0
        while True:
            for _, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                self.total_iter += 1
                total_iter = self.total_iter
                
                # test
                if total_iter % self.test_interval == 0:
                    acc1, acc5, *_ = self.test(self.model)
                    print("Acc1 = %.4f Acc5 = %.4f Iter = %d (before update) [prune_state = %s]" % 
                        (acc1, acc5, total_iter, self.prune_state))
                
                    
                # forward
                self.model.train()
                
                if self.prune_state == "update_reg" and total_iter % self.update_reg_interval == 0:
                    self._update_reg()
                    
                # normal training forward
                _,loss = self.model.get_loss(inputs,targets)
                self.optimizer.zero_grad()
                loss.backward()
                
                # after backward but before update, apply reg to the grad
                self._apply_reg()
                self.optimizer.step()

                # change prune state
                if self.prune_state == "stabilize_reg" and total_iter - self.iter_stabilize_reg == self.stabilize_reg_interval:
                    self._prune_and_build_new_model() 
                    return copy.deepcopy(self.model)