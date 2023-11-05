from .core import FastIG, GuidedIG, pgd_step, BIG, FGSM, SaliencyGradient, SmoothGradient, DL, IntegratedGradient, SaliencyMap,FastIGOD,FastIGTC
import torch
import numpy as np
import random
from tqdm import tqdm
from utils import check_device
import copy
import os
from explanation.evaluation import CausalMetric
from glob import glob
device = check_device()


def fast_ig(model, data, target, *args):
    r"""
    fast_ig explanation
    Arguments:
        model (nn.Sequential): Black-box model being explained.
        data (torch.Tensor): Input tensor.
        target (torch.Tensor): Target tensor.
        
    Examples::
        >>> explanation = fast_ig(model, images, labels)
        >>> attribution_map = explanation(model, images, labels)
    """
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    method = FastIG(model)
    result = method(data, target).squeeze()
    return np.expand_dims(result, axis=0)


def guided_ig(model, data, target, steps=15):
    r"""
    guided_ig explanation
    Arguments:
        model (nn.Sequential): Black-box model being explained.
        data (torch.Tensor): Input tensor.
        target (torch.Tensor): Target tensor.
        steps (int): Number of steps. (Default: 15)
        
    Examples::
        >>> explanation = guided_ig(model, images, labels, steps=15)
        >>> attribution_map = explanation(model, images, labels)
    """
    model = copy.deepcopy(model)[:2]
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    class_idx_str = 'class_idx_str'

    def call_model_function(images, call_model_args=None, expected_keys=None):
        target_class_idx = call_model_args[class_idx_str]
        images = torch.from_numpy(images).float().to(device)
        images = images.requires_grad_(True)
        output = model(images)
        m = torch.nn.Softmax(dim=1)
        output = m(output)
        outputs = output[:, target_class_idx]
        grads = torch.autograd.grad(
            outputs, images, grad_outputs=torch.ones_like(outputs))[0]
        gradients = grads.cpu().detach().numpy()
        return {'INPUT_OUTPUT_GRADIENTS': gradients}

    im = data.squeeze().cpu().detach().numpy()
    call_model_args = {class_idx_str: target}
    baseline = np.zeros(im.shape)
    method = GuidedIG()

    result = method.GetMask(
        im, call_model_function, call_model_args, x_steps=steps, x_baseline=baseline)
    return np.expand_dims(result, axis=0)


def agi(model, data, target, epsilon=0.05, max_iter=20, topk=20, num_classes=1000):
    r"""
    agi explanation
    Arguments:
        model (nn.Sequential): Black-box model being explained.
        data (torch.Tensor): Input tensor.
        target (torch.Tensor): Target tensor.
        epsilon (float): Maximum perturbation that the attacker can introduce. (Default: 0.05)
        max_iter (int): Maximum number of iterations. (Default: 20)
        topk (int): topk selected classes. (Default: 20)
        num_classes (int): Number of classes. (Default: 1000)
    
    Examples::
        >>> explanation = agi(model, images, labels, epsilon=0.05, max_iter=20, topk=20, num_classes=1000)
        >>> attribution_map = explanation(model, images, labels)
    """
    model = copy.deepcopy(model)[:2]
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    random.seed(3407)
    selected_ids = random.sample(list(range(0, num_classes-1)), topk)
    output = model(data)

    init_pred = output.argmax(-1)

    top_ids = selected_ids

    step_grad = 0

    for l in top_ids:

        targeted = torch.tensor([l] * data.shape[0]).to(device)

        if l < 999:
            targeted[targeted == init_pred] = l + 1
        else:
            targeted[targeted == init_pred] = l - 1

        delta, _ = pgd_step(
            data, epsilon, model, init_pred, targeted, max_iter)
        step_grad += delta

    adv_ex = step_grad.squeeze().detach().cpu().numpy()
    return adv_ex


def big(model, data, target, data_min=0, data_max=1, epsilons=[36, 64, 0.3 * 255, 0.5 * 255, 0.7 * 255, 0.9 * 255, 1.1 * 255], class_num=1000, gradient_steps=50):
    r"""
    big explanation
    Arguments:
        model (nn.Sequential): Black-box model being explained.
        data (torch.Tensor): Input tensor.
        target (torch.Tensor): Target tensor.
        data_min (float): Minimum value of input. (Default: 0)
        data_max (float): Maximum value of input. (Default: 1)
        epsilons (list): epsilons. (Default: [36, 64, 0.3 * 255, 0.5 * 255, 0.7 * 255, 0.9 * 255, 1.1 * 255])
        class_num (int): Number of classes. (Default: 1000)
        gradient_steps (int): Number of iterations. (Default: 50)
        
    Examples::
        >>> explanation = big(model, images, labels, data_min=0, data_max=1, epsilons=[36, 64, 0.3 * 255, 0.5 * 255, 0.7 * 255, 0.9 * 255, 1.1 * 255], class_num=1000, gradient_steps=50)
        >>> attribution_map = explanation(model, images, labels)
    """
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    attacks = [FGSM(eps, data_min, data_max) for eps in epsilons]
    big = BIG(model, attacks, class_num)
    attribution_map, _ = big(model, data, target, gradient_steps)
    return attribution_map


def ig(model, data, target, gradient_steps=50):
    r"""
    ig explanation
    Arguments:
        model (nn.Sequential): Black-box model being explained.
        data (torch.Tensor): Input tensor.
        target (torch.Tensor): Target tensor.
        gradient_steps (int): Number of iterations. (Default: 50)
        
    Examples::
        >>> explanation = ig(model, images, labels, gradient_steps=50)
        >>> attribution_map = explanation(model, images, labels)
    """
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    ig = IntegratedGradient(model)
    return ig(data, target, gradient_steps=gradient_steps)


def sm(model, data, target, *args):
    r"""
    sm explanation
    Arguments:
        model (nn.Sequential): Black-box model being explained.
        data (torch.Tensor): Input tensor.
        target (torch.Tensor): Target tensor.
        
    Examples::
        >>> explanation = sm(model, images, labels)
        >>> attribution_map = explanation(model, images, labels)
    """
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    sm = SaliencyGradient(model)
    return sm(data, target)


def sg(model, data, target, stdevs=0.15, gradient_steps=50):
    r"""
    sg explanation
    Arguments:
        model (nn.Sequential): Black-box model being explained.
        data (torch.Tensor): Input tensor.
        target (torch.Tensor): Target tensor.
        stdevs (float): Standard deviation of noise. (Default: 0.15)
        gradient_steps (int): Number of iterations. (Default: 50)
        
    Examples::
        >>> explanation = sg(model, images, labels, stdevs=0.15, gradient_steps=50)
        >>> attribution_map = explanation(model, images, labels)
    """
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    sg = SmoothGradient(model, stdevs=stdevs)
    return sg(data, target, gradient_steps=gradient_steps)


def deeplift(model, data, target, *args):
    r"""
    deeplift explanation
    Arguments:
        model (nn.Sequential): Black-box model being explained.
        data (torch.Tensor): Input tensor.
        target (torch.Tensor): Target tensor.
        
    Examples::
        >>> explanation = deeplift(model, images, labels)
        >>> attribution_map = explanation(model, images, labels)
    """
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    dl = DL(model)
    return dl(data, target)


def saliencymap(model, data, target, *args):
    r"""
    saliencymap explanation
    Arguments:
        model (nn.Sequential): Black-box model being explained.
        data (torch.Tensor): Input tensor.
        target (torch.Tensor): Target tensor.
        
    Examples::
        >>> explanation = saliencymap(model, images, labels)
        >>> attribution_map = explanation(model, images, labels)
    """
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    saliencymap = SaliencyMap(model)
    return saliencymap(data, target)




def run_explanation(model, dataloader, method, **kwargs):
    r"""
    Run explanation.
    Arguments:
        model (nn.Sequential): model to explain.
        method (function): explanation method.
        dataloader (torch.utils.data.DataLoader): data loader.
        kwargs: arguments for explanation method.

    Examples::
        >>> attributions,all_data = run_explanation(model, dataloader, agi, epsilon=0.05, max_iter=20, topk=20, num_classes=1000)
    """
    attributions = list()
    all_data = list()
    for data,target in tqdm(dataloader):
        data = data.to(device)
        attributions.append(method(model,data,target,**kwargs))
        all_data.append(data.cpu().detach().numpy())
    attributions = np.concatenate(attributions,axis=0)
    all_data = np.concatenate(all_data,axis=0)
    return attributions,all_data

def caculate_insert_deletion(model, data, attributions, hw=224*224, num_classes=1000, batch_size=100):
    r"""
    Caculate insertion and deletion scores.
    Arguments:
        model (nn.Sequential): model to explain.
        data (np.ndarray): input data.
        attributions (np.ndarray): attribution map.
        hw (int): height * width of the image.
        num_classes (int): number of classes.
        batch_size (int): number of images for one small batch.
        
    Examples::
        >>> caculate_insert_deletion(model, data, attributions, hw=224*224, num_classes=1000, batch_size=100)
    """
    assert data.shape[0] % batch_size == 0
    data = torch.from_numpy(data).float()
    deletion = CausalMetric(
            model, 'del', substrate_fn=torch.zeros_like, hw=hw, num_classes=num_classes)
    insertion = CausalMetric(
        model, 'ins', substrate_fn=torch.zeros_like, hw=hw, num_classes=num_classes)
    scores = {'del': deletion.evaluate(
                data, attributions, batch_size), 'ins': insertion.evaluate(data, attributions, batch_size)}
    scores['ins'] = np.array(scores['ins'])
    scores['del'] = np.array(scores['del'])
    print('Insertion: ', scores['ins'].mean())
    print('Deletion: ', scores['del'].mean())