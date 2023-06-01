from .core import FastIG, GuidedIG, pgd_step, BIG, FGSM, SaliencyGradient, SmoothGradient, DL, IntegratedGradient, SaliencyMap
import torch
import numpy as np
import random
from tqdm import tqdm
from utils import check_device
import os
from Explanation.evaluation.tools import CausalMetric
from glob import glob
device = check_device()


def fast_ig(model, data, target, *args):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    method = FastIG(model)
    result = method(data, target).squeeze()
    return np.expand_dims(result, axis=0)


def guided_ig(model, data, target, steps=15):
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
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    attacks = [FGSM(eps, data_min, data_max) for eps in epsilons]
    big = BIG(model, attacks, class_num)
    attribution_map, _ = big(model, data, target, gradient_steps)
    return attribution_map


def ig(model, data, target, gradient_steps=50):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    ig = IntegratedGradient(model)
    return ig(data, target, gradient_steps=gradient_steps)


def sm(model, data, target, *args):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    sm = SaliencyGradient(model)
    return sm(data, target)


def sg(model, data, target, stdevs=0.15, gradient_steps=50):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    sg = SmoothGradient(model, stdevs=stdevs)
    return sg(data, target, gradient_steps=gradient_steps)


def deeplift(model, data, target, *args):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    dl = DL(model)
    return dl(data, target)


def saliencymap(model, data, target, *args):
    assert len(data.shape) == 4, "Input data must be 4D tensor"
    saliencymap = SaliencyMap(model)
    return saliencymap(data, target)


def process_dataloader(model, dataloader, method, experiment_name, batch_save, *args):
    attributions = []
    all_data = []
    all_target = []
    for i, (data, target) in tqdm(enumerate(dataloader), total=len(dataloader)):
        data = data.to(device)
        target = target.to(device)
        attribution = method(model, data, target, *args)
        data = data.cpu().detach().numpy()
        target = target.cpu().detach().numpy()
        if len(attribution.shape) == 3:
            attribution = np.expand_dims(attribution, axis=0)
            data = np.expand_dims(data, axis=0)
        if batch_save:
            save_attributions(attribution, data, target, experiment_name, i)
        else:
            attributions.append(attribution)
            all_data.append(data)
            all_target.append(target)
    if not batch_save:
        attributions = np.concatenate(attributions, axis=0)
        all_data = np.concatenate(all_data, axis=0)
        all_target = np.concatenate(all_target, axis=0)
        return save_attributions(attributions, all_data, all_target, experiment_name)


def save_attributions(attributions, all_data, all_target, experiment_name, batch_id=None):
    os.makedirs(f'outputs/Explanation/{experiment_name}', exist_ok=True)
    results = {
        'attributions': attributions,
        'data': all_data,
        'target': all_target
    }
    if batch_id is None:
        batch_id = "attributions"
    np.savez(
        f'outputs/Explanation/{experiment_name}/{batch_id}.npz', **results)
    return results


def caculate_insert_deletion(model, experiment_name, hw, num_classes, batch_size, results=None):
    if results is not None:
        attributions = results['attributions']
        data = torch.from_numpy(results['data']).to(device)
        deletion = CausalMetric(
            model, 'del', substrate_fn=torch.zeros_like, hw=hw, num_classes=num_classes)
        insertion = CausalMetric(
            model, 'ins', substrate_fn=torch.zeros_like, hw=hw, num_classes=num_classes)
        if len(data) % batch_size == 0:
            scores = {'del': deletion.evaluate(
                data, attributions, batch_size), 'ins': insertion.evaluate(data, attributions, batch_size)}
            scores['ins'] = np.array(scores['ins'])
            scores['del'] = np.array(scores['del'])
        else:
            leave = len(data) % batch_size
            if len(data) > batch_size:
                scores = {'del': deletion.evaluate(
                    data[:-leave], attributions[:-leave], batch_size), 'ins': insertion.evaluate(data[:-leave], attributions[:-leave], batch_size)}
                scores['ins'] = np.array(scores['ins'])
                scores['del'] = np.array(scores['del'])
            else:
                scores = {'del': np.array([]), 'ins': np.array([])}
            scores['ins'] = np.append(scores['ins'], insertion.evaluate(
                data[-leave:], attributions[-leave:], leave))
            scores['del'] = np.append(scores['del'], deletion.evaluate(
                data[-leave:], attributions[-leave:], leave))
        with open(f'outputs/Explanation/{experiment_name}/scores.txt', 'w') as f:
            f.write("Insertion: " + str(scores['ins'].mean()) + "\n")
            f.write("Deletion: " + str(scores['del'].mean()) + "\n")
            print("Insertion: " + str(scores['ins'].mean()))
            print("Deletion: " + str(scores['del'].mean()))
    else:
        results_files = glob.glob(
            f'outputs/Explanation/{experiment_name}/*.npz')
        scores = {'del': [], 'ins': []}
        for file in results_files:
            results = np.load(file)
            attributions = results['attributions']
            data = torch.from_numpy(results['data']).to(device)
            deletion = CausalMetric(
                model, 'del', substrate_fn=torch.zeros_like, hw=hw, num_classes=num_classes)
            insertion = CausalMetric(
                model, 'ins', substrate_fn=torch.zeros_like, hw=hw, num_classes=num_classes)
            scores['del'].extend(deletion.evaluate(
                data, attributions, len(data)).tolist())
            scores['ins'].extend(insertion.evaluate(
                data, attributions, len(data)).tolist())
        scores['ins'] = np.array(scores['ins'])
        scores['del'] = np.array(scores['del'])
        with open(f'outputs/Explanation/{experiment_name}/scores.txt', 'w') as f:
            f.write("Insertion: " + str(scores['ins'].mean()) + "\n")
            f.write("Deletion: " + str(scores['del'].mean()) + "\n")
            print("Insertion: " + str(scores['ins'].mean()))
            print("Deletion: " + str(scores['del'].mean()))
