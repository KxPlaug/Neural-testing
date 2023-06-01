from tqdm import tqdm
import torch
import numpy as np
from matplotlib import pyplot as plt
import json
from utils import check_device


# Plots image from tensor
def tensor_imshow(inp, title=None, **kwargs):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    # Mean and std for ImageNet
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp, **kwargs)
    if title is not None:
        plt.title(title)


# Given label number returns class name
def get_class_name(c):
    labels = json.load(open("imagenet_class_index.json"))
    # labels = np.loadtxt('synset_words.txt', str, delimiter='\t')
    return labels[str(c)][1]


def get_custom_class_name(c):
    labels = json.load(open("custom_class_index.json"))
    return labels[str(c)]


# HW = 224 * 224  # image area
# n_classes = 1000
device = check_device()


def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)


class CausalMetric():

    def __init__(self, model, mode, substrate_fn, hw, num_classes):
        r"""Create deletion/insertion metric instance.

        Args:
            model (nn.Module): Black-box model being explained.
            mode (str): 'del' or 'ins'.
            step (int): number of pixels modified per one iteration.
            substrate_fn (func): a mapping from old pixels to new pixels.
        """
        assert mode in ['del', 'ins']
        self.model = model
        self.mode = mode
        self.step = int(hw ** 0.5)
        self.substrate_fn = substrate_fn
        self.hw = hw
        self.num_classes = num_classes

    def single_run(self, img_tensor, explanation, verbose=0, save_to=None):
        r"""Run metric on one image-saliency pair.

        Args:
            img_tensor (Tensor): normalized image tensor.
            explanation (np.ndarray): saliency map.
            verbose (int): in [0, 1, 2].
                0 - return list of scores.
                1 - also plot final step.
                2 - also plot every step and print 2 top classes.
            save_to (str): directory to save every step plots to.

        Return:
            scores (nd.array): Array containing scores at every step.
        """
        pred = self.model(img_tensor.to(device))
        top, c = torch.max(pred, 1)
        c = c.cpu().numpy()[0]
        n_steps = (self.hw + self.step - 1) // self.step

        if self.mode == 'del':
            title = 'Deletion game'
            ylabel = 'Pixels deleted'
            start = img_tensor.clone()
            finish = self.substrate_fn(img_tensor)
        elif self.mode == 'ins':
            title = 'Insertion game'
            ylabel = 'Pixels inserted'
            start = self.substrate_fn(img_tensor)
            finish = img_tensor.clone()

        scores = np.empty(n_steps + 1)
        # Coordinates of pixels in order of decreasing saliency
        salient_order = np.flip(np.argsort(
            explanation.reshape(-1, self.hw), axis=1), axis=-1)
        for i in range(n_steps+1):
            pred = self.model(start.to(device))
            pr, cl = torch.topk(pred, 2)
            if verbose == 2:
                print('{}: {:.3f}'.format(
                    get_custom_class_name(cl[0][0]), float(pr[0][0])))
                print('{}: {:.3f}'.format(
                    get_custom_class_name(cl[0][1]), float(pr[0][1])))
            scores[i] = pred[0, c]
            # Render image if verbose, if it's the last step or if save is required.
            if verbose == 2 or (verbose == 1 and i == n_steps) or save_to:
                plt.figure(figsize=(10, 5))
                plt.subplot(121)
                plt.title('{} {:.1f}%, P={:.4f}'.format(
                    ylabel, 100 * i / n_steps, scores[i]))
                plt.axis('off')
                tensor_imshow(start[0])

                plt.subplot(122)
                plt.plot(np.arange(i+1) / n_steps, scores[:i+1])
                plt.xlim(-0.1, 1.1)
                plt.ylim(0, 1.05)
                plt.fill_between(np.arange(i+1) / n_steps,
                                 0, scores[:i+1], alpha=0.4)
                plt.title(title)
                plt.xlabel(ylabel)
                plt.ylabel(get_custom_class_name(c))
                if save_to:
                    plt.savefig(save_to + '/{:03d}.png'.format(i))
                    plt.close()
                else:
                    plt.show()
            if i < n_steps:
                coords = salient_order[:, self.step * i:self.step * (i + 1)]
                start.cpu().numpy().reshape(1, 3, self.hw)[
                    0, :, coords] = finish.cpu().numpy().reshape(1, 3, self.hw)[0, :, coords]
        return scores

    def evaluate(self, img_batch, exp_batch, batch_size):
        r"""Efficiently evaluate big batch of images.

        Args:
            img_batch (Tensor): batch of images.
            exp_batch (np.ndarray): batch of explanations.
            batch_size (int): number of images for one small batch.

        Returns:
            scores (nd.array): Array containing scores at every step for every image.
        """
        n_samples = img_batch.shape[0]
        predictions = torch.FloatTensor(n_samples, self.num_classes)
        assert n_samples % batch_size == 0
        for i in tqdm(range(n_samples // batch_size), desc='Predicting labels'):
            preds = self.model(
                img_batch[i*batch_size:(i+1)*batch_size].to(device)).detach().cpu()
            predictions[i*batch_size:(i+1)*batch_size] = preds
        top = np.argmax(predictions, -1)
        n_steps = (self.hw + self.step - 1) // self.step
        scores = np.empty((n_steps + 1, n_samples))
        salient_order = np.flip(np.argsort(
            exp_batch.reshape(-1, self.hw), axis=1), axis=-1)
        r = np.arange(n_samples).reshape(n_samples, 1)

        substrate = torch.zeros_like(img_batch)
        for j in tqdm(range(n_samples // batch_size), desc='Substrate'):
            substrate[j*batch_size:(j+1)*batch_size] = self.substrate_fn(
                img_batch[j*batch_size:(j+1)*batch_size])

        if self.mode == 'del':
            caption = 'Deleting  '
            start = img_batch.clone()
            finish = substrate
        elif self.mode == 'ins':
            caption = 'Inserting '
            start = substrate
            finish = img_batch.clone()

        # While not all pixels are changed
        for i in tqdm(range(n_steps+1), desc=caption + 'pixels'):
            # Iterate over batches
            for j in range(n_samples // batch_size):
                # Compute new scores
                preds = self.model(
                    start[j*batch_size:(j+1)*batch_size].to(device))
                preds = preds.detach().cpu().numpy()[range(
                    batch_size), top[j*batch_size:(j+1)*batch_size]]
                scores[i, j*batch_size:(j+1)*batch_size] = preds
            # Change specified number of most salient pixels to substrate pixels
            coords = salient_order[:, self.step * i:self.step * (i + 1)]
            if i < n_steps:
                for rr in r:
                    start.cpu().numpy().reshape(n_samples, 3, self.hw)[rr, :, coords.reshape(n_samples, 3, n_steps)[
                        rr]] = finish.cpu().numpy().reshape(n_samples, 3, self.hw)[rr, :, coords.reshape(n_samples, 3, n_steps)[rr]]
        print('AUC: {}'.format(auc(scores.mean(1))))
        return scores.transpose()
