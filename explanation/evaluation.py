import torch
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from utils import check_device
device = check_device()


class CausalMetric():

    def __init__(self, model, mode, substrate_fn, hw, num_classes):
        r"""Create deletion/insertion metric instance.
        Args:
            model (nn.Module): Black-box model being explained.
            mode (str): 'del' or 'ins'.
            substrate_fn (func): a mapping from old pixels to new pixels.
            hw (int): height * width of the image.
            num_classes (int): number of classes.
        """
        assert mode in ['del', 'ins']
        self.model = model
        self.mode = mode
        self.step = int(hw ** 0.5)
        self.substrate_fn = substrate_fn
        self.hw = hw
        self.num_classes = num_classes

    def evaluate(self, img_batch, exp_batch, batch_size):
        r"""Efficiently evaluate big batch of images.
        Args:
            img_batch (Tensor): batch of images.
            exp_batch (np.ndarray): batch of explanations.
            batch_size (int): number of images for one small batch.
        Returns:
            scores (nd.array): Array containing scores at every step for every image.
        """
        img_batch = img_batch.cpu()
        n_samples = img_batch.shape[0]
        predictions = torch.FloatTensor(n_samples, self.num_classes)
        assert n_samples % batch_size == 0
        for i in tqdm(range(n_samples // batch_size), desc='Predicting labels'):
            preds = self.model(
                img_batch[i*batch_size:(i+1)*batch_size].to(device)).cpu().detach()
            predictions[i*batch_size:(i+1)*batch_size] = preds
        top = np.argmax(predictions, -1)
        n_steps = (self.hw + self.step - 1) // self.step
        scores = np.empty((n_steps + 1, n_samples))
        salient_order = np.flip(np.argsort(
            exp_batch.reshape(n_samples, 3, self.hw), axis=-1), axis=-1)
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
                preds = self.model(start[j*batch_size:(j+1)*batch_size].to(device))
                preds = preds.cpu().detach().numpy()[range(
                    batch_size), top[j*batch_size:(j+1)*batch_size]]
                scores[i, j*batch_size:(j+1)*batch_size] = preds
            # Change specified number of most salient pixels to substrate pixels
            coords = salient_order[:, :, self.step * i:self.step * (i + 1)]
            for n_sample in range(n_samples):
                for channel in range(3):
                    start.cpu().numpy().reshape(n_samples, 3, self.hw)[n_sample, channel, coords[n_sample]] = finish.cpu(
                    ).numpy().reshape(n_samples, 3, self.hw)[n_sample, channel, coords[n_sample]]
        return scores
