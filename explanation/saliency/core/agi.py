import torch
import torch.nn.functional as F
import numpy as np


def fgsm_step(image, epsilon, data_grad_adv, data_grad_lab):
    delta = epsilon * data_grad_adv.sign()

    perturbed_image = image + delta
    perturbed_rect = torch.clamp(perturbed_image, min=0, max=1)
    delta = perturbed_rect - image
    delta = - data_grad_lab * delta
    return perturbed_rect, delta


def pgd_step(image, epsilon, model, init_pred, targeted, max_iter):
    perturbed_image = image.clone()

    leave_index = np.arange(image.shape[0]).tolist()
    for i in range(max_iter):

        perturbed_image.requires_grad = True
        output = model(perturbed_image)

        pred = output.argmax(-1)

        for j in leave_index:
            if pred[j] == targeted[j]:
                leave_index.remove(j)
        if len(leave_index) == 0:
            break

        output = F.softmax(output, dim=1)

        loss = output[:, targeted].sum()

        model.zero_grad()
        loss.backward(retain_graph=True)
        data_grad_adv = perturbed_image.grad.data.detach().clone()

        loss_lab = output[:, init_pred].sum()
        model.zero_grad()
        perturbed_image.grad.zero_()
        loss_lab.backward()
        data_grad_lab = perturbed_image.grad.data.detach().clone()
        perturbed_image, delta = fgsm_step(
            image, epsilon, data_grad_adv, data_grad_lab)

        if i == 0:
            c_delta = delta
        else:
            c_delta[leave_index] += delta[leave_index]

    return c_delta, perturbed_image
