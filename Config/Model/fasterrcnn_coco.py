from torch import nn
from torchvision.transforms import Normalize
import torch.nn.functional as F
import torch
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
from utils import count_num_layers,check_device
device = check_device()

TYPE = 'Object Detection'


class ComposedModel():
    def get_model(self):
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)  # modify this line to use your own model
        # if dataset is normalized, then no need to normalize again
        self.model.eval()
        self.model.to(device)
        self.model.get_loss = self.get_loss
        return self.model
        

    def get_loss(self, inputs, targets):
        """Compute loss for given inputs and targets.

        Args:
            inputs (Tensor): batch of inputs.
            targets (Tensor): batch of targets.

        Returns:
            outputs (Tensor): batch of outputs.
            loss (Tensor): loss value.
        """
        self.output_model.train()
        loss_dict = self.output_model(inputs,targets)
        loss = sum(loss for loss in loss_dict.values())
        return None,loss
