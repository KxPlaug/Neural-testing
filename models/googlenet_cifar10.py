from utils import count_num_layers,check_device
import torch
import torch.nn as nn
from torchvision.transforms import Normalize
import torch.nn.functional as F
device = check_device()
from models.pytorch_cifar10.googlenet import googlenet

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2471, 0.2435, 0.2616)

class ComposedModel():
    r"""
    ComposedModel

    Model Structure:
        Normalize -> Model -> Softmax
    """
    def get_model(self):
         # modify this line to use your own model
        self.model = googlenet(num_classes=10)
        state_dict = torch.load('ckpt/googlenet.pt', map_location=device)
        self.model.load_state_dict(state_dict) # load state dict
        self.output_model = nn.Sequential()
        # configure normalization layer
        self._configure_normalization(MEAN, STD)
        self.normalize = Normalize(MEAN, STD)
        self.output_model.add_module('normalize', self.normalize)
        self.output_model.add_module('model', self.model)
        # check if the last layer is Softmax or not
        if not self._check_last_layer(list(self.model.children())[-1]):
            # if not, add a Softmax layer
            self.output_layer = nn.Softmax(dim=-1)
            self.output_model.add_module('output_layer', self.output_layer)
        self.output_model.is_single_branch = False #  refer to the definition of single_branch_model
        self.output_model.num_layers = count_num_layers(self.model) # refer to the definition of num_layers
        self.output_model.eval()
        self.output_model.to(device)
        self.output_model.get_loss = self.get_loss
        return self.output_model
        

    def get_loss(self, inputs, targets):
        """Compute loss for given inputs and targets.

        Args:
            inputs (Tensor): batch of inputs.
            targets (Tensor): batch of targets.

        Returns:
            outputs (Tensor): batch of outputs.
            loss (Tensor): loss value.
        """
        outputs = self.output_model(inputs)
        return outputs,F.cross_entropy(outputs, targets)

    def _configure_normalization(self, mean, std):
        """Configure normalization layer.

        Args:
            mean (sequence): Sequence of means for each channel.
            std (sequence): Sequence of standard deviations for each channel.
        """
        self.normalize = Normalize(mean, std)

    def _check_last_layer(self, last_layer):
        """check if the last layer is Softmax or not

        Args:
            last_layer (nn.Module or nn.Sequential): last layer of the model
        """
        if isinstance(last_layer, nn.Sequential) or isinstance(last_layer, nn.ModuleList):
            last_layer = last_layer[-1]
            return self._check_last_layer(last_layer)
        elif isinstance(last_layer, (nn.Softmax, nn.LogSoftmax)):
            return True
        else:
            return False
