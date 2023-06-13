from torch import nn
from torchvision.transforms import Normalize
import torch.nn.functional as F
import torch
from torchvision.models import resnet50, ResNet50_Weights
from utils import count_num_layers,check_device
device = check_device()

MEAN = [0.485, 0.456, 0.406]  # MODIFY THIS LINE
STD = [0.229, 0.224, 0.225]  # MODIFY THIS LINE
NUM_CLASSES = 1000  # MODIFY THIS LINE
INPUT_SIZE = (224, 224)  # MODIFY THIS LINE
TYPE = 'Image Classification'  # MODIFY THIS LINE, options: 'Image Classification', 'Object Detection', 'Text Classification'
# single_branch_model = [
#     'mlp_7_linear',
#     'lenet',
#     'alexnet',
#     'vgg',
# ]

class ComposedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet50(num_classes=NUM_CLASSES)  # modify this line to use your own model
        weights = ResNet50_Weights.verify(ResNet50_Weights.DEFAULT)
        self.model.load_state_dict(weights.get_state_dict(
            progress=True))  # modify this line to load your own model weights
        self._configure_normalization(MEAN, STD)
        # if dataset is normalized, then no need to normalize again
        self.need_normalize = True
        # check if the last layer is Softmax or not
        if not self._check_last_layer(list(self.model.children())[-1]):
            # if not, add a Softmax layer
            self.output_layer = nn.Softmax(dim=-1)
        else:
            self.output_layer = nn.Identity()  # if yes, add an Identity layer
        self.is_single_branch = False #  refer to the definition of single_branch_model
        self.num_layers = count_num_layers(self.model) # refer to the definition of num_layers
        self.eval()
        self.to(device)
        
    def forward(self, x):
        if self.need_normalize:
            x = self.normalize(x)
        x = self.model(x)
        x = self.output_layer(x)
        return x

    def get_loss(self, inputs, targets):
        """Compute loss for given inputs and targets.

        Args:
            inputs (Tensor): batch of inputs.
            targets (Tensor): batch of targets.

        Returns:
            outputs (Tensor): batch of outputs.
            loss (Tensor): loss value.
        """
        outputs = self.model(inputs)
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
