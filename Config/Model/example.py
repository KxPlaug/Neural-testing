from torch import nn
from torchvision.transforms import Normalize
import torch.nn.functional as F
import torch
from torchvision.models import resnet50, ResNet50_Weights


MEAN = [0.485, 0.456, 0.406]  # MODIFY THIS LINE
STD = [0.229, 0.224, 0.225]  # MODIFY THIS LINE
NEED_NORMALIZE = True  # MODIFY THIS LINE
NUM_CLASSES = 1000  # MODIFY THIS LINE
INPUT_SIZE = (224, 224)  # MODIFY THIS LINE


# class SampleModel(nn.Module):
#     def __init__(self, n_classes=10):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
#         self.relu1 = nn.ReLU()
#         self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
#         self.relu2 = nn.ReLU()
#         self.pool = nn.MaxPool2d(2, 2)
#         self.classifier = nn.Sequential(
#             nn.Linear(32 * 8 * 8, 128),
#             nn.ReLU(),
#             nn.Linear(128, n_classes)
#         )

#     def forward(self, x):
#         x = self.normalize(x)
#         x = self.relu1(self.conv1(x))
#         x = self.pool(x)
#         x = self.relu2(self.conv2(x))
#         x = self.pool(x)
#         x = x.view(-1, 32 * 8 * 8)
#         x = self.classifier(x)
#         return x


class ComposedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet50(num_classes=NUM_CLASSES)  # modify this line to use your own model
        weights = ResNet50_Weights.verify(ResNet50_Weights.DEFAULT)
        self.model.load_state_dict(weights.get_state_dict(
            progress=True))  # modify this line to load your own model weights
        self._configure_normalization(MEAN, STD)
        # if dataset is normalized, then no need to normalize again
        self.need_normalize = NEED_NORMALIZE
        # check if the last layer is Softmax or not
        if not self._check_last_layer(list(self.model.children())[-1]):
            # if not, add a Softmax layer
            self.output_layer = nn.Softmax(dim=-1)
        else:
            self.output_layer = nn.Identity()  # if yes, add an Identity layer

    def forward(self, x):
        if self.need_normalize:
            x = self.normalize(x)
        x = self.model(x)
        x = self.output_layer(x)
        return x

    def compute_loss(self, inputs, targets):
        """Compute loss for given inputs and targets.

        Args:
            inputs (Tensor): batch of inputs.
            targets (Tensor): batch of targets.

        Returns:
            loss (Tensor): loss value.
        """
        return F.cross_entropy(self.model(inputs), targets)

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
