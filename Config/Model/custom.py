from torch import nn
from torchvision.transforms import Normalize
import torch.nn.functional as F
import torch
from utils import count_num_layers,check_device
device = check_device()

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks,num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out



MEAN = [0.5, 0.5, 0.5]  # MODIFY THIS LINE
STD = [0.5, 0.5, 0.5]  # MODIFY THIS LINE
NUM_CLASSES = 10  # MODIFY THIS LINE
INPUT_SIZE = (32, 32)  # MODIFY THIS LINE
# MODIFY THIS LINE, options: 'Image Classification', 'Object Detection', 'Text Classification'
TYPE = 'Image Classification'
# single_branch_model = [
#     'mlp_7_linear',
#     'lenet',
#     'alexnet',
#     'vgg',
# ]

class ComposedModel():
    def get_model(self):
         # modify this line to use your own model
        self.model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=NUM_CLASSES)
        state_dict = torch.load(
            f"Config/Model/weights/cifar10_resnet50_weights.pth", map_location=device)
        for key in list(state_dict.keys()):
            if key.startswith('module.'):
                state_dict[key[7:]] = state_dict[key]
                del state_dict[key]
        self.model.load_state_dict(state_dict)
        self.output_model = nn.Sequential()
        self._configure_normalization(MEAN, STD)
        # if dataset is normalized, then no need to normalize again
        self.need_normalize = True
        if self.need_normalize:
            self.normalize = Normalize(MEAN, STD)
            self.output_model.add_module('normalize', self.normalize)
        self.output_model.add_module('model', self.model)
        # check if the last layer is Softmax or not
        if not self._check_last_layer(list(self.model.children())[-1]):
            # if not, add a Softmax layer
            self.output_layer = nn.Softmax(dim=-1)
            self.output_model.add_module('output_layer', self.output_layer)
        self.is_single_branch = False #  refer to the definition of single_branch_model
        self.num_layers = count_num_layers(self.model) # refer to the definition of num_layers
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
