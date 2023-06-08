import torch
import torch.nn as nn

def check_device() -> torch.device:
    """Check the device of the machine.

    Returns:
        torch.device: The device of the machine.
    """
    value = torch.Tensor([0])
    try:
        value.to("cuda")
        return torch.device("cuda")
    except:
        value.to("mps")
        return torch.device("mps")
    finally:
        return torch.device("cpu")
    
def count_num_layers(model):
    num_conv_layers = 0
    num_fc_layers = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            num_conv_layers += 1
        elif isinstance(module, nn.Linear):
            num_fc_layers += 1

    return num_conv_layers+ num_fc_layers