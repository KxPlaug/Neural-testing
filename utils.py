import torch
import torch.nn as nn

def check_device() -> torch.device:
    """Check the device of the machine.

    Returns:
        torch.device: The device of the machine.
    """
    value = torch.Tensor([0])
    try:
        try:
            value.to("cuda")
            return torch.device("cuda")
        except:
            value.to("mps")
            return torch.device("mps")
    except:
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

def get_embedding_output(model,embedding,sents):
    def hook(module, input, output):
        output = embedding
        return output
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            handle = module.register_forward_hook(hook)
            break
    model.eval()
    with torch.no_grad():
        output = model(sents)
        handle.remove()
        return output
    