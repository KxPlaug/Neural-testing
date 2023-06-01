import torch

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