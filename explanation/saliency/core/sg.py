from captum.attr import Saliency, NoiseTunnel

class SmoothGradient:
    r"""
    SG
    Arguments:
        model (nn.Sequential): Black-box model being explained.
        stdevs (float): Standard deviation of noise. (Default: 0.15)
        
    __call__ :
        Arguments:
            data (torch.Tensor): Input tensor.
            target (torch.Tensor): Target tensor.
            gradient_steps (int): Number of gradient steps. (Default: 50)
            
        Returns:
            np.ndarray: Attribution map.
    """
    def __init__(self, model, stdevs=0.15):
        self.model = model
        self.saliency = NoiseTunnel(Saliency(model))
        self.stdevs = stdevs

    def __call__(self, data, target, gradient_steps=50):
        attribution_map = self.saliency.attribute(data,
                                                  target=target,
                                                  nt_samples = gradient_steps,
                                                  stdevs=self.stdevs,
                                                  abs=False)
        return attribution_map.detach().cpu().numpy()
