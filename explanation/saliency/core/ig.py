from captum.attr import IntegratedGradients

class IntegratedGradient:
    r"""
    Integrated Gradient
    Arguments:
        model (nn.Sequential): Black-box model being explained.
        
    __call__ :
        Arguments:
            data (torch.Tensor): Input tensor.
            target (torch.Tensor): Target tensor.
            gradient_steps (int): Number of gradient steps. (Default: 50)
            
        Returns:
            np.ndarray: Attribution map.
            
    Examples::
        >>> explanation = IntegratedGradient(model)
        >>> attribution_map = explanation(images, labels)
    """
    def __init__(self, model):
        self.model = model
        self.saliency = IntegratedGradients(model)

    def __call__(self, data, target, gradient_steps=50):
        attribution_map = self.saliency.attribute(data,
                                                  target=target.squeeze(),
                                                  baselines=None,
                                                  n_steps=gradient_steps,
                                                  method="riemann_trapezoid")
        return attribution_map.detach().cpu().numpy()