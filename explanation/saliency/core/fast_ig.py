class FastIG:
    def __init__(self, model):
        self.model = model

    def __call__(self, data, target):
        data.requires_grad_()
        _,loss = self.model.get_loss(data, target)
        loss.backward()
        return (data * data.grad).detach().cpu().numpy()