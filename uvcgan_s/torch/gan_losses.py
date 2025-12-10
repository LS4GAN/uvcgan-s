import torch
from torch import nn

from .select import extract_name_kwargs

def reduce_loss(loss, reduction):
    if (reduction is None) or (reduction == 'none'):
        return loss

    if reduction == 'mean':
        return loss.mean()

    if reduction == 'sum':
        return loss.sum()

    raise ValueError(f"Unknown reduction method: '{reduction}'")

class GANLoss(nn.Module):

    def __init__(
        self, label_real = 1, label_fake = 0, reduction = 'mean',
        **kwargs
    ):
        super().__init__(**kwargs)

        self.reduction = reduction
        self.register_buffer('label_real', torch.tensor(label_real))
        self.register_buffer('label_fake', torch.tensor(label_fake))

    def _expand_label_as(self, x, is_real):
        result = self.label_real if is_real else self.label_fake
        return result.to(dtype = x.dtype).expand_as(x)

    def eval_loss(self, x, is_real, is_generator):
        raise NotImplementedError

    def forward(self, x, is_real, is_generator = False):
        if isinstance(x, (list, tuple)):
            result = sum(self.forward(y, is_real, is_generator) for y in x)
            return result / len(x)

        return self.eval_loss(x, is_real, is_generator)

class LSGANLoss(GANLoss):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss = nn.MSELoss(reduction = self.reduction)

    def eval_loss(self, x, is_real, is_generator = False):
        label = self._expand_label_as(x, is_real)
        return self.loss(x, label)

class BCEGANLoss(GANLoss):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss = nn.BCEWithLogitsLoss(reduction = self.reduction)

    def eval_loss(self, x, is_real, is_generator = False):
        label = self._expand_label_as(x, is_real)
        return self.loss(x, label)

class SoftplusGANLoss(GANLoss):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss = nn.Softplus()

    def eval_loss(self, x, is_real, is_generator = False):
        if is_real:
            result = self.loss(x)
        else:
            result = self.loss(-x)

        return reduce_loss(result, self.reduction)

class WGANLoss(GANLoss):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.reduction != 'mean':
            raise NotImplementedError

    def eval_loss(self, x, is_real, is_generator = False):
        if is_real:
            result = -x.mean()
        else:
            result = x.mean()

        return reduce_loss(result, self.reduction)

class HingeGANLoss(GANLoss):

    def __init__(self, margin = 1, **kwargs):
        super().__init__(**kwargs)
        self._margin = margin
        self._relu   = nn.ReLU()

        if self.reduction != 'mean':
            raise NotImplementedError

    def eval_loss(self, x, is_real, is_generator = False):
        if is_generator:
            if is_real:
                result = -x.mean()
            else:
                result = x.mean()
        else:
            if is_real:
                result = self._relu(self._margin - x).mean()
            else:
                result = self._relu(self._margin + x).mean()

        return reduce_loss(result, self.reduction)

GAN_LOSSES = {
    'lsgan'    : LSGANLoss,
    'wgan'     : WGANLoss,
    'softplus' : SoftplusGANLoss,
    'hinge'    : HingeGANLoss,
    'bce'      : BCEGANLoss,
    'vanilla'  : BCEGANLoss,
}

def select_gan_loss(gan_loss):
    name, kwargs = extract_name_kwargs(gan_loss)

    if name in GAN_LOSSES:
        return GAN_LOSSES[name](**kwargs)

    raise ValueError(f"Unknown gan loss: '{name}'")


