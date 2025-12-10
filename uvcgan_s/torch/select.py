import copy
import torch
from torch import nn

from .layers.activation import Exponential

def extract_name_kwargs(obj):
    if isinstance(obj, dict):
        obj    = copy.copy(obj)
        name   = obj.pop('name')
        kwargs = obj
    else:
        name   = obj
        kwargs = {}

    return (name, kwargs)

def get_norm_layer(norm, features):
    name, kwargs = extract_name_kwargs(norm)

    if name is None:
        return nn.Identity(**kwargs)

    if name == 'layer':
        return nn.LayerNorm((features,), **kwargs)

    if name == 'batch':
        return nn.BatchNorm2d(features, **kwargs)

    if name == 'group':
        return nn.GroupNorm(num_channels = features, **kwargs)

    if name == 'instance':
        return nn.InstanceNorm2d(features, **kwargs)

    raise ValueError("Unknown Layer: '%s'" % name)

def get_norm_layer_fn(norm):
    return lambda features : get_norm_layer(norm, features)

def get_activ_layer(activ):
    # pylint: disable=too-many-return-statements
    name, kwargs = extract_name_kwargs(activ)

    if (name is None) or (name == 'linear'):
        return nn.Identity()

    if name == 'gelu':
        return nn.GELU(**kwargs)

    if name == 'selu':
        return nn.SELU(**kwargs)

    if name == 'relu':
        return nn.ReLU(inplace = True, **kwargs)

    if name == 'leakyrelu':
        return nn.LeakyReLU(inplace = True, **kwargs)

    if name == 'tanh':
        return nn.Tanh()

    if name == 'sigmoid':
        return nn.Sigmoid()

    if name == 'exp':
        return Exponential(**kwargs)

    raise ValueError("Unknown activation: '%s'" % name)

def select_activation(activation):
    return get_activ_layer(activation)

def select_optimizer(parameters, optimizer):
    name, kwargs = extract_name_kwargs(optimizer)

    if name == 'AdamW':
        return torch.optim.AdamW(parameters, **kwargs)

    if name == 'SGD':
        return torch.optim.SGD(parameters, **kwargs)

    if name == 'Adam':
        return torch.optim.Adam(parameters, **kwargs)

    raise ValueError("Unknown optimizer: '%s'" % name)

def select_loss(loss, reduction = None):
    name, kwargs = extract_name_kwargs(loss)

    if reduction is not None:
        kwargs['reduction'] = reduction

    if name.lower() in [ 'l1', 'mae' ]:
        return nn.L1Loss(**kwargs)

    if name.lower() in [ 'l2', 'mse' ]:
        return nn.MSELoss(**kwargs)

    if name.lower() in [ 'bce', 'binary-cross-entropy' ]:
        return nn.BCELoss(**kwargs)

    if name.lower() in [ 'bce-logits', ]:
        return nn.BCEWithLogitsLoss(**kwargs)

    raise ValueError("Unknown loss: '%s'" % name)

