import math

import torch
from torch import nn

from torch.nn.init import _calculate_correct_fan, calculate_gain

class LearningRateEqualizer(nn.Module):

    def __init__(self, mode = 'fan_in', nonlinearity = 'leaky_relu', a = 0):
        super().__init__()

        self._mode  = mode
        self._activ = nonlinearity
        self._param = a

    def _calc_scale(self, w):
        fan  = _calculate_correct_fan(w, self._mode)
        gain = calculate_gain(self._activ, self._param)
        std  = gain / math.sqrt(fan)

        return std

    def forward(self, w):
        scale = self._calc_scale(w)
        return w * scale

# NOTE:
# WARNING:
#   Behavior of parametrization changes between pytorch 1.9 and 1.10.
#   In pytorch  1.9 original_weight = unparametrized weight
#   In pytorch >1.9 original_weight = right_inverse(unparametrized weight)
#   To make the behavior consistent -- the right_inverse is masked for now.
#
#    def right_inverse(self, w):
#        scale = self._calc_scale(w)
#        return w / scale

def apply_lr_equal_to_module(module, name, param):
    if isinstance(module, torch.nn.utils.parametrize.ParametrizationList):
        return

    if not hasattr(module, name):
        return

    w = getattr(module, name)

    if (w is None) or len(w.shape) < 2:
        return

    torch.nn.utils.parametrize.register_parametrization(module, name, param)

def apply_lr_equal(module, tensor_name = "weight", **kwargs):
    parametrization = LearningRateEqualizer(**kwargs)
    submodule_list  = [ x[1] for x in module.named_modules() ]

    for m in submodule_list:
        apply_lr_equal_to_module(m, tensor_name, parametrization)

