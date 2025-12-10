import torch
from torch import nn

class Exponential(nn.Module):

    def __init__(self, beta = 1):
        super().__init__()
        self._beta = beta

    def forward(self, x):
        return torch.exp(self._beta * x)

