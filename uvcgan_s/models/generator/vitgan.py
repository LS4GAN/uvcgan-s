# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes

import torch

import numpy as np
from torch import nn

from uvcgan_s.torch.select import select_activation
from uvcgan_s.torch.layers.transformer import (
    calc_tokenized_size, img_to_tokens, img_from_tokens,
    ViTInput, TransformerEncoder, FourierEmbedding
)

class ModulatedLinear(nn.Module):
    # arXiv: 2011.13775

    def __init__(
        self, in_features, out_features, w_features, activ, eps = 1e-8,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        self.bias   = nn.Parameter(torch.empty((out_features,)))
        self._eps   = eps

        self.A     = nn.Linear(w_features, in_features)
        self.activ = select_activation(activ)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight)
        nn.init.zeros_(self.bias)

    def get_modulated_weight(self, s):
        # s : (N, in_features)

        # w : (1, out_features, in_features)
        w = self.weight.unsqueeze(0)

        # s : (N, in_features)
        #  -> (N, 1, in_features)
        s = s.unsqueeze(1)

        # weight : (N, out_features, in_features)
        weight = w * s

        # norm : (N, out_features, 1)
        norm = torch.rsqrt(
            self._eps + torch.sum(weight**2, dim = 2, keepdim = True)
        )

        return weight * norm

    def forward(self, x, w):
        # x : (N, L, in_features)
        # w : (N, w_features)

        # s : (N, in_features)
        s = self.A(w)

        # weight : (N, out_features, in_features)
        weight = self.get_modulated_weight(s)

        # weight : (N, out_features, in_features)
        # x      : (N, L, in_features)
        # result : (N, L, out_features)
        result = torch.matmul(weight.unsqueeze(1), x.unsqueeze(-1)).squeeze(-1)

        result = result + self.bias

        return self.activ(result)

class ViTGANOutput(nn.Module):

    def __init__(self, features, token_shape, activ, **kwargs):
        super().__init__(**kwargs)

        self.embed = ViTGANInput(features, token_shape[1], token_shape[2])

        self.fc1 = ModulatedLinear(features, features, features, activ)
        self.fc2 = ModulatedLinear(features, token_shape[0], features, None)

        self._token_shape = token_shape

    def _map_fn(self, x, w):
        # x : (N, H_t * W_t, features)
        # w : (N, features)

        # result : (N, H_t * W_t, features)
        result = self.fc1(x, w)

        # result : (N, H_t * W_t, C)
        result = self.fc2(result, w)

        # result : (N, H_t * W_t, C)
        #       -> (N, C, H_t * W_t)
        result = result.permute(0, 2, 1)

        # (N, 1, C, H_t, W_t)
        return result.reshape(
            (result.shape[0], 1, result.shape[1], *self._token_shape[1:])
        )

    def forward(self, y):
        # y : (N, L, features)
        # e : (N, H_t * W_t, features)
        e = self.embed(len(y))

        # result : (N, L, C, H_t, W_t)
        result = torch.stack(
            [ self._map_fn(e, w) for w in torch.unbind(y, dim = 1) ],
            dim = 1
        )

        return result

class ViTGANInput(nn.Module):

    def __init__(self, features, height, width, **kwargs):
        super().__init__(**kwargs)
        self._height   = height
        self._width    = width

        x = torch.arange(width).to(torch.float32)
        y = torch.arange(height).to(torch.float32)

        x, y   = torch.meshgrid(x, y)
        self.x = x.reshape((1, -1))
        self.y = y.reshape((1, -1))

        self.register_buffer('x_const', self.x)
        self.register_buffer('y_const', self.y)

        self.embed = FourierEmbedding(features, height, width)

    def forward(self, batch_size = None):
        # result : (1, height * width, features)
        result = self.embed(self.y_const, self.x_const)

        if batch_size is not None:
            # result : (1, height * width, features)
            #       -> (batch_size, height * width, features)
            result = result.expand((batch_size, *result.shape[1:]))

        return result

class ViTGANGenerator(nn.Module):

    def __init__(
        self, features, n_heads, n_blocks, ffn_features, embed_features,
        activ, norm, input_shape, output_shape, token_size, rescale = False,
        rezero = True, **kwargs
    ):
        super().__init__(**kwargs)

        assert input_shape == output_shape
        image_shape = input_shape

        self.image_shape    = image_shape
        self.token_size     = token_size
        self.token_shape    = (image_shape[0], *token_size)
        self.token_features = np.prod([image_shape[0], *token_size])
        self.N_h, self.N_w  = calc_tokenized_size(image_shape, token_size)
        self.rescale        = rescale

        self.gan_input = ViTInput(
            self.token_features, embed_features, features, self.N_h, self.N_w
        )

        self.encoder = TransformerEncoder(
            features, ffn_features, n_heads, n_blocks, activ, norm, rezero
        )

        self.gan_output = ViTGANOutput(features, self.token_shape, 'relu')

    # pylint: disable=no-self-use
    def calc_scale(self, x):
        # x : (N, C, H, W)
        return x.abs().mean(dim = (1, 2, 3), keepdim = True) + 1e-8

    def forward(self, x):
        # x : (N, C, H, W)
        if self.rescale:
            scale = self.calc_scale(x)
            x = x / scale

        # itokens : (N, N_h, N_w, C, H_c, W_c)
        itokens = img_to_tokens(x, self.token_shape[1:])

        # itokens : (N, N_h,  N_w, C,  H_c,  W_c)
        #        -> (N, N_h * N_w, C * H_c * W_c)
        #         = (N, L,         in_features)
        itokens = itokens.reshape((itokens.shape[0], self.N_h * self.N_w, -1))

        # y : (N, L, features)
        y = self.gan_input(itokens)
        y = self.trans(y)

        # otokens : (N, L, C, H_t, W_t)
        otokens = self.gan_output(y)

        # otokens : (N, L,        C, H_t, W_t)
        #        -> (N, N_h, N_w, C, H_c, W_c)
        otokens = otokens.reshape(
            (otokens.shape[0], self.N_h, self.N_w, *otokens.shape[3:])
        )

        result = img_from_tokens(otokens)
        if self.rescale:
            result = result * scale

        return result

