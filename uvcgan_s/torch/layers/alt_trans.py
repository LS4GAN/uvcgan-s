# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes

import torch
from torch import nn

from uvcgan_s.torch.select import get_norm_layer
from .attention import select_attention
from .transformer import (
    img_to_pixelwise_tokens, img_from_pixelwise_tokens,
    PositionWiseFFN, ViTInput
)

class AltTransformerBlock(nn.Module):

    def __init__(
        self, features, ffn_features, n_heads, activ = 'gelu', norm = None,
        attention = 'dot', rezero = True, **kwargs
    ):
        super().__init__(**kwargs)

        self.norm1 = get_norm_layer(norm, features)
        self.atten = select_attention(
            attention,
            embed_dim   = features,
            num_heads   = n_heads,
            batch_first = False,
        )

        self.norm2 = get_norm_layer(norm, features)
        self.ffn   = PositionWiseFFN(features, ffn_features, activ)

        self.rezero = rezero

        if rezero:
            self.re_alpha = nn.Parameter(torch.zeros((1, )))
        else:
            self.re_alpha = 1

    def forward(self, x):
        # x: (L, N, features)

        # Step 1: Multi-Head Self Attention
        y1 = self.norm1(x)
        y1, _atten_weights = self.atten(y1, y1, y1)

        y  = x + self.re_alpha * y1

        # Step 2: PositionWise Feed Forward Network
        y2 = self.norm2(y)
        y2 = self.ffn(y2)

        y  = y + self.re_alpha * y2

        return y

    def extra_repr(self):
        return 're_alpha = %e' % (self.re_alpha, )

class AltTransformerEncoder(nn.Module):

    def __init__(
        self, features, ffn_features, n_heads, n_blocks, activ, norm,
        attention = 'dot', rezero = True, **kwargs
    ):
        super().__init__(**kwargs)

        self.encoder = nn.Sequential(*[
            AltTransformerBlock(
                features, ffn_features, n_heads, activ, norm, attention, rezero
            ) for _ in range(n_blocks)
        ])

    def forward(self, x):
        # x : (N, L, features)

        # y : (L, N, features)
        y = x.permute((1, 0, 2))
        y = self.encoder(y)

        # result : (N, L, features)
        result = y.permute((1, 0, 2))

        return result

class AltPixelwiseViT(nn.Module):

    def __init__(
        self, features, n_heads, n_blocks, ffn_features, embed_features,
        activ, norm, image_shape, attention = 'dot', rezero = True, **kwargs
    ):
        super().__init__(**kwargs)

        self.image_shape = image_shape

        self.trans_input = ViTInput(
            image_shape[0], embed_features, features,
            image_shape[1], image_shape[2],
        )

        self.encoder = AltTransformerEncoder(
            features, ffn_features, n_heads, n_blocks, activ, norm,
            attention, rezero
        )

        self.trans_output = nn.Linear(features, image_shape[0])

    def forward(self, x):
        # x : (N, C, H, W)

        # itokens : (N, L, C)
        itokens = img_to_pixelwise_tokens(x)

        # y : (N, L, features)
        y = self.trans_input(itokens)
        y = self.encoder(y)

        # otokens : (N, L, C)
        otokens = self.trans_output(y)

        # result : (N, C, H, W)
        result = img_from_pixelwise_tokens(otokens, self.image_shape)

        return result

