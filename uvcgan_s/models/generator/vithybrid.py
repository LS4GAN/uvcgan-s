# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes

from torch import nn

from uvcgan_s.torch.select             import get_norm_layer, get_activ_layer
from uvcgan_s.torch.layers.transformer import PixelwiseViT
from uvcgan_s.torch.layers.cnn         import (
    get_downsample_x2_layer, get_upsample_x2_layer
)

def construct_downsample_stem(
    features_list, activ, norm, downsample, image_shape
):
    result = nn.Sequential()

    result.add_module(
        "downsample_base",
        nn.Sequential(
            nn.Conv2d(
                image_shape[0], features_list[0], kernel_size = 3, padding = 1
            ),
            get_activ_layer(activ),
        )
    )

    prev_features = features_list[0]
    curr_size     = image_shape[1:]

    for idx,features in enumerate(features_list):
        layer_down, next_features = \
            get_downsample_x2_layer(downsample, features)

        result.add_module(
            "downsample_block_%d" % idx,
            nn.Sequential(
                get_norm_layer(norm, prev_features),
                nn.Conv2d(
                    prev_features, features, kernel_size = 3, padding = 1
                ),
                get_activ_layer(activ),

                layer_down,
            )
        )

        prev_features = next_features
        curr_size     = (curr_size[0] // 2, curr_size[1] // 2)

    return result, prev_features, curr_size

def construct_upsample_stem(
    features_list, input_features, activ, norm, upsample, image_shape
):
    result        = nn.Sequential()
    prev_features = input_features

    for idx,features in reversed(list(enumerate(features_list))):
        layer_up, next_features \
            = get_upsample_x2_layer(upsample, prev_features)

        result.add_module(
            "upsample_block_%d" % idx,
            nn.Sequential(
                layer_up,

                get_norm_layer(norm, next_features),
                nn.Conv2d(
                    next_features, features, kernel_size = 3, padding = 1
                ),
                get_activ_layer(activ)
            )
        )

        prev_features = features


    result.add_module(
        "upsample_base",
        nn.Sequential(
            nn.Conv2d(prev_features, image_shape[0], kernel_size = 1),
        )
    )

    return result

class ViTHybridGenerator(nn.Module):

    def __init__(
        self, features, n_heads, n_blocks, ffn_features, embed_features,
        activ, norm, input_shape, output_shape,
        stem_features_list, stem_activ, stem_norm,
        stem_downsample = 'conv',
        stem_upsample   = 'upsample-conv',
        rezero          = True,
        **kwargs
    ):
        # pylint: disable=too-many-locals
        super().__init__(**kwargs)

        assert input_shape == output_shape
        image_shape = input_shape

        self.image_shape = image_shape

        self.stem_down, self.token_features, output_size = \
            construct_downsample_stem(
                stem_features_list, stem_activ, stem_norm, stem_downsample,
                image_shape
            )

        self.N_h, self.N_w = output_size

        self.bottleneck = PixelwiseViT(
            features, n_heads, n_blocks, ffn_features, embed_features,
            activ, norm,
            image_shape = (self.token_features, self.N_h, self.N_w),
            rezero      = rezero
        )

        self.stem_up = construct_upsample_stem(
            stem_features_list, self.token_features, stem_activ, stem_norm,
            stem_upsample, image_shape
        )

    def forward(self, x):
        # x : (N, C, H, W)

        # z : (N, token_features, N_h, N_w)
        z = self.stem_down(x)
        z = self.bottleneck(z)

        # result : (N, C, H, W)
        result = self.stem_up(z)

        return result

