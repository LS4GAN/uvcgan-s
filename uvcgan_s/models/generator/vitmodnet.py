# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes

from torch import nn

from uvcgan_s.torch.layers.transformer import (
    ExtendedPixelwiseViT, CExtPixelwiseViT
)
from uvcgan_s.torch.layers.modnet      import ModNet
from uvcgan_s.torch.select             import get_activ_layer

class ViTModNetGenerator(nn.Module):

    def __init__(
        self, features, n_heads, n_blocks, ffn_features, embed_features,
        activ, norm, input_shape, output_shape, modnet_features_list,
        modnet_activ,
        modnet_norm       = None,
        modnet_downsample = 'conv',
        modnet_upsample   = 'upsample-conv',
        modnet_rezero     = False,
        modnet_demod      = True,
        rezero            = True,
        activ_output      = None,
        style_rezero      = True,
        style_bias        = True,
        n_ext             = 1,
        **kwargs
    ):
        # pylint: disable = too-many-locals
        super().__init__(**kwargs)

        mod_features = features * n_ext

        self.net = ModNet(
            modnet_features_list, modnet_activ, modnet_norm,
            input_shape, output_shape,
            modnet_downsample, modnet_upsample, mod_features, modnet_rezero,
            modnet_demod, style_rezero, style_bias, return_mod = False
        )

        bottleneck = ExtendedPixelwiseViT(
            features, n_heads, n_blocks, ffn_features, embed_features,
            activ, norm,
            image_shape = self.net.get_inner_shape(),
            rezero      = rezero,
            n_ext       = n_ext,
        )

        self.net.set_bottleneck(bottleneck)

        self.output = get_activ_layer(activ_output)

    def forward(self, x):
        # x : (N, C, H, W)
        result = self.net(x)
        return self.output(result)

class CViTModNetGenerator(nn.Module):

    def __init__(
        self, features, n_heads, n_blocks, ffn_features, embed_features,
        activ, norm, input_shape, output_shape, modnet_features_list,
        modnet_activ,
        modnet_norm       = None,
        modnet_downsample = 'conv',
        modnet_upsample   = 'upsample-conv',
        modnet_rezero     = False,
        modnet_demod      = True,
        rezero            = True,
        activ_output      = None,
        style_rezero      = True,
        style_bias        = True,
        n_control_in      = 100,
        n_control_out     = 100,
        return_feedback   = True,
        **kwargs
    ):
        # pylint: disable = too-many-locals
        super().__init__(**kwargs)

        self.control_in      = nn.Linear(n_control_in, features)
        self.control_out     = nn.Linear(features, n_control_out)
        self.return_feedback = return_feedback
        mod_features         = features

        self.net = ModNet(
            modnet_features_list, modnet_activ, modnet_norm,
            input_shape, output_shape,
            modnet_downsample, modnet_upsample, mod_features, modnet_rezero,
            modnet_demod, style_rezero, style_bias, return_mod = True
        )

        bottleneck = CExtPixelwiseViT(
            features, n_heads, n_blocks, ffn_features, embed_features,
            activ, norm,
            image_shape = self.net.get_inner_shape(),
            rezero      = rezero,
        )

        self.net.set_bottleneck(bottleneck)

        self.output = get_activ_layer(activ_output)

    def forward(self, x, control):
        # x : (N, C, H, W)

        mod = self.control_in(control)

        result, mod = self.net(x, mod)

        result = self.output(result)

        if not self.return_feedback:
            return result

        feedback = self.control_out(mod)
        return result, feedback

