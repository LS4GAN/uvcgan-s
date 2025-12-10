import torch
from torch import nn

from uvcgan_s.torch.select        import get_activ_layer
from uvcgan_s.torch.layers.resnet import ResNetEncoder

class ResNetDisc(nn.Module):

    def __init__(
        self, image_shape, block_specs, activ, norm,
        rezero = True, activ_output = None, reduce_output_channels = True
    ):
        # pylint: disable=too-many-arguments
        super().__init__()

        self._net = ResNetEncoder(
            image_shape, block_specs, activ, norm, rezero
        )

        self._output_shape  = self._net.output_shape
        self._input_shape   = image_shape

        output_layers = []

        if reduce_output_channels:
            output_layers.append(
                nn.Conv2d(self._output_shape[0], 1, kernel_size = 1)
            )
            self._output_shape = (1, *self._output_shape[1:])

        if activ_output is not None:
            output_layers.append(get_activ_layer(activ_output))

        self._out = nn.Sequential(*output_layers)

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def output_shape(self):
        return self._output_shape

    def forward(self, x):
        y = self._net(x)
        return self._out(y)

class ResNetFFTDisc(nn.Module):

    def __init__(
        self, image_shape, block_specs, activ, norm,
        rezero = True, activ_output = None, reduce_output_channels = True
    ):
        # pylint: disable=too-many-arguments
        super().__init__()

        # Doubling channels for FFT amplitudes
        input_shape = (2 * image_shape[0], *image_shape[1:])

        self._net = ResNetEncoder(
            input_shape, block_specs, activ, norm, rezero
        )

        self._output_shape  = self._net.output_shape
        self._input_shape   = image_shape

        output_layers = []

        if reduce_output_channels:
            output_layers.append(
                nn.Conv2d(self._output_shape[0], 1, kernel_size = 1)
            )
            self._output_shape = (1, *self._output_shape[1:])

        if activ_output is not None:
            output_layers.append(get_activ_layer(activ_output))

        self._out = nn.Sequential(*output_layers)

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def output_shape(self):
        return self._output_shape

    def forward(self, x):
        # x     : (N, C, H, W)
        # x_fft : (N, C, H, W)
        x_fft = torch.fft.fft2(x).abs()

        # z : (N, 2C, H, W)
        z = torch.cat([ x, x_fft ], dim = 1)

        y = self._net(z)

        return self._out(y)

