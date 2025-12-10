from torch import nn

from uvcgan_s.torch.select        import get_activ_layer
from uvcgan_s.torch.layers.resnet import ResNetEncoder

class ResNetGen(nn.Module):

    def __init__(
        self, input_shape, output_shape, block_specs, activ, norm,
        rezero = True, activ_output = None
    ):
        # pylint: disable=too-many-arguments
        super().__init__()

        self._net = ResNetEncoder(
            input_shape, block_specs, activ, norm, rezero
        )

        self._output_shape  = self._net.output_shape
        self._input_shape   = input_shape

        assert tuple(output_shape) == self._output_shape, (
            f"Output shape {self._output_shape}"
            f" != desired shape {output_shape}"
        )

        self._out = get_activ_layer(activ_output)

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def output_shape(self):
        return self._output_shape

    def forward(self, x):
        y = self._net(x)
        return self._out(y)

