from uvcgan_s.base.networks    import select_base_generator
from uvcgan_s.models.funcs     import default_model_init

from .vitunet   import ViTUNetGenerator
from .vitmodnet import ViTModNetGenerator, CViTModNetGenerator
from .resnet    import ResNetGen
from .dcgan     import DCGANGenerator

def select_generator(name, **kwargs):
    # pylint: disable=too-many-return-statements

    if name == 'vit-unet':
        return ViTUNetGenerator(**kwargs)

    if name == 'vit-modnet':
        return ViTModNetGenerator(**kwargs)

    if name == 'cvit-modnet':
        return CViTModNetGenerator(**kwargs)

    if name == 'resnet':
        return ResNetGen(**kwargs)

    if name == 'dcgan':
        return DCGANGenerator(**kwargs)

    input_shape  = kwargs.pop('input_shape')
    output_shape = kwargs.pop('output_shape')

    assert input_shape == output_shape
    return select_base_generator(name, image_shape = input_shape, **kwargs)

def construct_generator(model_config, input_shape, output_shape, device):
    model = select_generator(
        model_config.model,
        input_shape  = input_shape,
        output_shape = output_shape,
        **model_config.model_args
    )

    return default_model_init(model, model_config, device)

