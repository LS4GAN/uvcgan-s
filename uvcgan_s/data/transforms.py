import torch
import torchvision
from torchvision import transforms

from uvcgan_s.torch.select import extract_name_kwargs

FromNumpy = lambda : torch.from_numpy

TRANSFORM_DICT = {
    'center-crop'            : transforms.CenterCrop,
    'color-jitter'           : transforms.ColorJitter,
    'random-crop'            : transforms.RandomCrop,
    'random-flip-vertical'   : transforms.RandomVerticalFlip,
    'random-flip-horizontal' : transforms.RandomHorizontalFlip,
    'random-rotation'        : transforms.RandomRotation,
    'random-resize-crop'     : transforms.RandomResizedCrop,
    'random-solarize'        : transforms.RandomSolarize,
    'random-invert'          : transforms.RandomInvert,
    'gaussian-blur'          : transforms.GaussianBlur,
    'resize'                 : transforms.Resize,
    'normalize'              : transforms.Normalize,
    'pad'                    : transforms.Pad,
    'grayscale'              : transforms.Grayscale,
    'to-tensor'              : transforms.ToTensor,
    'from-numpy'             : FromNumpy,
    'CenterCrop'             : transforms.CenterCrop,
    'ColorJitter'            : transforms.ColorJitter,
    'RandomCrop'             : transforms.RandomCrop,
    'RandomVerticalFlip'     : transforms.RandomVerticalFlip,
    'RandomHorizontalFlip'   : transforms.RandomHorizontalFlip,
    'RandomRotation'         : transforms.RandomRotation,
    'Resize'                 : transforms.Resize,
}

INTERPOLATION_DICT = {
    'nearest'  : transforms.InterpolationMode.NEAREST,
    'bilinear' : transforms.InterpolationMode.BILINEAR,
    'bicubic'  : transforms.InterpolationMode.BICUBIC,
    'lanczos'  : transforms.InterpolationMode.LANCZOS,
}

def parse_interpolation(kwargs):
    if 'interpolation' in kwargs:
        kwargs['interpolation'] = INTERPOLATION_DICT[kwargs['interpolation']]

def select_single_transform(transform):
    name, kwargs = extract_name_kwargs(transform)

    if name == 'random-apply':
        transform = select_transform_basic(kwargs.pop('transforms'))
        return transforms.RandomApply(transform, **kwargs)

    if name not in TRANSFORM_DICT:
        raise ValueError(f"Unknown transform: '{name}'")

    parse_interpolation(kwargs)

    return TRANSFORM_DICT[name](**kwargs)

def select_transform_basic(transform, compose = False):
    result = []

    if transform is not None:
        if not isinstance(transform, (list, tuple)):
            transform = [ transform, ]

        result = [
            select_single_transform(x) for x in transform if x != 'none'
        ]

    if compose:
        if len(result) == 1:
            return result[0]
        else:
            return torchvision.transforms.Compose(result)
    else:
        return result

def select_transform(transform, add_to_tensor = True):
    if transform == 'none':
        return None

    result = select_transform_basic(transform)

    # NOTE: this uglinness is for backward compat
    if add_to_tensor:
        if not isinstance(transform, (list, tuple)):
            transform = [ transform, ]

        need_transform = True

        if any(t == 'to-tensor' for t in transform):
            need_transform = False

        if any(t == 'from-numpy' for t in transform):
            need_transform = False

        if any(t == 'none' for t in transform):
            need_transform = False

        if need_transform:
            result.append(torchvision.transforms.ToTensor())

    return torchvision.transforms.Compose(result)

