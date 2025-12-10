import os
import random

import torchvision.transforms.functional as TF

from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

from uvcgan_s.consts import SPLIT_TRAIN
from .image_domain_folder import ImageDomainFolder


class ToyMixBlurDataset(Dataset):
    # pylint: disable=too-many-instance-attributes

    def __init__(
        self, path,
        domain_a0        = 'cat',
        domain_a1        = 'dog',
        split            = SPLIT_TRAIN,
        alpha            = 0.5,
        alpha_range      = 0.0,
        blur_kernel_size = 5,
        seed             = 42,
        transform        = None,
        **kwargs
    ):
        # pylint: disable=too-many-arguments
        super().__init__(**kwargs)

        self._split       = split
        self._alpha       = alpha
        self._alpha_range = alpha_range
        self._blur_ks     = blur_kernel_size
        self._transform   = transform

        path_a0 = os.path.join(path, split, domain_a0)
        path_a1 = os.path.join(path, split, domain_a1)

        self._imgs_a0 = ImageDomainFolder.find_images_in_dir(path_a0)
        self._imgs_a1 = ImageDomainFolder.find_images_in_dir(path_a1)

        self._rng = random.Random(seed)

        self._pairing_indices = list(range(len(self._imgs_a1)))
        self._rng.shuffle(self._pairing_indices)

    def __len__(self):
        return len(self._imgs_a0)

    def _get_alpha(self):
        if self._split == SPLIT_TRAIN:
            offset = random.uniform(
                -self._alpha_range / 2, self._alpha_range / 2
            )
            return self._alpha + offset
        else:
            return self._alpha

    def _apply_blur(self, img):
        return TF.gaussian_blur(img, kernel_size=self._blur_ks)

    def __getitem__(self, index):

        if self._split == SPLIT_TRAIN:
            index_a1 = random.randint(0, len(self._imgs_a1) - 1)
        else:
            index_a1 = self._pairing_indices[
                index % len(self._pairing_indices)
            ]

        img_a0 = default_loader(self._imgs_a0[index])
        img_a1 = default_loader(self._imgs_a1[index_a1])

        if self._transform is not None:
            img_a0 = self._transform(img_a0)
            img_a1 = self._transform(img_a1)

        alpha = self._get_alpha()
        mixed = alpha * img_a0 + (1.0 - alpha) * img_a1
        mixed = self._apply_blur(mixed)

        return mixed

