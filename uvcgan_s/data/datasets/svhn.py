from torchvision.datasets import SVHN

class SVHNDataset(SVHN):

    def __init__(
        self, path, split, transform, return_target = False, **kwargs
    ):
        # pylint: disable=too-many-arguments
        super().__init__(
            path,
            split     = split,
            transform = transform,
            download  = True,
            **kwargs
        )

        self._return_target = return_target

    def __getitem__(self, index):
        item = super().__getitem__(index)

        if self._return_target:
            return item

        return item[0]

