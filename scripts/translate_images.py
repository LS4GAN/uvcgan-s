#!/usr/bin/env python

import argparse
import collections
import os

import tqdm
import numpy as np
from PIL import Image

from uvcgan_s.consts import MERGE_NONE
from uvcgan_s.data   import construct_data_loaders
from uvcgan_s.eval.funcs import (
    start_model_eval, tensor_to_image, slice_data_loader, get_eval_savedir
)
from uvcgan_s.utils.parsers import add_standard_eval_parsers

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Translate images')
    add_standard_eval_parsers(parser)

    parser.add_argument(
        '--domain',
        dest    = 'domain',
        type    = int,
        default = 0,
        help    = "dataset domain",
    )

    parser.add_argument(
        '--label',
        dest    = 'label',
        type    = str,
        default = None,
        help    = "evaluation label",
    )

    parser.add_argument(
        '--format',
        dest     = 'format',
        choices  = [ 'image', 'ndarray' ],
        help     = 'data format',
        required = True,
    )

    parser.add_argument(
        '--ext',
        dest     = 'ext',
        default  = [ 'png', ],
        help     = 'image save format',
        nargs    = '+',
    )

    return parser.parse_args()

def save_image(image, root, index, ext):
    image = np.round(255 * np.clip(image, 0, 1)).astype(np.uint8)
    image = Image.fromarray(np.squeeze(image))

    path = os.path.join(root, f'sample_{index}')
    for e in ext:
        image.save(path + '.' + e)

def save_np_array(arr, root, index):
    path = os.path.join(root, f'sample_{index}.npz')
    np.savez_compressed(path, np.squeeze(arr))

def save_image_array(image, savedir, index, ext):
    if image.ndim not in [ 3 ]:
        save_np_array(image, savedir, index)

    if image.shape[2] in [ 1, 3 ]:
        save_image(image, savedir, index, ext)

    else:
        save_np_array(image, savedir, index)

def save_data(model, savedir, sample_counter, fmt, ext):
    for (name, torch_image) in model.get_images().items():
        if torch_image is None:
            continue

        if torch_image.ndim < 4:
            continue

        root = os.path.join(savedir, name)
        os.makedirs(root, exist_ok = True)

        for index in range(torch_image.shape[0]):
            sample_index = sample_counter[name]
            image = tensor_to_image(torch_image[index])

            if fmt == 'ndarray':
                save_np_array(image, root, sample_index)
            else:
                save_image_array(image, root, sample_index, ext)

            sample_counter[name] += 1

def translate_images(
    model, data_it, n_eval, batch_size, domain, fmt, ext, savedir
):
    # pylint: disable=too-many-arguments
    sample_counter = collections.defaultdict(int)
    data_it, steps = slice_data_loader(data_it, batch_size, n_eval)

    for batch in tqdm.tqdm(
        data_it, desc = f'Translating {domain}', total = steps
    ):
        model.set_input(batch, domain = domain)
        model.forward_nograd()

        save_data(model, savedir, sample_counter, fmt, ext)

def main():
    cmdargs = parse_cmdargs()

    args, model, evaldir = start_model_eval(
        cmdargs.model, cmdargs.epoch, cmdargs.model_state,
        merge_type = MERGE_NONE,
        batch_size = cmdargs.batch_size
    )

    args.config.data.datasets = [ args.config.data.datasets[cmdargs.domain], ]

    dl = construct_data_loaders(
        args.config.data, args.config.batch_size, split = cmdargs.split
    )

    savedir = get_eval_savedir(
        evaldir, f'translated({cmdargs.label})_domain({cmdargs.domain})',
        cmdargs.model_state, cmdargs.split,
    )

    translate_images(
        model, dl, cmdargs.n_eval, args.batch_size,
        cmdargs.domain, cmdargs.format, cmdargs.ext,
        savedir
    )

if __name__ == '__main__':
    main()

