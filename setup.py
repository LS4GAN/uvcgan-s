#!/usr/bin/env python

import setuptools

setuptools.setup(
    name             = 'uvcgan_s',
    version          = '0.1.0',
    author           = 'The LS4GAN Project Developers',
    author_email     = 'dtorbunov@bnl.gov',
    classifiers      = [
        'Programming Language :: Python :: 3 :: Only',
    ],
    description      = "Reference Implementation of Stratified CycleGAN",
    packages         = setuptools.find_packages(
        include = [ 'uvcgan_s', 'uvcgan_s.*' ]
    ),
    install_requires = [
        'numpy', 'pandas', 'tqdm', 'Pillow', 'h5py', 'einops'
    ],
)

