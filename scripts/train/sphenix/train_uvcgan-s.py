import itertools
import os
import torch

from uvcgan_s import ROOT_OUTDIR, train

#torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

DATA_PATH = 'sphenix/2025-06-05_jet_bkg_sub'
LR        = 5e-5
GP_LAMBDA = 0.01

DISC_BLOCKS = [
    # (1, 24, 64)
    (
        'stem',
        { 'kernel_size' : 3, 'padding' : 1, 'stride' : 1, 'features' : 64 }
    ),
    # (64, 24, 64)
    ('resnet', 3),
    (
        'stem',
        { 'kernel_size' : 2, 'padding' : 0, 'stride' : 2, 'features' : 128 }
    ),
    # (128, 12, 32)
    ('resnet', 3),
    (
        'stem',
        { 'kernel_size' : 2, 'padding' : 0, 'stride' : 2, 'features' : 256 }
    ),
    # (256, 6, 16)
    ('resnet', 3),
    (
        'stem',
        { 'kernel_size' : 2, 'padding' : 0, 'stride' : 2, 'features' : 512 }
    ),
    # (512, 3, 8)
    ('resnet', 3),
]

args_dict = {
    'batch_size' : 4,
    'data' : {
        'datasets' : [
            # noise
            {
                'dataset' : {
                    'name'   : 'h5array-domain-hierarchy',
                    'path'   : DATA_PATH,
                    'domain' : 'background',
                },
                'shape'           : (1, 24, 64),
                'transform_train' : None,
                'transform_test'  : None,
            },
            # signal
            {
                'dataset' : {
                    'name'   : 'h5array-domain-hierarchy',
                    'path'   : DATA_PATH,
                    'domain' : 'signal',
                },
                'shape'           : (1, 24, 64),
                'transform_train' : None,
                'transform_test'  : None,
            },
            # embedded signal
            {
                'dataset' : {
                    'name'   : 'h5array-domain-hierarchy',
                    'path'   : DATA_PATH,
                    'domain' : 'embed',
                },
                'shape'           : (1, 24, 64),
                'transform_train' : None,
                'transform_test'  : None,
            },
        ],
        'merge_type' : 'unpaired',
        'workers'    : 1,
    },
    'epochs'        : 400,
    'discriminator' : {
        'model'      : 'resnet',
        'model_args' : {
            'block_specs' : DISC_BLOCKS,
            'norm'        : 'batch',
            'activ'       : 'leakyrelu',
            'rezero'      : True,
            'reduce_output_channels' : True,
        },
        'optimizer' : {
            'name'  : 'Adam',
            'lr'    : LR,
            'betas' : (0.5, 0.99),
        },
        'weight_init' : {
            'name'      : 'normal',
            'init_gain' : 0.02,
        },
        'spectr_norm' : True,
    },
    'generator' : {
        'model' : 'vit-modnet',
        'model_args' : {
            'features'             : 384,
            'n_heads'              : 6,
            'n_blocks'             : 12,
            'ffn_features'         : 1536,
            'embed_features'       : 384,
            'activ'                : 'gelu',
            'norm'                 : 'layer',
            'modnet_features_list' : [96, 192, 384],
            'modnet_activ'         : 'leakyrelu',
            'modnet_norm'          : None,
            'modnet_downsample'    : 'conv',
            'modnet_upsample'      : 'upsample-conv',
            'modnet_rezero'        : False,
            'rezero'               : True,
            'activ_output'         : None,
            'style_rezero'         : True,
            'style_bias'           : True,
            'n_ext'                : 1,
        },
        'optimizer'  : {
            'name'  : 'Adam',
            'lr'    : LR,
            'betas' : (0.5, 0.99),
        },
        'weight_init' : {
            'name' : 'kaiming',
        },
    },
    'model' : 'uvcgan-s',
    'model_args' : {
        'lambda_adv_a0'   : 1,
        'lambda_adv_a1'   : 1,
        'lambda_adv_b'    : 1,
        'lambda_cyc_a0'   : 10,
        'lambda_cyc_a1'   : 100,
        'lambda_cyc_b'    : 10,
        'lambda_idt_aa'   : 0.5,
        'lambda_idt_bb'   : 0.5,
        'ema_momentum'    : 0.9999,
        'data_norm'       : {
            'name' : 'log',
            'bias' : 0.1,
        },
        'gp_cache_period' : 0,
        'grad_clip'       : { 'norm' : 0.5 },
        'norm_loss_a0'    : False,
        'norm_loss_a1'    : False,
        'norm_loss_b'     : False,
        'norm_disc_a0'    : False,
        'norm_disc_a1'    : False,
        'norm_disc_b'     : False,
        'head_queue_size' : 0,
        'head_config'     : 'idt'
    },
    'seed'  : 0,
    'scheduler' : {
        'name' : 'linear-v2',
        'start_factor' : 0.01,
        'end_factor'   : 1.0,
        'total_iters'  : 32,
    },
    'loss'             : 'hinge',
    'steps_per_epoch'  : 2000,
    'transfer'         : None,
    'gradient_penalty' : {
        'center'    : 0,
        'lambda_gp' : GP_LAMBDA,
        'mix_type'  : 'real-fake',
        'reduction' : 'mean',
    },
# args
    'label'  : 'sgn_bkg_sub',
    'outdir' : os.path.join(ROOT_OUTDIR, 'sphenix', 'uvcgan-s'),
    'log_level'  : 'DEBUG',
    'checkpoint' : 50,
}

train(args_dict)

