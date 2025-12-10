import itertools
import os
import torch

from uvcgan_s import ROOT_OUTDIR, train

#torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

DATA_PATH = 'afhq_resized_lanczos'
LR        = 1e-4
GP_LAMBDA = 1.0

args_dict = {
    'batch_size' : 1,
    'data' : {
        'datasets' : [
            # noise
            {
                'dataset' : {

                    'name'   : 'image-domain-hierarchy',
                    'domain' : 'cat',
                    'path'   : DATA_PATH,
                },
                'shape'           : (3, 256, 256),
                'transform_train' : None,
                'transform_test'  : None,
            },
            # signal
            {
                'dataset' : {
                    'name'   : 'image-domain-hierarchy',
                    'domain' : 'dog',
                    'path'   : DATA_PATH,
                },
                'shape'           : (3, 256, 256),
                'transform_train' : None,
                'transform_test'  : None,
            },
            # embedded signal
            {
                'dataset' : {
                    'name'             : 'toy-mix-blur',
                    'domain_a0'        : 'cat',
                    'domain_a1'        : 'dog',
                    'alpha'            : 0.5,
                    'alpha_range'      : 0.0,
                    'blur_kernel_size' : 5,
                    'seed'             : 0,
                    'path'             : DATA_PATH,
                },
                'shape'           : (3, 256, 256),
                'transform_train' : None,
                'transform_test'  : None,
            },
        ],
        'merge_type' : 'unpaired',
        'workers'    : 1,
    },
    'epochs'        : 100,
    'discriminator' : {
        'model'      : 'n_layers',
        'model_args' : {
            'n_layers' : 5,
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
        'spectr_norm' : False,
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
            'modnet_features_list' : [48, 96, 192, 384],
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
        'lambda_cyc_a1'   : 10,
        'lambda_cyc_b'    : 10,
        'lambda_idt_aa'   : 0.5,
        'lambda_idt_bb'   : 0.5,
        'ema_momentum'    : 0.999,
        'data_norm'       : None,
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
    'scheduler' : None,
    'loss'             : 'hinge',
    'steps_per_epoch'  : 1000,
    'transfer'         : None,
    'gradient_penalty' : {
        'center'    : 0,
        'lambda_gp' : GP_LAMBDA,
        'mix_type'  : 'real-fake',
        'reduction' : 'mean',
    },
# args
    'label'  : 'cat_dog_sub',
    'outdir' : os.path.join(ROOT_OUTDIR, 'toy_mix_blur', 'uvcgan-s'),
    'log_level'  : 'DEBUG',
    'checkpoint' : 10,
}

train(args_dict)

