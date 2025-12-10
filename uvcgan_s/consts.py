import os

CONFIG_NAME = 'config.json'
ROOT_DATA   = os.path.join(os.environ.get('UVCGAN_S_DATA',   'data'))
ROOT_OUTDIR = os.path.join(os.environ.get('UVCGAN_S_OUTDIR', 'outdir'))

SPLIT_TRAIN = 'train'
SPLIT_VAL   = 'val'
SPLIT_TEST  = 'test'

MERGE_PAIRED   = 'paired'
MERGE_UNPAIRED = 'unpaired'
MERGE_NONE     = 'none'

MODEL_STATE_TRAIN = 'train'
MODEL_STATE_EVAL  = 'eval'
