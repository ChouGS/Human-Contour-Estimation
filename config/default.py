from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN

_C = CN()

# Overall settings
_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.DATA_DIR = ''
_C.PERF_FNAME = ''
_C.GPUS = (0,)
_C.WORKERS = 4
_C.PRINT_FREQ = 20
_C.AUTO_RESUME = True

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# Dataset settings
_C.DATASET = CN()
_C.DATASET.ROOT = ''

# Model settings
_C.MODEL = CN()

_C.MODEL.NUM_NODES = 63
_C.MODEL.NODE_FEATURE_DIM = 128
_C.MODEL.IN_DIM = 48*64
_C.MODEL.IN_CHANNEL = 64
_C.MODEL.IMAGE_SIZE = (256, 192)
_C.MODEL.NUM_JOINTS = 63
_C.MODEL.GNN_TYPE = 'GCN'

# Training settings
_C.TRAIN = CN()

# Learning rate
_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [90, 110]
_C.TRAIN.LR = 0.001

# Optimizer
_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001
_C.TRAIN.NESTEROV = False
_C.TRAIN.GAMMA1 = 0.99
_C.TRAIN.GAMMA2 = 0.0

# Others
_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 140
_C.TRAIN.BATCH_SIZE_PER_GPU = 32
_C.TRAIN.SHUFFLE = True
_C.TRAIN.SIZE = 0

# Testing settings
_C.TEST = CN()
_C.TEST.BATCH_SIZE_PER_GPU = 32
_C.TEST.SIZE = 5941

def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)

    # cfg.MODEL.PRETRAINED = os.path.join(
    #     cfg.DATA_DIR, cfg.MODEL.PRETRAINED
    # )

    # if cfg.TEST.MODEL_FILE:
    #     cfg.TEST.MODEL_FILE = os.path.join(
    #         cfg.DATA_DIR, cfg.TEST.MODEL_FILE
    #     )

    cfg.freeze()

if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)

