# --------------------------------------------------------
# Modified by Mzero
# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------'

import os
import yaml
from yacs.config import CfgNode as CN


_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.TRAIN_BATCH = 4
_C.DATA.TEST_BATCH = 1
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = ''
# Dataset name
_C.DATA.DATASET = 'DIV2K' #'imagenet'
# Input image size
_C.DATA.IMG_SIZE = 256
# path
_C.DATA.train_data_dir= r"/mnt/wutong/datasets/DIV2K/DIV2K_train_HR"
_C.DATA.val_data_dir= r"/mnt/wutong/datasets/DIV2K/DIV2K_valid_HR"
_C.DATA.test_data_dir= r"/mnt/wutong/datasets/DIV2K/DIV2K_valid_HR"
_C.DATA.VAL_RATIO = 0.1
_C.DATA.VAL_SEED = 42
# Strong train-time augmentation (RandomResizedCrop + ColorJitter) for
# ImageFolder-style classification datasets. Off by default to preserve the
# original light augmentation; turn on to fight overfitting on small datasets.
_C.DATA.STRONG_AUG = False
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = 'bicubic'
# Use zipped dataset instead of folder dataset
# could be overwritten by command line argument
_C.DATA.ZIP_MODE = False
# Cache Data in Memory, could be overwritten by command line argument
_C.DATA.CACHE_MODE = 'part'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 8

# [SimMIM] Mask patch size for MaskGenerator
_C.DATA.MASK_PATCH_SIZE = 32
# [SimMIM] Mask ratio for MaskGenerator
_C.DATA.MASK_RATIO = 0.6
_C.DATA.REQUIRE_DISTRIBUTION = True
_C.DATA.RANGE = 5
_C.DATA.SCALE = 10000

#--------------------------------------
# channel setting
#-------------------------------------  
_C.CHANNEL=CN()
_C.CHANNEL.TYPE='awgn'
_C.CHANNEL.SNR=[20]
_C.CHANNEL.ADAPTIVE='CA'
# Mismatched-CSI ablation: the channel always uses the true (swept/sampled) SNR,
# but when BLIND_MODEL is True the model (encoder + head) is fed a fixed MODEL_SNR
# instead, so it cannot adapt to the channel. This makes accuracy rise with SNR
# (vs the flat curve when the model knows the true SNR via CSI-ReST).
_C.CHANNEL.BLIND_MODEL = False
_C.CHANNEL.MODEL_SNR = 10.0
# Compact-code transmission (classification): global-average-pool the encoder
# feature to a per-channel code BEFORE the channel, so the noise hits the compact
# code directly instead of a large spatial map (whose pooling would average the
# noise away). This is what makes classification accuracy depend on SNR.
_C.CHANNEL.COMPACT_CODE = False
# Single-SNR training: if set (e.g. 20), every training batch uses this fixed SNR
# instead of sampling from CHANNEL.SNR. Evaluation still sweeps CHANNEL.SNR. Trains
# a model that never saw noise -> test accuracy degrades as SNR drops (rising curve).
_C.CHANNEL.TRAIN_SNR = None
# Separate evaluation SNR grid: if set (a list), validation AND test sweep these
# SNRs instead of CHANNEL.SNR, while training still samples from CHANNEL.SNR. Use
# to train on a dense SNR grid but report on a coarser one. None = reuse CHANNEL.SNR.
_C.CHANNEL.EVAL_SNR = None
# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = 'vssm'
# Model name
_C.MODEL.NAME = 'vssm_tiny'
# Pretrained weight from checkpoint, could be imagenet22k pretrained weight
# could be overwritten by command line argument
_C.MODEL.PRETRAINED = ''
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.1
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1

# MMpretrain models for test
_C.MODEL.MMCKPT = False
_C.MODEL.disc_num_layers=3
_C.MODEL.use_actnorm=False
# VSSM parameters
_C.MODEL.VSSM = CN()

_C.MODEL.VSSM.PATCH_SIZE = 2
_C.MODEL.VSSM.IN_CHANS = 3
_C.MODEL.VSSM.OUT_CHANS = 36
_C.MODEL.VSSM.DEPTHS = [2, 2, 9, 2]
_C.MODEL.VSSM.EMBED_DIM = [128,192,256,780]
_C.MODEL.VSSM.SSM_D_STATE = 16
_C.MODEL.VSSM.SSM_RATIO = 2.0
_C.MODEL.VSSM.SSM_RANK_RATIO = 2.0
_C.MODEL.VSSM.SSM_DT_RANK = "auto"
_C.MODEL.VSSM.SSM_ACT_LAYER = "gelu"
_C.MODEL.VSSM.SSM_CONV = 3
_C.MODEL.VSSM.SSM_CONV_BIAS = True
_C.MODEL.VSSM.SSM_DROP_RATE = 0.0
_C.MODEL.VSSM.SSM_SIMPLE_INIT = False
_C.MODEL.VSSM.SSM_FORWARDTYPE = "v2"
_C.MODEL.VSSM.MLP_RATIO = 4.0
_C.MODEL.VSSM.MLP_ACT_LAYER = "gelu"
_C.MODEL.VSSM.MLP_DROP_RATE = 0.0
_C.MODEL.VSSM.PATCH_NORM = True
_C.MODEL.VSSM.NORM_LAYER = "ln"
_C.MODEL.VSSM.DOWNSAMPLE = "v1"
_C.MODEL.VSSM.PATCHEMBED = "v1"
_C.MODEL.VSSM.SCAN = "cross"
_C.MODEL.VSSM.PE = "no"
_C.MODEL.VSSM.SCAN_NUMBER = 4
_C.MODEL.VSSM.Extent = 'conv'
_C.MODEL.VSSM.channel_input = 'conv'
_C.MODEL.VSSM.USE_DEFSCAN = False
_C.MODEL.VSSM.DEFSCAN_DEF_INIT = 0.05
_C.MODEL.VSSM.TRI_MERGE_LR_MULT = 3.0
_C.MODEL.VSSM.TRI_MERGE_DEBUG = False
# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 1
_C.TRAIN.SAVE_FRE=1 
_C.TRAIN.EVAL_FRE=10
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 1e-4
_C.TRAIN.BASE_LR = 1e-4 #5e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 1
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
# warmup_prefix used in CosineLRScheduler
_C.TRAIN.LR_SCHEDULER.WARMUP_PREFIX = True
# [SimMIM] Gamma / Multi steps value, used in MultiStepLRScheduler
_C.TRAIN.LR_SCHEDULER.GAMMA = 0.1
_C.TRAIN.LR_SCHEDULER.MULTISTEPS = []

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# [SimMIM] Layer decay for fine-tuning
_C.TRAIN.LAYER_DECAY = 1.0

# loss function
_C.TRAIN.LOSS='MSE'
_C.TRAIN.DATA_PARALLEL=False
_C.TRAIN.EVAL_MATRIX='PSNR'
_C.TRAIN.GAN_LOSS=False
_C.TRAIN.DIS_WEIGHT=0.5
_C.TRAIN.START_EPOCH=10
_C.TRAIN.ENCODER_PATH='/mnt/wutong/MambaJSCCcheckpoints/Journal/encoder'
_C.TRAIN.DECODER_PATH='/mnt/wutong/MambaJSCCcheckpoints/Journal/decoder'
_C.TRAIN.LOG_PATH=''
# -----------------------------------------------------------------------------
# Classification task settings (independent downstream task on the JSCC latent)
# -----------------------------------------------------------------------------
_C.CLS = CN()
# Training stage: "from_scratch" | "finetune_encoder"
_C.CLS.STAGE = "from_scratch"
_C.CLS.NUM_CLASSES = 10
# Latent classifier head
_C.CLS.SNR_EMBED_DIM = 32
_C.CLS.HEAD_HIDDEN_DIM = 256
_C.CLS.HEAD_DROPOUT = 0.0
_C.CLS.HEAD_LR = 1e-3
_C.CLS.HEAD_WEIGHT_DECAY = 1e-4
_C.CLS.USE_SNR_EMBED = True
# CrossEntropy label smoothing (0.0 = off). A cheap regularizer for small datasets.
_C.CLS.LABEL_SMOOTHING = 0.0
# Mixup alpha for Beta(alpha, alpha) input mixing (0.0 = off). Strong regularizer
# against memorization; typical values 0.1-0.4.
_C.CLS.MIXUP_ALPHA = 0.0
# Checkpoint output directories
_C.CLS.ENCODER_PATH = ""
_C.CLS.CLASSIFIER_PATH = ""
# Optional warm-start checkpoints
_C.CLS.PRETRAIN_ENCODER = ""
_C.CLS.PRETRAIN_CLASSIFIER = ""
# MoE
_C.TRAIN.MOE = CN()
# Only save model on master device
_C.TRAIN.MOE.SAVE_MASTER = False
# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Color jitter factor
_C.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
_C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
# Random erase prob
_C.AUG.REPROB = 0.25
# Random erase mode
_C.AUG.REMODE = 'pixel'
# Random erase count
_C.AUG.RECOUNT = 1
# Mixup alpha, mixup enabled if > 0
_C.AUG.MIXUP = 0.8
# Cutmix alpha, cutmix enabled if > 0
_C.AUG.CUTMIX = 1.0
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
_C.AUG.CUTMIX_MINMAX = None
# Probability of performing mixup or cutmix when either/both is enabled
_C.AUG.MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
_C.AUG.MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.AUG.MIXUP_MODE = 'batch'

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use center crop when testing
_C.TEST.CROP = True
# Whether to use SequentialSampler as validation sampler
_C.TEST.SEQUENTIAL = False
_C.TEST.SHUFFLE = False

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# [SimMIM] Whether to enable pytorch amp, overwritten by command line argument
_C.ENABLE_AMP = False

# Enable Pytorch automatic mixed precision (amp).
_C.AMP_ENABLE = True
# [Deprecated] Mixed precision opt level of apex, if O0, no apex amp is used ('O0', 'O1', 'O2')
_C.AMP_OPT_LEVEL = ''
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''
# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
# Frequency to save checkpoint
_C.SAVE_FREQ = 1
# Frequency to logging info
_C.PRINT_FREQ = 10
# Fixed random seed
_C.SEED = 0
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
# for acceleration
_C.FUSED_LAYERNORM = False


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
            print(1)
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):

    _update_config_from_file(config, args.model_config_path)
    _update_config_from_file(config, args.train_config_path)

    #print(config.MODEL.VSSM.EMBED_DIM, config.MODEL.VSSM.DEPTHS)
def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config
