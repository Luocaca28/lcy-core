'''
@author: Tong Wu
@contact: wu_tong@sjtu.edu.cn
'''


import os
import time
import json
import random
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.utils import accuracy, AverageMeter

from configs.config import get_config


from utils.utils import seed_torch

from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count

from timm.utils import ModelEma as ModelEma
from run.train import train_MambaJSCC
from run.eval import test_MambaJSCC

from utils.utils import GPUManager

if os.environ.get("CUDA_VISIBLE_DEVICES"):
    print("Using CUDA_VISIBLE_DEVICES={}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
else:
    gm=GPUManager()
    device_idx=gm.auto_choice(mode=3)
    os.environ["CUDA_VISIBLE_DEVICES"] =  str(device_idx)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", default="DIV2K")
    parser.add_argument("--project_path", default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--mode", default="test", choices=["train", "test"])
    parsed = parser.parse_args()
    project_path = os.path.abspath(parsed.project_path)
    parsed.model_config_path = os.path.join(
        project_path, "configs", "vssm", f"vssm_tiny_{parsed.config_name}.yaml"
    )
    parsed.train_config_path = os.path.join(
        project_path, "configs", "train", f"vssm_tiny_{parsed.config_name}.yaml"
    )
    return parsed
    



def main(args):

    config = get_config(args)

    if args.mode=='train': 

        seed_torch()
        train_MambaJSCC(config) 
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return
        seed_torch()
        test_MambaJSCC(config)
        
    elif args.mode == 'test':

        seed_torch()
        test_MambaJSCC(config)

if __name__ == "__main__":
    main(parse_args())
