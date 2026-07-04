"""PURE_CLS baseline: entry point (NO CHANNEL).

Copy of ``tasks/classification/main_cls.py`` wired to the pure (no-channel)
train/eval copies. Part of the removable ``classification_pure`` baseline -- see
``tasks/classification_pure/MANIFEST.md``.

    python tasks/classification_pure/main_pure_cls.py --mode train \
      --config tasks/classification_pure/configs/Imagenet100_pure_cls.yaml
"""

import argparse
import os
import sys
from pathlib import Path

import torch.distributed as dist


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Pure-compression classification (no channel) for DefMambaJSCC")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "tasks" / "classification_pure" / "configs" / "Cifar100_pure_cls.yaml"),
        help="Pure classification task config yaml.",
    )
    parser.add_argument(
        "--model_config",
        default=str(REPO_ROOT / "configs" / "vssm" / "vssm_tiny_CIFAR10_multitask_defscan.yaml"),
        help="Shared backbone model config yaml (has USE_DEFSCAN=True, OUT_CHANS=32).",
    )
    parser.add_argument("--project_path", default=str(REPO_ROOT))
    parser.add_argument("--mode", default="train", choices=["train", "test"])
    parser.add_argument(
        "--stage",
        default=None,
        choices=["finetune_encoder", "from_scratch"],
        help="Optional override for CLS.STAGE.",
    )
    parser.add_argument("--pretrain_encoder", default=None, help="Optional override for CLS.PRETRAIN_ENCODER.")
    parser.add_argument("--pretrain_classifier", default=None, help="Optional override for CLS.PRETRAIN_CLASSIFIER.")
    parsed = parser.parse_args()
    parsed.project_path = os.path.abspath(parsed.project_path)
    parsed.model_config_path = os.path.abspath(parsed.model_config)
    parsed.train_config_path = os.path.abspath(parsed.config)
    return parsed


def _apply_overrides(config, args):
    if not any([args.stage, args.pretrain_encoder, args.pretrain_classifier]):
        return config
    cfg = config.clone()
    cfg.defrost()
    if args.stage:
        cfg.CLS.STAGE = args.stage
    if args.pretrain_encoder is not None:
        cfg.CLS.PRETRAIN_ENCODER = args.pretrain_encoder
    if args.pretrain_classifier is not None:
        cfg.CLS.PRETRAIN_CLASSIFIER = args.pretrain_classifier
    cfg.freeze()
    return cfg


def main(args):
    from configs.config import get_config
    from tasks.classification_pure.eval_pure_cls import test_classification
    from tasks.classification_pure.train_pure_cls import train_classification
    from utils.utils import seed_torch

    config = _apply_overrides(get_config(args), args)

    if args.mode == "train":
        seed_torch()
        train_classification(config)
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return
        seed_torch()
        test_classification(config)
    elif args.mode == "test":
        seed_torch()
        test_classification(config)


if __name__ == "__main__":
    main(parse_args())
