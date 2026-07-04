"""PURE_CLS baseline: evaluation entry points (NO CHANNEL).

Tear-off note: this file is a copy of ``tasks/classification/eval_cls.py`` with the
channel removed. The pipeline is ``image -> encoder -> classifier`` (no channel:
``z_hat = feature``; SNR, if the config's structure uses it, is a fixed constant).
It is part of the removable ``classification_pure`` baseline
-- see ``tasks/classification_pure/MANIFEST.md``. It imports shared modules but
modifies none of them.
"""

import os
import time

import torch
from tqdm import tqdm

from data.datasets import get_loader
from tasks.classification.task_head import LatentClassifierHead
from utils.engine import checkpoint_tag, get_log_dir
from utils.utils import save_model, seed_torch


# SNR value fed to the encoder and head. There is NO channel in this directory
# (z_hat = feature); this constant only sets what SNR the model believes it
# operates at, which matters for two tiers sharing this loop:
#   tier (1) pure  (ADAPTIVE='no',  USE_SNR_EMBED=False): value is ignored --
#            the non-'ssm' scan drops snr (models/vmamba.py) and the head skips it.
#   tier (2) snr20 (ADAPTIVE='ssm', USE_SNR_EMBED=True):  CSI-ReST and the SNR
#            embed consume it, matching the comm version fed a constant 20 dB
#            but with no channel noise.
_FIXED_SNR = 20.0


def classification_checkpoint_name(config):
    """Run identity for classification checkpoints (encoder + classifier share it)."""
    stage = getattr(config.CLS, "STAGE", "from_scratch")
    return f"CLS_{stage}_" + checkpoint_tag(config)


def classifier_save_dir(config):
    path = getattr(config.CLS, "CLASSIFIER_PATH", "")
    if path:
        os.makedirs(path, exist_ok=True)
        return path
    log_dir = get_log_dir(config)
    path = os.path.join(os.path.dirname(os.path.normpath(log_dir)), "classifier")
    os.makedirs(path, exist_ok=True)
    return path + os.sep


def encoder_save_dir(config):
    path = getattr(config.CLS, "ENCODER_PATH", "")
    if path:
        os.makedirs(path, exist_ok=True)
        return path
    log_dir = get_log_dir(config)
    path = os.path.join(os.path.dirname(os.path.normpath(log_dir)), "cls_encoder")
    os.makedirs(path, exist_ok=True)
    return path + os.sep


def build_latent_classifier(config):
    snr_list = getattr(config.CHANNEL, "SNR", [20])
    snr_max = max(snr_list) if isinstance(snr_list, (list, tuple)) else float(snr_list)
    return LatentClassifierHead(
        latent_dim=config.MODEL.VSSM.OUT_CHANS,
        num_classes=config.CLS.NUM_CLASSES,
        hidden_dim=config.CLS.HEAD_HIDDEN_DIM,
        snr_embed_dim=config.CLS.SNR_EMBED_DIM,
        use_snr=config.CLS.USE_SNR_EMBED,  # False in the pure config -> SNR ignored
        snr_max=snr_max,
        dropout=config.CLS.HEAD_DROPOUT,
    )


def save_classification_models(config, encoder, classifier):
    name = classification_checkpoint_name(config) + ".pt"
    save_model(encoder, os.path.join(encoder_save_dir(config), name))
    save_model(classifier, os.path.join(classifier_save_dir(config), name))


def _prepare_input(config, input_image):
    if config.DATA.DATASET in ("CIFAR10", "CIFAR100", "cifar100", "CIFAR-100"):
        return torch.nn.functional.interpolate(
            input_image,
            (config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
            mode="bilinear",
            align_corners=False,
        )
    return input_image


@torch.no_grad()
def evaluate_classification(
    config,
    encoder,
    classifier,
    test_loader=None,
    criterion_cls=None,
    save_curves=True,
    prefix="pure",
):
    """Single-pass accuracy with NO channel. Returns ``([acc], mean_loss)`` so the
    train loop's ``sum(curve)/len(curve)`` bookkeeping stays unchanged.

    ``save_curves``/``prefix`` are accepted for call-site compatibility with the
    comm version but unused (there is no SNR axis to sweep in pure compression).
    """
    if test_loader is None:
        _, test_loader = get_loader(config)
    device = next(encoder.parameters()).device
    encoder.eval()
    classifier.eval()

    correct = 0
    total = 0
    loss_sum = 0.0
    n_batch = 0
    all_time = 0.0
    seed_torch()
    with tqdm(test_loader, dynamic_ncols=False) as tqdm_data:
        for input_image, labels in tqdm_data:
            if not torch.is_tensor(labels):
                raise ValueError("Classification requires class labels.")
            input_image = input_image.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            input_image = _prepare_input(config, input_image)

            start = time.time()
            feature = encoder(input_image, _FIXED_SNR)
            z_hat = feature  # no channel: encoder feature goes straight to the head
            logits = classifier(z_hat, _FIXED_SNR)
            all_time += time.time() - start

            pred = logits.argmax(dim=1)
            batch_correct = (pred == labels).sum().item()
            batch_total = labels.numel()
            acc_batch = batch_correct / max(batch_total, 1)
            if criterion_cls is not None:
                loss_sum += criterion_cls(logits, labels).item()

            correct += batch_correct
            total += batch_total
            n_batch += 1
            tqdm_data.set_postfix({"Acc": acc_batch})

    acc = correct / max(total, 1)
    mean_loss = (loss_sum / n_batch) if criterion_cls is not None else None
    print("Pure-compression accuracy:", acc)
    if mean_loss is not None:
        print("Cls loss:", mean_loss)
    return [acc], mean_loss


@torch.no_grad()
def test_classification(config):
    _, test_loader = get_loader(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = classification_checkpoint_name(config) + ".pt"
    encoder = torch.load(
        os.path.join(encoder_save_dir(config), name),
        weights_only=False,
        map_location=device,
    ).to(device)
    classifier = torch.load(
        os.path.join(classifier_save_dir(config), name),
        weights_only=False,
        map_location=device,
    ).to(device)
    evaluate_classification(config, encoder, classifier, test_loader=test_loader)
