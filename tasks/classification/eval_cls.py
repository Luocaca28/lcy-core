"""Classification task: evaluation entry points.

Evaluates an encoder + latent classifier across the SNR sweep and reports
accuracy. Shared plumbing (log dir, channel forward, run identity) comes from
``utils.engine``; this task never imports from ``run``.
"""

import csv
import os
import time

import torch
from tqdm import tqdm

from data.datasets import get_loader
from models.channel import Channel
from tasks.classification.task_head import LatentClassifierHead
from utils.engine import apply_channel, apply_channel_compact, checkpoint_tag, get_log_dir
from utils.utils import save_model, seed_torch


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
        use_snr=config.CLS.USE_SNR_EMBED,
        snr_max=snr_max,
        dropout=config.CLS.HEAD_DROPOUT,
        head_type=getattr(config.CLS, "HEAD_TYPE", "pool"),
        num_queries=getattr(config.CLS, "NUM_QUERIES", 4),
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


def _save_curve(snr_list, values, name, log_dir, prefix="snr"):
    csv_path = os.path.join(log_dir, f"{prefix}_{name}_curve.csv")
    png_path = os.path.join(log_dir, f"{prefix}_{name}_curve.png")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["snr", name])
        writer.writerows(zip(snr_list, values))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(snr_list, values, marker="o")
        plt.xlabel("SNR (dB)")
        plt.ylabel(name)
        plt.title(f"SNR-{name}")
        plt.grid(True)
        # Accuracy on a fixed [0, 1] axis so a flat plateau reads as flat and a
        # real drop is visible -- auto-scaling zooms into sub-1% noise and makes a
        # dead-flat curve look like wild fluctuation.
        if name == "acc":
            plt.ylim(0.0, 1.0)
        plt.tight_layout()
        plt.savefig(png_path, dpi=200)
        plt.close()
    except Exception as exc:
        print(f"Failed to save {name} curve: {exc}")


@torch.no_grad()
def evaluate_classification(
    config,
    encoder,
    classifier,
    test_loader=None,
    criterion_cls=None,
    save_curves=True,
    prefix="snr",
):
    if test_loader is None:
        _, test_loader = get_loader(config)
    device = next(encoder.parameters()).device
    channel = Channel(config)
    encoder.eval()
    classifier.eval()

    # Validation/test sweep CHANNEL.EVAL_SNR when provided (training still samples
    # CHANNEL.SNR), else fall back to CHANNEL.SNR.
    snr_list = getattr(config.CHANNEL, "EVAL_SNR", None) or config.CHANNEL.SNR
    acc_all, loss_all = [], []
    all_time = 0.0

    for snr in snr_list:
        # Channel uses the true (swept) SNR; the model is fed a fixed SNR when
        # BLIND_MODEL is set (mismatched-CSI ablation -> accuracy rises with SNR).
        model_snr = (
            float(config.CHANNEL.MODEL_SNR)
            if getattr(config.CHANNEL, "BLIND_MODEL", False)
            else snr
        )
        correct = 0
        total = 0
        loss_sum = 0.0
        n_batch = 0
        seed_torch()
        with tqdm(test_loader, dynamic_ncols=False) as tqdm_data:
            for input_image, labels in tqdm_data:
                if not torch.is_tensor(labels):
                    raise ValueError("Classification requires class labels.")
                input_image = input_image.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                input_image = _prepare_input(config, input_image)

                start = time.time()
                feature = encoder(input_image, model_snr)
                if getattr(config.CHANNEL, "COMPACT_CODE", False):
                    z_hat = apply_channel_compact(channel, config, feature, snr)
                else:
                    z_hat = apply_channel(channel, config, feature, snr)
                logits = classifier(z_hat, model_snr)
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
                tqdm_data.set_postfix({"SNR": snr, "Acc": acc_batch})

        acc_all.append(correct / max(total, 1))
        if criterion_cls is not None:
            loss_all.append(loss_sum / n_batch)

    print(all_time / (len(snr_list) * len(test_loader) * config.DATA.TEST_BATCH))
    print("SNRs:", snr_list)
    print("Accuracy:", acc_all)
    if criterion_cls is not None:
        print("Cls loss:", loss_all)
    if save_curves:
        log_dir = get_log_dir(config)
        _save_curve(snr_list, acc_all, "acc", log_dir, prefix=prefix)
        if loss_all:
            _save_curve(snr_list, loss_all, "cls_loss", log_dir, prefix=prefix)
    mean_loss = sum(loss_all) / len(loss_all) if loss_all else None
    return acc_all, mean_loss


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
