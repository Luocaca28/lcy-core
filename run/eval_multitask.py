import csv
import os
import time

import torch
from tqdm import tqdm

from data.datasets import get_loader
from models.channel import Channel
from models.network import Mamba_decoder, Mamba_encoder
from models.task_head import LatentClassifierHead
from run.eval import _get_log_dir, _msssim_value, _psnr_value
from utils.distortion import MS_SSIM
from utils.utils import save_model, seed_torch


def multitask_checkpoint_name(config):
    return "OUTCHANS{}_extent{}_loss{}_SCANnum{}_SNR{}_adp{}_type{}_depth{}_embed{}_nums{}_rsl{}".format(
        config.MODEL.VSSM.OUT_CHANS,
        config.MODEL.VSSM.Extent,
        config.TRAIN.LOSS,
        config.MODEL.VSSM.SCAN_NUMBER,
        config.CHANNEL.SNR,
        config.CHANNEL.ADAPTIVE,
        config.CHANNEL.TYPE,
        len(config.MODEL.VSSM.EMBED_DIM),
        config.MODEL.VSSM.EMBED_DIM,
        config.MODEL.VSSM.DEPTHS,
        config.DATA.IMG_SIZE,
    )


def classifier_save_dir(config):
    path = getattr(config.TASK, "CLASSIFIER_PATH", "")
    if path:
        os.makedirs(path, exist_ok=True)
        return path
    log_dir = _get_log_dir(config)
    path = os.path.join(os.path.dirname(os.path.normpath(log_dir)), "classifier")
    os.makedirs(path, exist_ok=True)
    return path + os.sep


def build_latent_classifier(config):
    snr_list = getattr(config.CHANNEL, "SNR", [20])
    snr_max = max(snr_list) if isinstance(snr_list, (list, tuple)) else float(snr_list)
    return LatentClassifierHead(
        latent_dim=config.MODEL.VSSM.OUT_CHANS,
        num_classes=config.TASK.NUM_CLASSES,
        hidden_dim=config.TASK.HEAD_HIDDEN_DIM,
        snr_embed_dim=config.TASK.SNR_EMBED_DIM,
        use_snr=config.TASK.USE_SNR_EMBED,
        snr_max=snr_max,
        dropout=config.TASK.HEAD_DROPOUT,
    )


def _msssim_levels_for_size(size, max_levels=4, window_size=11):
    levels = 1
    current = int(size)
    while levels < max_levels and current // 2 >= window_size:
        current = current // 2
        levels += 1
    return levels


def save_multitask_models(config, encoder, decoder, classifier):
    name = multitask_checkpoint_name(config) + ".pt"
    save_model(encoder, os.path.join(config.TRAIN.ENCODER_PATH, name))
    save_model(decoder, os.path.join(config.TRAIN.DECODER_PATH, name))
    save_model(classifier, os.path.join(classifier_save_dir(config), name))


def _format_received(config, received, pwr, h, snr):
    if config.CHANNEL.TYPE == "rayleigh":
        sigma_square = 1.0 / (10 ** (snr / 10))
        received = torch.conj(h) * received / (torch.abs(h) ** 2 + sigma_square)
    elif config.CHANNEL.TYPE == "awgn":
        pass
    else:
        raise ValueError("channel type error")
    return torch.cat((torch.real(received), torch.imag(received)), dim=2) * torch.sqrt(pwr)


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
        plt.tight_layout()
        plt.savefig(png_path, dpi=200)
        plt.close()
    except Exception as exc:
        print(f"Failed to save {name} curve: {exc}")


@torch.no_grad()
def eval_MambaJSCC_multitask_models(
    config,
    encoder,
    decoder,
    classifier,
    test_loader=None,
    criterion_rec=None,
    criterion_cls=None,
    save_curves=True,
    prefix="snr",
):
    if test_loader is None:
        _, test_loader = get_loader(config)
    device = next(encoder.parameters()).device
    channel = Channel(config)
    msssim_levels = _msssim_levels_for_size(config.DATA.IMG_SIZE, max_levels=4)
    msssim_calculator = MS_SSIM(data_range=1.0, levels=msssim_levels, channel=3).to(device)
    encoder.eval()
    decoder.eval()
    classifier.eval()

    snr_list = config.CHANNEL.SNR
    psnr_all, msssim_all, acc_all, loss_all = [], [], [], []
    all_time = 0.0

    for snr in snr_list:
        psnr_sum = 0.0
        msssim_sum = 0.0
        acc_sum = 0.0
        loss_sum = 0.0
        n_batch = 0
        seed_torch()
        with tqdm(test_loader, dynamic_ncols=False) as tqdm_data:
            for input_image, labels in tqdm_data:
                if not torch.is_tensor(labels):
                    raise ValueError("Multitask classification requires class labels.")
                input_image = input_image.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                if config.DATA.DATASET == "CIFAR10":
                    input_image = torch.nn.functional.interpolate(
                        input_image,
                        (config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                        mode="bilinear",
                        align_corners=False,
                    )

                start = time.time()
                feature = encoder(input_image, snr)
                received, pwr, h = channel.forward(feature, snr)
                z_hat = _format_received(config, received, pwr, h, snr)
                recon_image = decoder(z_hat, snr)
                logits = classifier(z_hat, snr)
                all_time += time.time() - start

                psnr_batch = sum(
                    _psnr_value(recon_image[j : j + 1], input_image[j : j + 1])
                    for j in range(recon_image.shape[0])
                ) / recon_image.shape[0]
                msssim_batch = sum(
                    _msssim_value(recon_image[j : j + 1], input_image[j : j + 1], msssim_calculator)
                    for j in range(recon_image.shape[0])
                ) / recon_image.shape[0]
                acc_batch = (logits.argmax(dim=1) == labels).float().mean().item()

                if criterion_rec is not None and criterion_cls is not None:
                    rec_loss = criterion_rec(recon_image, input_image, feature, opt_idx=0, global_step=0)
                    cls_loss = criterion_cls(logits, labels)
                    loss_sum += (
                        config.TASK.REC_LOSS_WEIGHT * rec_loss.item()
                        + config.TASK.CLS_LOSS_WEIGHT * cls_loss.item()
                    )

                psnr_sum += psnr_batch
                msssim_sum += msssim_batch
                acc_sum += acc_batch
                n_batch += 1
                tqdm_data.set_postfix(
                    {
                        "SNR": snr,
                        "PSNR": psnr_batch,
                        "MS-SSIM": msssim_batch,
                        "Acc": acc_batch,
                    }
                )

        psnr_all.append(psnr_sum / n_batch)
        msssim_all.append(msssim_sum / n_batch)
        acc_all.append(acc_sum / n_batch)
        if criterion_rec is not None and criterion_cls is not None:
            loss_all.append(loss_sum / n_batch)

    print(all_time / (len(snr_list) * len(test_loader) * config.DATA.TEST_BATCH))
    print("SNRs:", snr_list)
    print("PSNR:", psnr_all)
    print("MS-SSIM:", msssim_all)
    print("Accuracy:", acc_all)
    if save_curves:
        log_dir = _get_log_dir(config)
        _save_curve(snr_list, psnr_all, "psnr", log_dir, prefix=prefix)
        _save_curve(snr_list, msssim_all, "MS-SSIM", log_dir, prefix=prefix)
        _save_curve(snr_list, acc_all, "acc", log_dir, prefix=prefix)
    mean_loss = sum(loss_all) / len(loss_all) if loss_all else None
    return psnr_all, msssim_all, acc_all, mean_loss


@torch.no_grad()
def test_MambaJSCC_multitask(config):
    _, test_loader = get_loader(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = multitask_checkpoint_name(config) + ".pt"
    encoder = torch.load(
        os.path.join(config.TRAIN.ENCODER_PATH, name),
        weights_only=False,
        map_location=device,
    ).to(device)
    decoder = torch.load(
        os.path.join(config.TRAIN.DECODER_PATH, name),
        weights_only=False,
        map_location=device,
    ).to(device)
    classifier = torch.load(
        os.path.join(classifier_save_dir(config), name),
        weights_only=False,
        map_location=device,
    ).to(device)
    eval_MambaJSCC_multitask_models(config, encoder, decoder, classifier, test_loader=test_loader)
