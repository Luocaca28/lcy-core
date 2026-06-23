"""Reconstruction task: evaluation entry points for DefMambaJSCC.

Evaluates a trained encoder/decoder pair across the configured SNR sweep and
reports PSNR / MS-SSIM. Shared plumbing (log dir, channel forward, checkpoint
naming) comes from ``utils.engine``.
"""

import csv
import os
import time

import numpy as np
import torch
from torchvision.utils import save_image
from tqdm import tqdm

from data.datasets import get_loader
from models.channel import Channel
from utils.distortion import MS_SSIM, ReconstructionMetric
from utils.engine import apply_channel, checkpoint_tag, get_log_dir
from utils.utils import seed_torch


def _get_output_root(config):
    log_dir = get_log_dir(config)
    return os.path.dirname(os.path.normpath(log_dir))


def _metric_file_token(metric_name):
    if metric_name == "PSNR":
        return "psnr"
    return metric_name


def _save_curve(snr_list, values, metric_name, log_dir, prefix="snr"):
    token = _metric_file_token(metric_name)
    csv_path = os.path.join(log_dir, f"{prefix}_{token}_curve.csv")
    png_path = os.path.join(log_dir, f"{prefix}_{token}_curve.png")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["snr", metric_name])
        writer.writerows(zip(snr_list, values))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(snr_list, values, marker="o")
        plt.xlabel("SNR (dB)")
        plt.ylabel(metric_name)
        plt.title(f"SNR-{metric_name}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(png_path, dpi=200)
        plt.close()
    except Exception as exc:
        print(f"Failed to save SNR curve plot: {exc}")


def _save_eval_curves(snr_list, psnr_all, msssim_all, config, prefix="snr"):
    log_dir = get_log_dir(config)
    _save_curve(snr_list, psnr_all, "PSNR", log_dir, prefix=prefix)
    _save_curve(snr_list, msssim_all, "MS-SSIM", log_dir, prefix=prefix)


def _psnr_value(x, y):
    mse = torch.nn.functional.mse_loss(x.clamp(0.0, 1.0), y.clamp(0.0, 1.0))
    if mse.item() == 0:
        return float("inf")
    return (-10.0 * torch.log10(mse)).item()


def _msssim_value(x, y, calculator):
    # utils.distortion.MS_SSIM returns 1 - ms_ssim; convert it back to MS-SSIM.
    return 1.0 - calculator(x.clamp(0.0, 1.0), y.clamp(0.0, 1.0)).mean().item()


def _target_name(target, index):
    if isinstance(target, (list, tuple)):
        return str(target[index])
    return str(target)


def _reconstruction_checkpoint_paths(config):
    tag = checkpoint_tag(config)
    encoder_path = config.TRAIN.ENCODER_PATH + tag + ".pt"
    decoder_path = config.TRAIN.DECODER_PATH + tag + ".pt"
    return encoder_path, decoder_path


@torch.no_grad()
def evaluate_reconstruction(
    config,
    encoder,
    decoder,
    test_loader=None,
    save_recon=True,
    save_curves=True,
    prefix="snr",
    criterion=None,
    global_step=0,
):
    if test_loader is None:
        _, test_loader = get_loader(config)
    channel = Channel(config)
    B, C, H, W = next(iter(test_loader))[0].shape
    print(H, W)
    device = next(encoder.parameters()).device
    msssim_calculator = MS_SSIM(data_range=1.0, levels=4, channel=3).to(device)
    encoder.eval()
    decoder.eval()
    performance_all = []
    psnr_all = []
    msssim_all = []
    loss_all = []
    snr_list = config.CHANNEL.SNR
    output_root = _get_output_root(config)
    recon_root = os.path.join(output_root, "recon") if save_recon else None
    log_dir = get_log_dir(config)
    print(f"----------Evaluating reconstruction: {checkpoint_tag(config)}")
    all_time = 0

    for snr in snr_list:
        performance_avg = 0
        psnr_avg = 0
        msssim_avg = 0
        loss_avg = 0
        per_image_rows = []
        if save_recon:
            recon_dir = os.path.join(recon_root, f"SNR_{snr}")
            os.makedirs(recon_dir, exist_ok=True)
        seed_torch()
        with tqdm(test_loader, dynamic_ncols=False) as tqdm_data:
            for i, (input_image, target) in enumerate(tqdm_data):
                input_image = input_image.to(device, non_blocking=True)
                if config.DATA.DATASET == "CIFAR10":
                    input_image = torch.nn.functional.interpolate(
                        input_image, (128, 128), mode="nearest"
                    )
                start_encoder = time.time()
                feature = encoder(input_image, snr)
                end_encoder = time.time()
                cbr = feature.numel() / 2 / input_image.numel()

                z_hat = apply_channel(channel, config, feature, snr)
                start_decoder = time.time()
                recon_image = decoder(z_hat, snr)
                end_decoder = time.time()
                all_time = all_time + end_encoder - start_encoder + end_decoder - start_decoder
                if criterion is not None:
                    loss_value = criterion(
                        recon_image, input_image, feature, opt_idx=0, global_step=global_step
                    )
                    loss_avg += loss_value.item()

                batch_psnr_values = []
                batch_msssim_values = []
                for sample_idx in range(recon_image.shape[0]):
                    recon_sample = recon_image[sample_idx : sample_idx + 1]
                    input_sample = input_image[sample_idx : sample_idx + 1]
                    psnr = _psnr_value(recon_sample, input_sample)
                    msssim = _msssim_value(recon_sample, input_sample, msssim_calculator)
                    batch_psnr_values.append(psnr)
                    batch_msssim_values.append(msssim)

                    if save_recon:
                        name = _target_name(target, sample_idx)
                        stem = os.path.splitext(os.path.basename(name))[0]
                        recon_name = f"{stem}_SNR{snr}_PSNR{psnr:.4f}_MSSSIM{msssim:.6f}.png"
                        save_image(recon_sample.clamp(0.0, 1.0), os.path.join(recon_dir, recon_name))
                        per_image_rows.append([name, snr, psnr, msssim])

                psnr_batch = sum(batch_psnr_values) / len(batch_psnr_values)
                msssim_batch = sum(batch_msssim_values) / len(batch_msssim_values)
                if config.TRAIN.EVAL_MATRIX == "MSSSIM":
                    performance = msssim_batch
                else:
                    performance = psnr_batch
                performance_avg = performance_avg + performance
                psnr_avg = psnr_avg + psnr_batch
                msssim_avg = msssim_avg + msssim_batch
                tqdm_data.set_postfix(
                    {
                        "matrix": performance,
                        "PSNR": psnr_batch,
                        "MS-SSIM": msssim_batch,
                        "CBR": cbr,
                        "SNR": snr,
                        "per": (performance, performance_avg / (i + 1)),
                    }
                )

        performance_all.append(performance_avg / (i + 1))
        psnr_all.append(psnr_avg / (i + 1))
        msssim_all.append(msssim_avg / (i + 1))
        if criterion is not None:
            loss_all.append(loss_avg / (i + 1))
        if save_recon:
            metric_csv = os.path.join(log_dir, f"per_image_metrics_SNR_{snr}.csv")
            with open(metric_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["image", "snr", "psnr", "ms_ssim"])
                writer.writerows(per_image_rows)

    print(all_time / (len(snr_list) * len(test_loader) * config.DATA.TEST_BATCH))
    print("SNRs:", snr_list)
    print("performance:", performance_all)
    print("PSNR:", psnr_all)
    print("MS-SSIM:", msssim_all)
    if criterion is not None:
        print("loss:", loss_all)
    if save_curves:
        _save_eval_curves(snr_list, psnr_all, msssim_all, config, prefix=prefix)
    mean_loss = sum(loss_all) / len(loss_all) if loss_all else None
    return performance_all, psnr_all, msssim_all, mean_loss


@torch.no_grad()
def test_reconstruction(config):
    _, test_loader = get_loader(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_path, decoder_path = _reconstruction_checkpoint_paths(config)
    encoder = torch.load(encoder_path, weights_only=False, map_location=device).to(device)
    decoder = torch.load(decoder_path, weights_only=False, map_location=device).to(device)
    evaluate_reconstruction(
        config,
        encoder,
        decoder,
        test_loader=test_loader,
        save_recon=True,
        prefix="snr",
    )


def evaluate_reconstruction_with_snr_error(config, mode=2):
    """Robustness diagnostic: inject Gaussian SNR estimation error.

    mode 1: fixed encoder SNR, perturbed channel SNR.
    mode 2: perturbed encoder SNR, fixed channel SNR.
    """
    _, test_loader = get_loader(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_path, decoder_path = _reconstruction_checkpoint_paths(config)
    encoder = torch.load(encoder_path, weights_only=False, map_location=device).to(device)
    decoder = torch.load(decoder_path, weights_only=False, map_location=device).to(device)

    channel = Channel(config)
    metric = ReconstructionMetric(config)
    encoder.eval()
    decoder.eval()

    snr_list = [1, 5, 10, 15, 20]
    error_rate = [0.01, 0.1, 0.5, 1, 2]
    print(f"----------Evaluating SNR error: {checkpoint_tag(config)}")
    for error in error_rate:
        performance_all = []
        for snr in snr_list:
            performance_avg = 0
            seed_torch()
            with tqdm(test_loader, dynamic_ncols=False) as tqdm_data:
                for i, (input_image, target) in enumerate(tqdm_data):
                    input_image = input_image.to(device, non_blocking=True)
                    snr_error = snr + np.random.normal(0, error)

                    if mode == 1:
                        feature = encoder(input_image, snr)
                        received, pwr, h = channel.forward(feature, snr_error)
                        if config.CHANNEL.TYPE == "rayleigh":
                            sigma_square = 1.0 / (10 ** (snr / 10))
                            received = torch.conj(h) * received / (torch.abs(h) ** 2 + sigma_square)
                        elif config.CHANNEL.TYPE == "awgn":
                            pass
                        else:
                            raise ValueError("channel type error")
                        z_hat = torch.cat(
                            (torch.real(received), torch.imag(received)), dim=2
                        ) * torch.sqrt(pwr)
                        recon_image = decoder(z_hat, snr)
                    elif mode == 2:
                        feature = encoder(input_image, snr_error)
                        received, pwr, h = channel.forward(feature, snr)
                        if config.CHANNEL.TYPE == "rayleigh":
                            sigma_square = 1.0 / (10 ** (snr_error / 10))
                            received = torch.conj(h) * received / (torch.abs(h) ** 2 + sigma_square)
                        elif config.CHANNEL.TYPE == "awgn":
                            pass
                        else:
                            raise ValueError("channel type error")
                        z_hat = torch.cat(
                            (torch.real(received), torch.imag(received)), dim=2
                        ) * torch.sqrt(pwr)
                        recon_image = decoder(z_hat, snr_error)

                    cbr = feature.numel() / 2 / input_image.numel()
                    performance = metric(recon_image, input_image)
                    performance_avg = performance_avg + performance
                    tqdm_data.set_postfix(
                        {
                            "matrix": performance,
                            "CBR": cbr,
                            "SNR": snr,
                            "SNR_error": snr_error,
                            "per": (performance, performance_avg / (i + 1)),
                        }
                    )

            performance_all.append(performance_avg / (i + 1))

        print("SNRs:", snr_list)
        print(f"performance with {error}:", performance_all)


def test_mem_and_comp(config, encoder, decoder, input_size=(256, 256)):
    from torch_operation_counter import OperationsCounterMode

    class _Net(torch.nn.Module):
        def __init__(self, encoder, decoder):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder

        def forward(self, input):
            snr = 20
            x = self.encoder(input, snr)
            y = self.decoder(x, snr)
            return y

    device = next(encoder.parameters()).device
    network = _Net(encoder, decoder).to(device)
    input = torch.randn(1, 3, input_size[0], input_size[1], device=device)
    with OperationsCounterMode(network) as ops_counter:
        network(input)
    print(
        "MACs:{}G. Paras:{}M.".format(
            ops_counter.total_operations / 1e9,
            sum([p.numel() for p in [*network.parameters()][:-1]]) / 1e6,
        )
    )
