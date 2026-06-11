"""
@author: Tong Wu
@contact: wu_tong@sjtu.edu.cn
"""

from models.network import Mamba_encoder, Mamba_decoder
from models.channel import Channel
from data.datasets import get_loader
import torch
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image
from utils.utils import *
from utils.distortion import *
import time
import csv
import os


def _get_log_dir(config):
    log_path = getattr(config.TRAIN, "LOG_PATH", "")
    if log_path:
        os.makedirs(log_path, exist_ok=True)
        return log_path
    base = os.path.commonpath([config.TRAIN.ENCODER_PATH, config.TRAIN.DECODER_PATH])
    log_dir = os.path.join(base, "logs")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def _get_output_root(config):
    log_dir = _get_log_dir(config)
    return os.path.dirname(os.path.normpath(log_dir))


def _save_curve(snr_list, values, metric_name, log_dir, prefix="snr"):
    csv_path = os.path.join(log_dir, f"{prefix}_{metric_name.lower()}_curve.csv")
    png_path = os.path.join(log_dir, f"{prefix}_{metric_name.lower()}_curve.png")

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
    log_dir = _get_log_dir(config)
    _save_curve(snr_list, psnr_all, "PSNR", log_dir, prefix=prefix)
    _save_curve(snr_list, msssim_all, "MS-SSIM", log_dir, prefix=prefix)


def _psnr_value(x, y):
    mse = torch.nn.functional.mse_loss(
        x.clamp(0.0, 1.0) * 255.0, y.clamp(0.0, 1.0) * 255.0
    )
    return (10 * (torch.log(255.0 * 255.0 / mse) / np.log(10))).item()


def _msssim_value(x, y, calculator):
    return 1.0 - calculator(x.clamp(0.0, 1.0), y.clamp(0.0, 1.0)).mean().item()


def _target_name(target, index):
    if isinstance(target, (list, tuple)):
        return str(target[index])
    return str(target)


def _eval_loss_value(config, recon_image, input_image, feature, msssim_calculator):
    if config.TRAIN.LOSS == "PSNR":
        return torch.nn.functional.mse_loss(
            recon_image, input_image, reduction="sum"
        ) / recon_image.shape[0]
    if config.TRAIN.LOSS == "MSSSIM":
        return (
            msssim_calculator(recon_image.clamp(0.0, 1.0), input_image.clamp(0.0, 1.0)).mean()
            * recon_image.numel()
            / recon_image.shape[0]
        )

    criterion = loss_matrix(config)
    return criterion(recon_image, input_image, feature, opt_idx=0, global_step=0)


@torch.no_grad()
def eval_MambaJSCC_models(
    config,
    encoder,
    decoder,
    test_loader=None,
    save_recon=True,
    prefix="snr",
    return_loss=False,
):
    if test_loader is None:
        _, test_loader = get_loader(config)
    channel = Channel(config)
    B, C, H, W = next(iter(test_loader))[0].shape
    # test_mem_and_comp(config, encoder, decoder, input_size=(H, W))

    print(H, W)
    msssim_calculator = MS_SSIM(data_range=1.0, levels=4, channel=3).cuda()
    encoder.eval()
    decoder.eval()
    performance_all = []
    psnr_all = []
    msssim_all = []
    loss_all = []
    # SNR_list = [20] #config.CHANNEL.SNR
    SNR_list = config.CHANNEL.SNR
    output_root = _get_output_root(config)
    recon_root = os.path.join(output_root, "recon") if save_recon else None
    log_dir = _get_log_dir(config)
    print(
        "----------Evaluating: ls:128--OUTCHANS{}_extent{}_loss{}_SCANnum{}_SNR{}_adp{}_type{}_depth{}_embed{}_nums{}_rsl{}".format(
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
    )
    all_time = 0

    for SNR in SNR_list:

        number = 0
        performance_avg = 0
        psnr_avg = 0
        msssim_avg = 0
        loss_avg = 0
        per_image_rows = []
        if save_recon:
            recon_dir = os.path.join(recon_root, f"SNR_{SNR}")
            os.makedirs(recon_dir, exist_ok=True)
        seed_torch()
        with tqdm(test_loader, dynamic_ncols=False) as tqdmTestData:
            for i, (input_image, target) in enumerate(tqdmTestData):
                input_image = input_image.cuda()
                if config.DATA.DATASET == "CIFAR10":
                    input_image = torch.nn.functional.interpolate(
                        input_image, (128, 128), mode="nearest"
                    )
                # print(input_image.shape)
                # target = target.cuda()
                start_encoder = time.time()
                feature = encoder(input_image, SNR)
                end_encoder = time.time()
                CBR = feature.numel() / 2 / input_image.numel()

                received, pwr, h = channel.forward(feature, SNR)
                if config.CHANNEL.TYPE == "rayleigh":
                    sigma_square = 1.0 / (10 ** (SNR / 10))
                    received = torch.conj(h) * received / (torch.abs(h) ** 2 + sigma_square)

                elif config.CHANNEL.TYPE == "awgn":
                    pass
                else:
                    raise ValueError("channel type error")

                received = torch.cat(
                    (torch.real(received), torch.imag(received)), dim=2
                ) * torch.sqrt(pwr)
                start_decoder = time.time()
                recon_image = decoder(received, SNR)
                end_decocer = time.time()
                all_time = all_time + end_encoder - start_encoder + end_decocer - start_decoder
                loss_batch = _eval_loss_value(
                    config, recon_image, input_image, feature, msssim_calculator
                ).item()

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
                        recon_name = f"{stem}_SNR{SNR}_PSNR{psnr:.4f}_MSSSIM{msssim:.6f}.png"
                        save_image(recon_sample.clamp(0.0, 1.0), os.path.join(recon_dir, recon_name))
                        per_image_rows.append([name, SNR, psnr, msssim])

                psnr_batch = sum(batch_psnr_values) / len(batch_psnr_values)
                msssim_batch = sum(batch_msssim_values) / len(batch_msssim_values)
                if config.TRAIN.EVAL_MATRIX == "MSSSIM":
                    performance = msssim_batch
                else:
                    performance = psnr_batch
                performance_avg = performance_avg + performance
                psnr_avg = psnr_avg + psnr_batch
                msssim_avg = msssim_avg + msssim_batch
                loss_avg = loss_avg + loss_batch
                tqdmTestData.set_postfix(
                    {
                        "matrix": performance,
                        "loss": loss_batch,
                        "PSNR": psnr_batch,
                        "MS-SSIM": msssim_batch,
                        "CBR": CBR,
                        "SNR": SNR,
                        "per": (performance, performance_avg / (i + 1)),
                    }
                )

        performance_all.append(performance_avg / (i + 1))
        psnr_all.append(psnr_avg / (i + 1))
        msssim_all.append(msssim_avg / (i + 1))
        loss_all.append(loss_avg / (i + 1))
        if save_recon:
            metric_csv = os.path.join(log_dir, f"per_image_metrics_SNR_{SNR}.csv")
            with open(metric_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["image", "snr", "psnr", "ms_ssim"])
                writer.writerows(per_image_rows)

    print(all_time / (len(SNR_list) * len(test_loader) * config.DATA.TEST_BATCH))
    print("SNRs:", SNR_list)
    print("performance:", performance_all)
    print("PSNR:", psnr_all)
    print("MS-SSIM:", msssim_all)
    print("loss:", loss_all)
    _save_eval_curves(SNR_list, psnr_all, msssim_all, config, prefix=prefix)
    avg_loss = sum(loss_all) / len(loss_all)
    if return_loss:
        return performance_all, psnr_all, msssim_all, avg_loss
    return performance_all, psnr_all, msssim_all


@torch.no_grad()
def test_MambaJSCC(config):
    _, test_loader = get_loader(config)

    encoder_path = (
        config.TRAIN.ENCODER_PATH
        + "OUTCHANS{}_extent{}_loss{}_SCANnum{}_SNR{}_adp{}_type{}_depth{}_embed{}_nums{}_rsl{}".format(
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
        + ".pt"
    )
    decoder_path = (
        config.TRAIN.DECODER_PATH
        + "OUTCHANS{}_extent{}_loss{}_SCANnum{}_SNR{}_adp{}_type{}_depth{}_embed{}_nums{}_rsl{}".format(
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
        + ".pt"
    )

    encoder = torch.load(encoder_path, weights_only=False)
    decoder = torch.load(decoder_path, weights_only=False)
    eval_MambaJSCC_models(
        config,
        encoder,
        decoder,
        test_loader=test_loader,
        save_recon=True,
        prefix="snr",
    )


def eval_MambaJSCC_with_SNR_error(config, mode=2):
    """
    SNR error with Gaussian random distribution
    mode 1 stand for fix estimation with various SNR
    mode 2 stand for fix SNR with various estimation
    """
    _, test_loader = get_loader(config)

    encoder_path = (
        config.TRAIN.ENCODER_PATH
        + "OUTCHANS{}_extent{}_loss{}_SCANnum{}_SNR{}_adp{}_type{}_depth{}_embed{}_nums{}_rsl{}".format(
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
        + ".pt"
    )
    decoder_path = (
        config.TRAIN.DECODER_PATH
        + "OUTCHANS{}_extent{}_loss{}_SCANnum{}_SNR{}_adp{}_type{}_depth{}_embed{}_nums{}_rsl{}".format(
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
        + ".pt"
    )

    encoder = torch.load(encoder_path)
    decoder = torch.load(decoder_path)

    channel = Channel(config)

    matrix = eval_matrix(config)
    encoder.eval()
    decoder.eval()

    SNR_list = [1, 5, 10, 15, 20]  # config.CHANNEL.SNR
    error_rate = [0.01, 0.1, 0.5, 1, 2]
    print(
        "----------Evaluating SNR error :OUTCHANS{}_extent{}_loss{}_SCANnum{}_SNR{}_adp{}_type{}_depth{}_embed{}_nums{}_rsl{}".format(
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
    )
    for error in error_rate:
        performance_all = []
        for SNR in SNR_list:
            number = 0
            performance_avg = 0
            seed_torch()
            with tqdm(test_loader, dynamic_ncols=False) as tqdmTestData:
                for i, (input_image, target) in enumerate(tqdmTestData):
                    input_image = input_image.cuda()
                    SNR_error = SNR + np.random.normal(0, error)

                    if mode == 1:
                        feature = encoder(input_image, SNR)
                        received, pwr, h = channel.forward(feature, SNR_error)
                        if config.CHANNEL.TYPE == "rayleigh":
                            sigma_square = 1.0 / (10 ** (SNR / 10))
                            received = torch.conj(h) * received / (torch.abs(h) ** 2 + sigma_square)
                            # print(1)
                        elif config.CHANNEL.TYPE == "awgn":
                            pass
                        else:
                            raise ValueError("channel type error")

                        received = torch.cat(
                            (torch.real(received), torch.imag(received)), dim=2
                        ) * torch.sqrt(pwr)

                        recon_image = decoder(received, SNR)

                    elif mode == 2:
                        feature = encoder(input_image, SNR_error)
                        received, pwr, h = channel.forward(feature, SNR)
                        if config.CHANNEL.TYPE == "rayleigh":
                            sigma_square = 1.0 / (10 ** (SNR_error / 10))
                            received = torch.conj(h) * received / (torch.abs(h) ** 2 + sigma_square)
                            # print(1)
                        elif config.CHANNEL.TYPE == "awgn":
                            pass
                        else:
                            raise ValueError("channel type error")

                        received = torch.cat(
                            (torch.real(received), torch.imag(received)), dim=2
                        ) * torch.sqrt(pwr)

                        recon_image = decoder(received, SNR_error)

                    CBR = feature.numel() / 2 / input_image.numel()
                    performance = matrix(recon_image, input_image)
                    performance_avg = performance_avg + performance
                    tqdmTestData.set_postfix(
                        {
                            "matrix": performance,
                            "CBR": CBR,
                            "SNR": SNR,
                            "SNR_error": SNR_error,
                            "per": (performance, performance_avg / (i + 1)),
                        }
                    )

            performance_all.append(performance_avg / (i + 1))

        print("SNRs:", SNR_list)
        print(f"performance with {error}:", performance_all)


def test_mem_and_comp(config, encoder, decoder, input_size=(256, 256)):
    from torch_operation_counter import OperationsCounterMode

    class net(torch.nn.Module):
        def __init__(self, encoder, decoder):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder

        def forward(self, input):

            SNR = 20
            x = self.encoder(input, SNR)
            y = self.decoder(x, SNR)
            return y

    network = net(encoder, decoder).cuda()
    input = torch.randn(1, 3, input_size[0], input_size[1]).cuda()
    with OperationsCounterMode(network) as ops_counter:
        network(input)
    # macs,params=profile(network,inputs=(input,))
    # macs, params = clever_format([macs, params], "%.5f")
    print(
        "MACs:{}G. Paras:{}M.".format(
            ops_counter.total_operations / 1e9,
            sum([p.numel() for p in [*network.parameters()][:-1]]) / 1e6,
        )
    )
