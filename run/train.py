"""Reconstruction task: training entry point for DefMambaJSCC.

This is one of the two independent tasks. It trains the encoder + decoder for
image reconstruction over a noisy channel. All shared plumbing (distributed
helpers, optimizer/scheduler builders, channel forward, profiling, checkpoint
naming) lives in ``utils.engine`` so this file only expresses the reconstruction
training loop itself.
"""

import csv
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from data.datasets import get_loader
from models.channel import Channel
from models.network import Mamba_encoder, Mamba_decoder
from run.eval import evaluate_reconstruction
from utils.distortion import ReconstructionLoss, ReconstructionMetric
from utils.engine import (
    apply_channel,
    build_adamw_with_tri_merge,
    build_warmup_cosine_schedulers,
    checkpoint_tag,
    collect_tri_path_weight_stats,
    get_log_dir,
    inspect_tri_merge_grad,
    is_main_process,
    print_optimizer_groups,
    print_pipeline_profile,
    reduce_scalar,
    save_tri_path_weight_curve,
    setup_distributed,
    unwrap_parallel,
)
from utils.utils import save_model, seed_torch


def _save_loss_curve(train_records, val_records, log_dir):
    csv_path = os.path.join(log_dir, "loss_curve.csv")
    png_path = os.path.join(log_dir, "loss_curve.png")
    val_by_epoch = {row[0]: row[1] for row in val_records}
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for epoch, train_loss in train_records:
            writer.writerow([epoch, train_loss, val_by_epoch.get(epoch, "")])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        train_epochs = [row[0] for row in train_records]
        train_losses = [row[1] for row in train_records]
        val_epochs = [row[0] for row in val_records]
        val_losses = [row[1] for row in val_records]
        plt.figure()
        plt.plot(train_epochs, train_losses, marker="o", label="train")
        if val_records:
            plt.plot(val_epochs, val_losses, marker="s", label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Train/Val Loss")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(png_path, dpi=200)
        plt.close()
    except Exception as exc:
        print(f"Failed to save loss plot: {exc}")


def _save_reconstruction_checkpoints(config, encoder, decoder):
    tag = checkpoint_tag(config)
    save_model(unwrap_parallel(encoder), save_path=config.TRAIN.ENCODER_PATH + tag + ".pt")
    save_model(unwrap_parallel(decoder), save_path=config.TRAIN.DECODER_PATH + tag + ".pt")


def train_reconstruction(config):
    distributed, rank, local_rank = setup_distributed()
    val_data_dir = getattr(config.DATA, "val_data_dir", config.DATA.test_data_dir)
    if is_main_process():
        print(f"Validation data dir: {val_data_dir}")
    train_loader, val_loader = get_loader(config, test_data_dir=val_data_dir)
    device = torch.device("cuda", local_rank) if distributed else torch.device("cuda")

    encoder = Mamba_encoder(config).to(device)
    decoder = Mamba_decoder(config).to(device)
    if distributed:
        print(f"Using DDP rank {rank}, local_rank {local_rank}")
        encoder = DDP(encoder, device_ids=[local_rank], output_device=local_rank)
        decoder = DDP(decoder, device_ids=[local_rank], output_device=local_rank)
    channel = Channel(config)
    if is_main_process():
        print_pipeline_profile(config, [("Encoder", encoder), ("Decoder", decoder)], device)
    if distributed:
        dist.barrier()

    tri_lr_mult = getattr(config.MODEL.VSSM, "TRI_MERGE_LR_MULT", 1.0)
    optimizers = {
        "encoder": build_adamw_with_tri_merge(
            encoder, config.TRAIN.BASE_LR, config.TRAIN.WEIGHT_DECAY, tri_lr_mult
        ),
        "decoder": build_adamw_with_tri_merge(
            decoder, config.TRAIN.BASE_LR, config.TRAIN.WEIGHT_DECAY, tri_lr_mult
        ),
    }
    schedulers = build_warmup_cosine_schedulers(config, optimizers, train_loader)
    if is_main_process():
        for name, optimizer in optimizers.items():
            print_optimizer_groups(name, optimizer)

    criterion = ReconstructionLoss(config)
    metric = ReconstructionMetric(config)

    encoder.train()
    decoder.train()
    if is_main_process():
        print(config.MODEL.VSSM.EMBED_DIM, config.MODEL.VSSM.DEPTHS)
        print(f"----------training reconstruction: {checkpoint_tag(config)}-------")

    seed_torch()
    loss_records = []
    val_loss_records = []
    tri_weight_records = []
    log_dir = get_log_dir(config)
    eval_fre = getattr(config.TRAIN, "EVAL_FRE", 10)
    for e in range(config.TRAIN.EPOCHS):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(e)
        loss_ave = 0

        with tqdm(train_loader, dynamic_ncols=False, disable=not is_main_process()) as tqdm_data:
            for i, (input_image, target) in enumerate(tqdm_data):
                snr_list = config.CHANNEL.SNR
                snr = snr_list[torch.randint(0, len(snr_list), (1,)).item()]

                input_image = input_image.to(device, non_blocking=True)
                optimizers["encoder"].zero_grad()
                optimizers["decoder"].zero_grad()

                feature = encoder(input_image, snr)
                cbr = feature.numel() / input_image.numel() / 2
                z_hat = apply_channel(channel, config, feature, snr)
                recon_image = decoder(z_hat, snr)

                loss = criterion(recon_image, input_image, feature, opt_idx=0, global_step=e)
                loss.backward()
                if getattr(config.MODEL.VSSM, "TRI_MERGE_DEBUG", False) and is_main_process() and i == 0:
                    inspect_tri_merge_grad(encoder, f"epoch {e} encoder")
                    inspect_tri_merge_grad(decoder, f"epoch {e} decoder")

                performance = metric(recon_image, input_image)
                loss_ave = loss_ave + loss.item()

                torch.nn.utils.clip_grad_norm_(encoder.parameters(), config.TRAIN.CLIP_GRAD)
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), config.TRAIN.CLIP_GRAD)
                optimizers["encoder"].step()
                optimizers["decoder"].step()
                schedulers["encoder"].step()
                schedulers["decoder"].step()

                tqdm_data.set_postfix(
                    {
                        "e": e,
                        "loss": (loss.item(), loss_ave / (i + 1)),
                        "matrix": performance,
                        "CBR": cbr,
                        "SNR": snr,
                        "LR": tuple(group["lr"] for group in optimizers["encoder"].param_groups),
                    }
                )

        loss_ave = reduce_scalar(loss_ave / (i + 1), device)
        if is_main_process():
            loss_records.append([e + 1, loss_ave])
            tri_record = {"epoch": e + 1}
            tri_record.update(collect_tri_path_weight_stats(encoder, "encoder"))
            tri_record.update(collect_tri_path_weight_stats(decoder, "decoder"))
            tri_weight_records.append(tri_record)
        if is_main_process() and (e + 1) % config.TRAIN.SAVE_FRE == 0:
            _save_reconstruction_checkpoints(config, encoder, decoder)

        run_validation = eval_fre > 0 and (
            (e + 1) % eval_fre == 0 or (e + 1) == config.TRAIN.EPOCHS
        )
        if run_validation:
            if distributed:
                dist.barrier()
            if is_main_process():
                print(f"----------validation after epoch {e + 1}----------")
                encoder.eval()
                decoder.eval()
                save_final_val_curves = (e + 1) == config.TRAIN.EPOCHS
                _, _, _, val_loss = evaluate_reconstruction(
                    config,
                    unwrap_parallel(encoder),
                    unwrap_parallel(decoder),
                    test_loader=val_loader,
                    save_recon=False,
                    save_curves=save_final_val_curves,
                    prefix=f"val_epoch_{e + 1:03d}_snr",
                    criterion=criterion,
                    global_step=e,
                )
                if val_loss is not None:
                    val_loss_records.append([e + 1, val_loss])
                encoder.train()
                decoder.train()
            if distributed:
                dist.barrier()

    if is_main_process():
        _save_reconstruction_checkpoints(config, encoder, decoder)
        _save_loss_curve(loss_records, val_loss_records, log_dir)
        save_tri_path_weight_curve(tri_weight_records, log_dir)
    if distributed:
        dist.barrier()
