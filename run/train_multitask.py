import csv
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from data.datasets import get_loader
from models.channel import Channel
from models.network import Mamba_decoder, Mamba_encoder
from run.eval_multitask import (
    _format_received,
    build_latent_classifier,
    eval_MambaJSCC_multitask_models,
    save_multitask_models,
)
from run.train import (
    _build_adamw_with_tri_merge,
    _collect_tri_path_weight_stats,
    _count_params,
    _flops_of,
    _format_count,
    _format_count_with_commas,
    _get_log_dir,
    _inspect_tri_merge_grad,
    _is_main_process,
    _print_optimizer_groups,
    _param_memory_mb,
    _reduce_scalar,
    _save_tri_path_weight_curve,
    _setup_distributed,
    _unwrap_parallel,
)
from utils.distortion import eval_matrix, loss_matrix
from utils.utils import GradualWarmupScheduler, seed_torch


def _set_trainable(model, trainable):
    for param in model.parameters():
        param.requires_grad = trainable


def _has_trainable_params(model):
    return any(param.requires_grad for param in model.parameters())


def _configure_stage(config, encoder, decoder, classifier):
    stage = getattr(config.TASK, "STAGE", "joint").lower()
    if stage == "head_only":
        _set_trainable(encoder, False)
        _set_trainable(decoder, False)
        _set_trainable(classifier, True)
    elif stage == "encoder_head":
        _set_trainable(encoder, True)
        _set_trainable(decoder, False)
        _set_trainable(classifier, True)
    elif stage == "joint":
        _set_trainable(encoder, True)
        _set_trainable(decoder, True)
        _set_trainable(classifier, True)
    else:
        raise ValueError(f"Unknown TASK.STAGE: {config.TASK.STAGE}")

    if getattr(config.TASK, "FREEZE_ENCODER", False):
        _set_trainable(encoder, False)
    if getattr(config.TASK, "FREEZE_DECODER", False):
        _set_trainable(decoder, False)
    return stage


def _build_optimizers(config, encoder, decoder, classifier):
    tri_lr_mult = getattr(config.MODEL.VSSM, "TRI_MERGE_LR_MULT", 1.0)
    optimizers = {}
    if _has_trainable_params(encoder):
        optimizers["encoder"] = _build_adamw_with_tri_merge(
            encoder, config.TRAIN.BASE_LR, config.TRAIN.WEIGHT_DECAY, tri_lr_mult
        )
    if _has_trainable_params(decoder):
        optimizers["decoder"] = _build_adamw_with_tri_merge(
            decoder, config.TRAIN.BASE_LR, config.TRAIN.WEIGHT_DECAY, tri_lr_mult
        )
    if _has_trainable_params(classifier):
        optimizers["classifier"] = optim.AdamW(
            classifier.parameters(),
            lr=config.TASK.HEAD_LR,
            weight_decay=config.TASK.HEAD_WEIGHT_DECAY,
        )
    return optimizers


def _build_schedulers(config, optimizers, train_loader):
    total_steps = max(1, config.TRAIN.EPOCHS * len(train_loader))
    warmup_steps = max(1, int(config.TRAIN.WARMUP_EPOCHS * len(train_loader)))
    cosine_steps = max(1, total_steps - warmup_steps)
    schedulers = {}
    for name, optimizer in optimizers.items():
        cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=cosine_steps,
            eta_min=0,
            last_epoch=-1,
        )
        schedulers[name] = GradualWarmupScheduler(
            optimizer=optimizer,
            multiplier=2.0,
            warm_epoch=warmup_steps,
            after_scheduler=cosine,
        )
    return schedulers


def _zero_grad(optimizers):
    for optimizer in optimizers.values():
        optimizer.zero_grad()


def _optimizer_step(optimizers, schedulers):
    for name, optimizer in optimizers.items():
        optimizer.step()
        schedulers[name].step()


def _clip_trainable_grad(config, *models):
    for model in models:
        params = [p for p in model.parameters() if p.requires_grad]
        if params:
            torch.nn.utils.clip_grad_norm_(params, config.TRAIN.CLIP_GRAD)


def _prepare_input(config, input_image):
    if config.DATA.DATASET == "CIFAR10":
        return torch.nn.functional.interpolate(
            input_image,
            (config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
            mode="bilinear",
            align_corners=False,
        )
    return input_image


def _save_multitask_curve(records, log_dir):
    csv_path = os.path.join(log_dir, "multitask_curve.csv")
    png_path = os.path.join(log_dir, "multitask_curve.png")
    fieldnames = [
        "epoch",
        "train_total_loss",
        "train_rec_loss",
        "train_cls_loss",
        "train_acc",
        "val_total_loss",
        "val_acc",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = [row["epoch"] for row in records]
        train_losses = [row["train_total_loss"] for row in records]
        val_losses = [
            row["val_total_loss"] for row in records if row["val_total_loss"] != ""
        ]
        val_epochs = [
            row["epoch"] for row in records if row["val_total_loss"] != ""
        ]
        train_acc = [row["train_acc"] for row in records]
        val_acc = [row["val_acc"] for row in records if row["val_acc"] != ""]

        plt.figure()
        plt.plot(epochs, train_losses, marker="o", label="train loss")
        if val_losses:
            plt.plot(val_epochs, val_losses, marker="s", label="val loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, "multitask_loss_curve.png"), dpi=200)
        plt.close()

        plt.figure()
        plt.plot(epochs, train_acc, marker="o", label="train acc")
        if val_acc:
            plt.plot(val_epochs, val_acc, marker="s", label="val acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, "multitask_acc_curve.png"), dpi=200)
        plt.close()
    except Exception as exc:
        print(f"Failed to save multitask curves: {exc}")


def _module_train_state(stage, encoder, decoder, classifier):
    if stage == "head_only":
        encoder.eval()
        decoder.eval()
        classifier.train()
    elif stage == "encoder_head":
        encoder.train()
        decoder.eval()
        classifier.train()
    else:
        encoder.train()
        decoder.train()
        classifier.train()


def _print_multitask_model_profile(config, encoder, decoder, classifier, device):
    encoder = _unwrap_parallel(encoder)
    decoder = _unwrap_parallel(decoder)
    classifier = _unwrap_parallel(classifier)
    snr_list = config.CHANNEL.SNR
    profile_snr = snr_list[0] if isinstance(snr_list, (list, tuple)) else snr_list
    image_size = config.DATA.IMG_SIZE

    was_encoder_training = encoder.training
    was_decoder_training = decoder.training
    was_classifier_training = classifier.training
    encoder.eval()
    decoder.eval()
    classifier.eval()

    input_tensor = None
    feature = None
    decoder_input = None
    try:
        enc_params, _ = _count_params(encoder)
        dec_params, _ = _count_params(decoder)
        cls_params, _ = _count_params(classifier)
        total_params = enc_params + dec_params + cls_params
        enc_mem = _param_memory_mb(encoder)
        dec_mem = _param_memory_mb(decoder)
        cls_mem = _param_memory_mb(classifier)

        input_tensor = torch.randn(1, 3, image_size, image_size, device=device)
        with torch.no_grad():
            feature = encoder(input_tensor, profile_snr)
            decoder_input = torch.zeros_like(feature)

        enc_flops, _, enc_error = _flops_of(encoder, input_tensor, profile_snr)
        dec_flops, _, dec_error = _flops_of(decoder, decoder_input, profile_snr)
        cls_flops, _, cls_error = _flops_of(classifier, decoder_input, profile_snr)
        forward_flops = None
        train_flops_per_image = None
        train_flops_per_batch = None
        if enc_flops is not None and dec_flops is not None and cls_flops is not None:
            forward_flops = enc_flops + dec_flops + cls_flops
            train_flops_per_image = 3 * forward_flops
            train_flops_per_batch = train_flops_per_image * config.DATA.TRAIN_BATCH

        print(f"Encoder params: {_format_count(enc_params)} ({_format_count_with_commas(enc_params)})")
        print(f"Decoder params: {_format_count(dec_params)} ({_format_count_with_commas(dec_params)})")
        print(f"Classifier params: {_format_count(cls_params)} ({_format_count_with_commas(cls_params)})")
        print(f"Total params:   {_format_count(total_params)} ({_format_count_with_commas(total_params)})")
        print(f"Encoder size: {enc_mem:.2f} MB")
        print(f"Decoder size: {dec_mem:.2f} MB")
        print(f"Classifier size: {cls_mem:.2f} MB")
        print(f"Total size:   {enc_mem + dec_mem + cls_mem:.2f} MB")
        print(f"Encoder FLOPs: {_format_count(enc_flops)}")
        print(f"Decoder FLOPs: {_format_count(dec_flops)}")
        print(f"Classifier FLOPs: {_format_count(cls_flops)}")
        print(f"Forward FLOPs: {_format_count(forward_flops)}")
        print(f"Train FLOPs/image (fwd+bwd est.): {_format_count(train_flops_per_image)}")
        print(
            "Train FLOPs/batch "
            f"(batch={config.DATA.TRAIN_BATCH}, fwd+bwd est.): {_format_count(train_flops_per_batch)}"
        )
        if enc_error:
            print(f"Encoder FLOPs failed: {enc_error}")
        if dec_error:
            print(f"Decoder FLOPs failed: {dec_error}")
        if cls_error:
            print(f"Classifier FLOPs failed: {cls_error}")
    finally:
        del input_tensor, feature, decoder_input
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if was_encoder_training:
            encoder.train()
        if was_decoder_training:
            decoder.train()
        if was_classifier_training:
            classifier.train()


def train_MambaJSCC_multitask(config):
    if not getattr(config.TASK, "ENABLE", False):
        raise ValueError("Set TASK.ENABLE=True for multitask training.")

    distributed, rank, local_rank = _setup_distributed()
    val_data_dir = getattr(config.DATA, "val_data_dir", config.DATA.test_data_dir)
    if _is_main_process():
        print(f"Validation data dir: {val_data_dir}")
    train_loader, val_loader = get_loader(config, test_data_dir=val_data_dir)
    device = torch.device("cuda", local_rank) if distributed else torch.device("cuda")

    encoder = Mamba_encoder(config).to(device)
    decoder = Mamba_decoder(config).to(device)
    classifier = build_latent_classifier(config).to(device)
    stage = _configure_stage(config, encoder, decoder, classifier)

    if distributed:
        print(f"Using DDP rank {rank}, local_rank {local_rank}")
        if _has_trainable_params(encoder):
            encoder = DDP(encoder, device_ids=[local_rank], output_device=local_rank)
        if _has_trainable_params(decoder):
            decoder = DDP(decoder, device_ids=[local_rank], output_device=local_rank)
        if _has_trainable_params(classifier):
            classifier = DDP(classifier, device_ids=[local_rank], output_device=local_rank)

    channel = Channel(config)
    if _is_main_process():
        print(f"Multitask stage: {stage}")
        _print_multitask_model_profile(config, encoder, decoder, classifier, device)
    if distributed:
        dist.barrier()

    optimizers = _build_optimizers(config, encoder, decoder, classifier)
    schedulers = _build_schedulers(config, optimizers, train_loader)
    if _is_main_process():
        for name, optimizer in optimizers.items():
            _print_optimizer_groups(name, optimizer)

    criterion_rec = loss_matrix(config)
    criterion_cls = nn.CrossEntropyLoss()
    matrix = eval_matrix(config)
    log_dir = _get_log_dir(config)
    eval_fre = getattr(config.TRAIN, "EVAL_FRE", 10)
    records = []
    tri_weight_records = []

    seed_torch()
    for e in range(config.TRAIN.EPOCHS):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(e)
        _module_train_state(stage, encoder, decoder, classifier)

        total_sum = 0.0
        rec_sum = 0.0
        cls_sum = 0.0
        acc_sum = 0.0
        with tqdm(train_loader, dynamic_ncols=False, disable=not _is_main_process()) as tqdm_data:
            for i, (input_image, labels) in enumerate(tqdm_data):
                if not torch.is_tensor(labels):
                    raise ValueError("Multitask classification requires class labels.")
                snr_list = config.CHANNEL.SNR
                snr = snr_list[torch.randint(0, len(snr_list), (1,)).item()]

                input_image = input_image.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                input_image = _prepare_input(config, input_image)
                _zero_grad(optimizers)

                feature = encoder(input_image, snr)
                received, pwr, h = channel.forward(feature, snr)
                z_hat = _format_received(config, received, pwr, h, snr)
                recon_image = decoder(z_hat, snr)
                logits = classifier(z_hat, snr)

                rec_loss = criterion_rec(recon_image, input_image, feature, opt_idx=0, global_step=e)
                cls_loss = criterion_cls(logits, labels)
                if stage == "head_only":
                    loss = cls_loss
                else:
                    loss = (
                        config.TASK.REC_LOSS_WEIGHT * rec_loss
                        + config.TASK.CLS_LOSS_WEIGHT * cls_loss
                    )
                loss.backward()

                if getattr(config.MODEL.VSSM, "TRI_MERGE_DEBUG", False) and _is_main_process() and i == 0:
                    _inspect_tri_merge_grad(encoder, f"epoch {e} encoder")
                    _inspect_tri_merge_grad(decoder, f"epoch {e} decoder")

                _clip_trainable_grad(config, encoder, decoder, classifier)
                _optimizer_step(optimizers, schedulers)

                acc = (logits.argmax(dim=1) == labels).float().mean().item()
                total_sum += loss.item()
                rec_sum += rec_loss.item()
                cls_sum += cls_loss.item()
                acc_sum += acc
                performance = matrix(recon_image, input_image)
                tqdm_data.set_postfix(
                    {
                        "e": e,
                        "loss": (loss.item(), total_sum / (i + 1)),
                        "rec": rec_loss.item(),
                        "cls": cls_loss.item(),
                        "acc": acc,
                        "matrix": performance,
                        "SNR": snr,
                    }
                )

        n_batch = i + 1
        train_total = _reduce_scalar(total_sum / n_batch, device)
        train_rec = _reduce_scalar(rec_sum / n_batch, device)
        train_cls = _reduce_scalar(cls_sum / n_batch, device)
        train_acc = _reduce_scalar(acc_sum / n_batch, device)

        row = {
            "epoch": e + 1,
            "train_total_loss": train_total,
            "train_rec_loss": train_rec,
            "train_cls_loss": train_cls,
            "train_acc": train_acc,
            "val_total_loss": "",
            "val_acc": "",
        }
        if _is_main_process():
            tri_record = {"epoch": e + 1}
            tri_record.update(_collect_tri_path_weight_stats(encoder, "encoder"))
            tri_record.update(_collect_tri_path_weight_stats(decoder, "decoder"))
            tri_weight_records.append(tri_record)

        if _is_main_process() and (e + 1) % config.TRAIN.SAVE_FRE == 0:
            save_multitask_models(
                config,
                _unwrap_parallel(encoder),
                _unwrap_parallel(decoder),
                _unwrap_parallel(classifier),
            )

        run_validation = eval_fre > 0 and (
            (e + 1) % eval_fre == 0 or (e + 1) == config.TRAIN.EPOCHS
        )
        if run_validation:
            if distributed:
                dist.barrier()
            if _is_main_process():
                print(f"----------multitask validation after epoch {e + 1}----------")
                _, _, val_acc_curve, val_loss = eval_MambaJSCC_multitask_models(
                    config,
                    _unwrap_parallel(encoder),
                    _unwrap_parallel(decoder),
                    _unwrap_parallel(classifier),
                    test_loader=val_loader,
                    criterion_rec=criterion_rec,
                    criterion_cls=criterion_cls,
                    save_curves=(e + 1) == config.TRAIN.EPOCHS,
                    prefix=f"val_epoch_{e + 1:03d}_snr",
                )
                row["val_total_loss"] = val_loss if val_loss is not None else ""
                row["val_acc"] = sum(val_acc_curve) / len(val_acc_curve)
            if distributed:
                dist.barrier()

        if _is_main_process():
            records.append(row)

    if _is_main_process():
        save_multitask_models(
            config,
            _unwrap_parallel(encoder),
            _unwrap_parallel(decoder),
            _unwrap_parallel(classifier),
        )
        _save_multitask_curve(records, log_dir)
        _save_tri_path_weight_curve(tri_weight_records, log_dir)
    if distributed:
        dist.barrier()
