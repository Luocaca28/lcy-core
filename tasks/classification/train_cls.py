"""Classification task: training entry point.

This is one of the two independent tasks. It trains the encoder + a latent
classifier on top of the shared Mamba backbone over a noisy channel. All shared
plumbing comes from ``utils.engine`` -- this task does not import from ``run``.

Stages (``CLS.STAGE``) -- both train the encoder and the classifier head:

- ``from_scratch``     : train encoder + head jointly from random init.
- ``finetune_encoder`` : warm-start the encoder from a reconstruction checkpoint
                          (``CLS.PRETRAIN_ENCODER``) and fine-tune it with the head.
"""

import csv
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from data.datasets import get_loader, TransformSubset
from models.channel import Channel
from models.network import Mamba_encoder
from tasks.classification.eval_cls import (
    build_latent_classifier,
    evaluate_classification,
    save_classification_models,
    _prepare_input,
)
from utils.engine import (
    apply_channel,
    apply_channel_compact,
    build_adamw_with_tri_merge,
    build_warmup_cosine_schedulers,
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
from utils.utils import seed_torch


STAGES = ("from_scratch", "finetune_encoder")


def _resolve_stage(config):
    stage = getattr(config.CLS, "STAGE", "from_scratch").lower()
    if stage not in STAGES:
        raise ValueError(f"Unknown CLS.STAGE: {config.CLS.STAGE}. Use one of {STAGES}.")
    return stage


def _load_pretrained_module(path, model, device):
    if not path:
        return model
    loaded = torch.load(path, map_location=device, weights_only=False)
    if isinstance(loaded, dict):
        model.load_state_dict(loaded, strict=False)
        return model
    return loaded.to(device)


def _maybe_load_pretrained(config, stage, encoder, classifier, device):
    if stage == "finetune_encoder":
        encoder = _load_pretrained_module(
            getattr(config.CLS, "PRETRAIN_ENCODER", ""), encoder, device
        )
    elif getattr(config.CLS, "PRETRAIN_ENCODER", ""):
        print("Ignoring CLS.PRETRAIN_ENCODER because CLS.STAGE is from_scratch.")
    classifier = _load_pretrained_module(
        getattr(config.CLS, "PRETRAIN_CLASSIFIER", ""), classifier, device
    )
    return encoder, classifier


def _build_optimizers(config, encoder, classifier):
    tri_lr_mult = getattr(config.MODEL.VSSM, "TRI_MERGE_LR_MULT", 1.0)
    return {
        "encoder": build_adamw_with_tri_merge(
            encoder, config.TRAIN.BASE_LR, config.TRAIN.WEIGHT_DECAY, tri_lr_mult
        ),
        "classifier": optim.AdamW(
            classifier.parameters(),
            lr=config.CLS.HEAD_LR,
            weight_decay=config.CLS.HEAD_WEIGHT_DECAY,
        ),
    }


def _zero_grad(optimizers):
    for optimizer in optimizers.values():
        optimizer.zero_grad()


def _optimizer_step(optimizers, schedulers):
    for name, optimizer in optimizers.items():
        optimizer.step()
        schedulers[name].step()


def _clip_grad(config, *models):
    for model in models:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)


def _save_classification_curve(records, log_dir):
    csv_path = os.path.join(log_dir, "classification_curve.csv")
    fieldnames = [
        "epoch", "train_cls_loss", "train_acc", "val_cls_loss", "val_acc",
        "clean_train_cls_loss", "clean_train_acc",
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
        train_losses = [row["train_cls_loss"] for row in records]
        val_losses = [row["val_cls_loss"] for row in records if row["val_cls_loss"] != ""]
        val_epochs = [row["epoch"] for row in records if row["val_cls_loss"] != ""]
        train_acc = [row["train_acc"] for row in records]
        val_acc = [row["val_acc"] for row in records if row["val_acc"] != ""]
        ct_loss = [row["clean_train_cls_loss"] for row in records if row.get("clean_train_cls_loss", "") != ""]
        ct_loss_epochs = [row["epoch"] for row in records if row.get("clean_train_cls_loss", "") != ""]
        ct_acc = [row["clean_train_acc"] for row in records if row.get("clean_train_acc", "") != ""]
        ct_acc_epochs = [row["epoch"] for row in records if row.get("clean_train_acc", "") != ""]

        plt.figure()
        plt.plot(epochs, train_losses, marker="o", label="train cls loss (mixup)")
        if ct_loss:
            plt.plot(ct_loss_epochs, ct_loss, marker="^", label="clean train cls loss")
        if val_losses:
            plt.plot(val_epochs, val_losses, marker="s", label="val cls loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross entropy")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, "classification_loss_curve.png"), dpi=200)
        plt.close()

        plt.figure()
        plt.plot(epochs, train_acc, marker="o", label="train acc (mixup)")
        if ct_acc:
            plt.plot(ct_acc_epochs, ct_acc, marker="^", label="clean train acc")
        if val_acc:
            plt.plot(val_epochs, val_acc, marker="s", label="val acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, "classification_acc_curve.png"), dpi=200)
        plt.close()
    except Exception as exc:
        print(f"Failed to save classification curves: {exc}")


def train_classification(config):
    distributed, rank, local_rank = setup_distributed()
    val_data_dir = getattr(config.DATA, "val_data_dir", config.DATA.test_data_dir)
    if is_main_process():
        print(f"Validation data dir: {val_data_dir}")
    train_loader, val_loader = get_loader(config, test_data_dir=val_data_dir)
    device = torch.device("cuda", local_rank) if distributed else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    stage = _resolve_stage(config)
    encoder = Mamba_encoder(config).to(device)
    classifier = build_latent_classifier(config).to(device)
    encoder, classifier = _maybe_load_pretrained(config, stage, encoder, classifier, device)

    if distributed:
        print(f"Using DDP rank {rank}, local_rank {local_rank}")
        encoder = DDP(encoder, device_ids=[local_rank], output_device=local_rank)
        classifier = DDP(classifier, device_ids=[local_rank], output_device=local_rank)

    channel = Channel(config)
    if is_main_process():
        print(f"Classification stage: {stage}")
        print(f"PRETRAIN_ENCODER: {getattr(config.CLS, 'PRETRAIN_ENCODER', '')}")
        print(f"PRETRAIN_CLASSIFIER: {getattr(config.CLS, 'PRETRAIN_CLASSIFIER', '')}")
        print("Loss: L_cls = CrossEntropy(logits, labels)")
        print(f"HEAD_LR: {config.CLS.HEAD_LR}")
        print_pipeline_profile(config, [("Encoder", encoder), ("Classifier", classifier)], device)
    if distributed:
        dist.barrier()

    optimizers = _build_optimizers(config, encoder, classifier)
    schedulers = build_warmup_cosine_schedulers(config, optimizers, train_loader)
    if is_main_process():
        for name, optimizer in optimizers.items():
            print_optimizer_groups(name, optimizer)

    criterion_cls = nn.CrossEntropyLoss(
        label_smoothing=float(getattr(config.CLS, "LABEL_SMOOTHING", 0.0))
    )
    mixup_alpha = float(getattr(config.CLS, "MIXUP_ALPHA", 0.0))
    mixup_dist = (
        torch.distributions.Beta(mixup_alpha, mixup_alpha) if mixup_alpha > 0 else None
    )
    if is_main_process() and mixup_dist is not None:
        print(f"Mixup enabled: alpha={mixup_alpha}")
    log_dir = get_log_dir(config)
    eval_fre = getattr(config.TRAIN, "EVAL_FRE", 10)
    records = []
    tri_weight_records = []

    # Clean-train eval loader: train-split images with the val (clean, un-mixed,
    # un-augmented) transform, capped to ~val size for speed. Evaluated exactly
    # like val, so clean-train vs val accuracy are on the SAME footing and their
    # gap is the TRUE generalization gap. (The in-loop train_acc is computed on
    # Mixup'd images and is not comparable to val -- do not read overfitting off it.)
    train_snr = getattr(config.CHANNEL, "TRAIN_SNR", None)
    if is_main_process() and train_snr is not None:
        print(f"Single-SNR training at SNR={train_snr} dB; eval sweeps {config.CHANNEL.SNR}")

    compact_code = getattr(config.CHANNEL, "COMPACT_CODE", False)
    if is_main_process() and compact_code:
        print("Compact-code transmission: channel acts on the pooled code (SNR-sensitive).")

    clean_train_loader = None
    _tr_ds = getattr(train_loader, "dataset", None)
    _va_ds = getattr(val_loader, "dataset", None)
    if isinstance(_tr_ds, TransformSubset) and isinstance(_va_ds, TransformSubset):
        _n = min(len(_va_ds), len(_tr_ds.indices))
        clean_train_loader = torch.utils.data.DataLoader(
            TransformSubset(_tr_ds.base_dataset, list(_tr_ds.indices)[:_n], _va_ds.transform),
            batch_size=config.DATA.TEST_BATCH,
            shuffle=False,
        )

    seed_torch()
    for e in range(config.TRAIN.EPOCHS):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(e)
        encoder.train()
        classifier.train()

        cls_sum = 0.0
        correct = 0
        total = 0
        with tqdm(train_loader, dynamic_ncols=False, disable=not is_main_process()) as tqdm_data:
            for i, (input_image, labels) in enumerate(tqdm_data):
                if not torch.is_tensor(labels):
                    raise ValueError("Classification requires class labels.")
                snr_list = config.CHANNEL.SNR
                if train_snr is not None:
                    snr = train_snr
                else:
                    snr = snr_list[torch.randint(0, len(snr_list), (1,)).item()]
                # Channel sees the true SNR; the model may be fed a fixed SNR
                # (BLIND_MODEL) so it cannot adapt to the channel -- this is what
                # makes accuracy rise with SNR (the mismatched-CSI ablation).
                model_snr = (
                    float(config.CHANNEL.MODEL_SNR)
                    if getattr(config.CHANNEL, "BLIND_MODEL", False)
                    else snr
                )

                input_image = input_image.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                input_image = _prepare_input(config, input_image)
                _zero_grad(optimizers)

                if mixup_dist is not None:
                    # Mixup: blend each image with another in the batch and train
                    # on the convex combination of both targets. Strongly discourages
                    # memorizing individual samples.
                    lam = mixup_dist.sample().item()
                    perm = torch.randperm(input_image.size(0), device=device)
                    input_image = lam * input_image + (1.0 - lam) * input_image[perm]
                    labels_b = labels[perm]
                    feature = encoder(input_image, model_snr)
                    if compact_code:
                        z_hat = apply_channel_compact(channel, config, feature, snr)
                    else:
                        z_hat = apply_channel(channel, config, feature, snr)
                    logits = classifier(z_hat, model_snr)
                    cls_loss = lam * criterion_cls(logits, labels) + (1.0 - lam) * criterion_cls(
                        logits, labels_b
                    )
                else:
                    feature = encoder(input_image, model_snr)
                    if compact_code:
                        z_hat = apply_channel_compact(channel, config, feature, snr)
                    else:
                        z_hat = apply_channel(channel, config, feature, snr)
                    logits = classifier(z_hat, model_snr)
                    cls_loss = criterion_cls(logits, labels)
                cls_loss.backward()

                if getattr(config.MODEL.VSSM, "TRI_MERGE_DEBUG", False) and is_main_process() and i == 0:
                    inspect_tri_merge_grad(encoder, f"epoch {e} encoder")

                _clip_grad(config, encoder, classifier)
                _optimizer_step(optimizers, schedulers)

                pred = logits.argmax(dim=1)
                batch_correct = (pred == labels).sum().item()
                batch_total = labels.numel()
                acc = batch_correct / max(batch_total, 1)
                cls_sum += cls_loss.item()
                correct += batch_correct
                total += batch_total
                tqdm_data.set_postfix({"e": e, "cls": cls_loss.item(), "acc": acc, "SNR": snr})

        n_batch = i + 1
        train_cls = reduce_scalar(cls_sum / n_batch, device)
        train_acc = reduce_scalar(correct / max(total, 1), device)

        row = {
            "epoch": e + 1,
            "train_cls_loss": train_cls,
            "train_acc": train_acc,
            "val_cls_loss": "",
            "val_acc": "",
            "clean_train_cls_loss": "",
            "clean_train_acc": "",
        }
        if is_main_process():
            tri_record = {"epoch": e + 1}
            tri_record.update(collect_tri_path_weight_stats(encoder, "encoder"))
            tri_weight_records.append(tri_record)

        if is_main_process() and (e + 1) % config.TRAIN.SAVE_FRE == 0:
            save_classification_models(
                config, unwrap_parallel(encoder), unwrap_parallel(classifier)
            )

        run_validation = eval_fre > 0 and (
            (e + 1) % eval_fre == 0 or (e + 1) == config.TRAIN.EPOCHS
        )
        if run_validation:
            if distributed:
                dist.barrier()
            if is_main_process():
                print(f"----------classification validation after epoch {e + 1}----------")
                val_acc_curve, val_loss = evaluate_classification(
                    config,
                    unwrap_parallel(encoder),
                    unwrap_parallel(classifier),
                    test_loader=val_loader,
                    criterion_cls=criterion_cls,
                    save_curves=(e + 1) == config.TRAIN.EPOCHS,
                    prefix=f"val_epoch_{e + 1:03d}_snr",
                )
                row["val_cls_loss"] = val_loss if val_loss is not None else ""
                row["val_acc"] = sum(val_acc_curve) / len(val_acc_curve)
                if clean_train_loader is not None:
                    print("---- clean-train accuracy (un-mixed, for overfit gap) ----")
                    ct_curve, ct_loss = evaluate_classification(
                        config,
                        unwrap_parallel(encoder),
                        unwrap_parallel(classifier),
                        test_loader=clean_train_loader,
                        criterion_cls=criterion_cls,
                        save_curves=False,
                        prefix="clean_train",
                    )
                    row["clean_train_acc"] = sum(ct_curve) / len(ct_curve)
                    row["clean_train_cls_loss"] = ct_loss if ct_loss is not None else ""
            if distributed:
                dist.barrier()

        if is_main_process():
            records.append(row)

    if is_main_process():
        save_classification_models(
            config, unwrap_parallel(encoder), unwrap_parallel(classifier)
        )
        _save_classification_curve(records, log_dir)
        save_tri_path_weight_curve(tri_weight_records, log_dir)
    if distributed:
        dist.barrier()
