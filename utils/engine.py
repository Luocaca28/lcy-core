"""Shared training/evaluation engine for the DefMambaJSCC pipeline.

This module is the common foundation used by both independent tasks:

- reconstruction (``run/train.py`` + ``run/eval.py``)
- classification (``tasks/classification/``)

It deliberately depends only on low-level utilities (never on ``run`` or
``tasks``), so both tasks can build on top of it without importing each other.
It groups the plumbing that used to be duplicated across those files:
distributed helpers, optimizer/scheduler builders, the channel forward,
checkpoint naming, model profiling and the deformable tri-path weight probes.
"""

import csv
import gc
import os
from datetime import timedelta

import torch
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from utils.utils import GradualWarmupScheduler


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------
def setup_distributed():
    """Initialise the process group when launched with WORLD_SIZE > 1.

    Returns ``(distributed, rank, local_rank)``.
    """
    if "WORLD_SIZE" not in os.environ or int(os.environ["WORLD_SIZE"]) <= 1:
        return False, 0, 0
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    return True, dist.get_rank(), local_rank


def is_main_process():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def reduce_scalar(value, device):
    """Average a python scalar across all ranks (no-op when single process)."""
    if not (dist.is_available() and dist.is_initialized()):
        return value
    tensor = torch.tensor(value, dtype=torch.float32, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor = tensor / dist.get_world_size()
    return tensor.item()


def unwrap_parallel(model):
    """Return the underlying module behind a (Distributed)DataParallel wrapper."""
    return model.module if isinstance(model, (torch.nn.DataParallel, DDP)) else model


# ---------------------------------------------------------------------------
# Optimizer / scheduler builders
# ---------------------------------------------------------------------------
def _split_tri_merge_params(model):
    """Separate deformable tri-merge params (own LR) from the rest."""
    base_params, tri_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "tri_merge" in name:
            tri_params.append(param)
        else:
            base_params.append(param)
    return base_params, tri_params


def build_adamw_with_tri_merge(model, base_lr, weight_decay, tri_lr_mult):
    """AdamW with a dedicated param group (scaled LR, no decay) for tri-merge."""
    base_params, tri_params = _split_tri_merge_params(model)
    groups = [{"params": base_params, "lr": base_lr, "weight_decay": weight_decay}]
    if tri_params:
        groups.append(
            {
                "params": tri_params,
                "lr": base_lr * tri_lr_mult,
                "weight_decay": 0.0,
            }
        )
    return optim.AdamW(groups)


def print_optimizer_groups(name, optimizer):
    for idx, group in enumerate(optimizer.param_groups):
        print(
            f"{name} group {idx}: lr={group['lr']} "
            f"wd={group.get('weight_decay', 0.0)} n_params={len(group['params'])}"
        )


def build_warmup_cosine_schedulers(config, optimizers, train_loader):
    """Build a warmup -> cosine-annealing scheduler for each named optimizer.

    ``optimizers`` is a ``{name: optimizer}`` mapping; the returned schedulers
    use the same keys. Step counts mirror the original per-iteration schedule.
    """
    total_steps = max(1, config.TRAIN.EPOCHS * len(train_loader))
    warmup_steps = max(1, int(config.TRAIN.WARMUP_EPOCHS * len(train_loader)))
    cosine_steps = max(1, total_steps - warmup_steps)
    schedulers = {}
    for name, optimizer in optimizers.items():
        cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=cosine_steps, eta_min=0, last_epoch=-1
        )
        schedulers[name] = GradualWarmupScheduler(
            optimizer=optimizer,
            multiplier=2.0,
            warm_epoch=warmup_steps,
            after_scheduler=cosine,
        )
    return schedulers


# ---------------------------------------------------------------------------
# Channel forward
# ---------------------------------------------------------------------------
def apply_channel(channel, config, feature, snr):
    """Push an encoder feature through the channel and return the decoder input.

    Handles the Rayleigh inverse filter and the real/imag interleaving shared by
    every task. Returns the received latent ``z_hat`` of shape
    ``(B, 2*C, H, W)`` scaled by ``sqrt(power)``.
    """
    received, pwr, h = channel.forward(feature, snr)
    if config.CHANNEL.TYPE == "rayleigh":
        sigma_square = 1.0 / (10 ** (snr / 10))
        received = torch.conj(h) * received / (torch.abs(h) ** 2 + sigma_square)
    elif config.CHANNEL.TYPE == "awgn":
        pass
    else:
        raise ValueError("channel type error")
    return torch.cat((torch.real(received), torch.imag(received)), dim=2) * torch.sqrt(pwr)


# ---------------------------------------------------------------------------
# Logging / checkpoint naming
# ---------------------------------------------------------------------------
def get_log_dir(config):
    """Resolve (and create) the log directory for the current run."""
    log_path = getattr(config.TRAIN, "LOG_PATH", "")
    if log_path:
        os.makedirs(log_path, exist_ok=True)
        return log_path
    base = os.path.commonpath([config.TRAIN.ENCODER_PATH, config.TRAIN.DECODER_PATH])
    log_dir = os.path.join(base, "logs")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def checkpoint_tag(config):
    """Descriptive run identity used as the reconstruction checkpoint filename.

    Centralises the long format string that used to be repeated across the
    train/eval/tooling code so every call site stays in sync.
    """
    vssm = config.MODEL.VSSM
    return (
        "OUTCHANS{}_extent{}_loss{}_SCANnum{}_SNR{}_adp{}_type{}_depth{}_embed{}_nums{}_rsl{}".format(
            vssm.OUT_CHANS,
            vssm.Extent,
            config.TRAIN.LOSS,
            vssm.SCAN_NUMBER,
            config.CHANNEL.SNR,
            config.CHANNEL.ADAPTIVE,
            config.CHANNEL.TYPE,
            len(vssm.EMBED_DIM),
            vssm.EMBED_DIM,
            vssm.DEPTHS,
            config.DATA.IMG_SIZE,
        )
    )


# ---------------------------------------------------------------------------
# Deformable tri-path weight probes
# ---------------------------------------------------------------------------
def collect_tri_path_weight_stats(model, prefix):
    """Summarise the learned third-path (deformable) merge weights."""
    diag_values, offdiag_values, abs_values, base_abs_values = [], [], [], []
    for name, module in unwrap_parallel(model).named_modules():
        if "tri_merge" not in name or not hasattr(module, "proj"):
            continue
        weight = module.proj.weight.detach()
        out_channels, in_channels, kernel = weight.shape
        if kernel != 1 or in_channels != 3 * out_channels:
            continue
        c = out_channels
        w0 = weight[:, 0 * c : 1 * c, 0]
        w1 = weight[:, 1 * c : 2 * c, 0]
        wd = weight[:, 2 * c : 3 * c, 0]
        diag_values.append(torch.diag(wd).mean())
        offdiag_values.append((wd - torch.diag(torch.diag(wd))).abs().mean())
        abs_values.append(wd.abs().mean())
        base_abs_values.append(0.5 * (w0.abs().mean() + w1.abs().mean()))

    if not abs_values:
        return {
            f"{prefix}_wd_diag_mean": "",
            f"{prefix}_wd_offdiag_abs_mean": "",
            f"{prefix}_wd_abs_mean": "",
            f"{prefix}_wd_to_base_abs_ratio": "",
        }

    wd_diag_mean = torch.stack(diag_values).mean().item()
    wd_offdiag_mean = torch.stack(offdiag_values).mean().item()
    wd_abs_mean = torch.stack(abs_values).mean().item()
    base_abs_mean = torch.stack(base_abs_values).mean().item()
    return {
        f"{prefix}_wd_diag_mean": wd_diag_mean,
        f"{prefix}_wd_offdiag_abs_mean": wd_offdiag_mean,
        f"{prefix}_wd_abs_mean": wd_abs_mean,
        f"{prefix}_wd_to_base_abs_ratio": wd_abs_mean / (base_abs_mean + 1e-12),
    }


def save_tri_path_weight_curve(records, log_dir):
    if not records:
        return
    csv_path = os.path.join(log_dir, "tri_path_weight_curve.csv")
    png_path = os.path.join(log_dir, "tri_path_weight_curve.png")
    fieldnames = list(records[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = [row["epoch"] for row in records]
        plt.figure()
        for key in fieldnames:
            if key == "epoch":
                continue
            values = [row[key] for row in records]
            if any(value == "" for value in values):
                continue
            plt.plot(epochs, values, marker="o", label=key)
        plt.xlabel("Epoch")
        plt.ylabel("Tri-path weight statistic")
        plt.title("Third Path Weight Curve")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(png_path, dpi=200)
        plt.close()
    except Exception as exc:
        print(f"Failed to save tri-path weight plot: {exc}")


def inspect_tri_merge_grad(model, tag):
    for name, param in model.named_parameters():
        if "tri_merge" not in name:
            continue
        grad = None if param.grad is None else param.grad.detach().abs().mean().item()
        print(f"[{tag}] {name}: grad_mean={grad}")


# ---------------------------------------------------------------------------
# Parameter / FLOPs profiling
# ---------------------------------------------------------------------------
def format_count(value):
    if value is None:
        return "N/A"
    if value >= 1e12:
        return f"{value / 1e12:.3f} T"
    if value >= 1e9:
        return f"{value / 1e9:.3f} G"
    if value >= 1e6:
        return f"{value / 1e6:.3f} M"
    if value >= 1e3:
        return f"{value / 1e3:.3f} K"
    return str(value)


def format_count_with_commas(value):
    return "N/A" if value is None else f"{value:,}"


def count_params(model):
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def param_memory_mb(model):
    return sum(param.numel() * param.element_size() for param in model.parameters()) / (1024 ** 2)


class _SNRForward(torch.nn.Module):
    """Adapter that pins the SNR argument so FLOPs counters see a single input."""

    def __init__(self, model, snr):
        super().__init__()
        self.model = model
        self.snr = snr

    def forward(self, x):
        return self.model(x, self.snr)


def flops_of(model, input_tensor, snr):
    try:
        from fvcore.nn import FlopCountAnalysis

        wrapped = _SNRForward(model, snr)
        flops = FlopCountAnalysis(wrapped, input_tensor)
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        total = flops.total()
        unsupported = dict(flops.unsupported_ops())
        del flops, wrapped
        gc.collect()
        if input_tensor.is_cuda:
            torch.cuda.empty_cache()
        return total, unsupported, None
    except Exception as exc:
        gc.collect()
        if input_tensor.is_cuda:
            torch.cuda.empty_cache()
        return None, {}, str(exc)


def print_pipeline_profile(config, named_modules, device):
    """Report params / memory / FLOPs for a sequential module pipeline.

    ``named_modules`` is a list of ``(label, module)`` executed in order: the
    first module is fed a random image and every following module is fed the
    previous module's output. This covers both the reconstruction pipeline
    (encoder -> decoder) and the classification pipeline (encoder -> classifier).
    """
    named_modules = [(label, unwrap_parallel(module)) for label, module in named_modules]
    snr_list = config.CHANNEL.SNR
    profile_snr = snr_list[0] if isinstance(snr_list, (list, tuple)) else snr_list
    image_size = config.DATA.IMG_SIZE

    previous_training = [module.training for _, module in named_modules]
    for _, module in named_modules:
        module.eval()

    tensors_to_free = []
    try:
        params, mems, flops = {}, {}, {}
        current_input = torch.randn(1, 3, image_size, image_size, device=device)
        tensors_to_free.append(current_input)

        last_index = len(named_modules) - 1
        for idx, (label, module) in enumerate(named_modules):
            total_params, _ = count_params(module)
            params[label] = total_params
            mems[label] = param_memory_mb(module)
            module_flops, _, error = flops_of(module, current_input, profile_snr)
            flops[label] = module_flops
            if error:
                print(f"{label} FLOPs failed: {error}")
            # Feed the next stage with this module's output. The final stage is
            # never executed here (its true runtime input shape may differ from
            # the encoder feature, mirroring the original profiling behaviour).
            if idx < last_index:
                with torch.no_grad():
                    current_input = module(current_input, profile_snr)
                tensors_to_free.append(current_input)

        total_params = sum(params.values())
        total_mem = sum(mems.values())
        forward_flops = None
        if all(value is not None for value in flops.values()):
            forward_flops = sum(flops.values())

        for label, _ in named_modules:
            print(
                f"{label} params: {format_count(params[label])} "
                f"({format_count_with_commas(params[label])})"
            )
        print(f"Total params:   {format_count(total_params)} ({format_count_with_commas(total_params)})")
        for label, _ in named_modules:
            print(f"{label} size: {mems[label]:.2f} MB")
        print(f"Total size:   {total_mem:.2f} MB")
        for label, _ in named_modules:
            print(f"{label} FLOPs: {format_count(flops[label])}")
        print(f"Forward FLOPs: {format_count(forward_flops)}")
        if forward_flops is not None:
            print(f"Train FLOPs/image (fwd+bwd est.): {format_count(3 * forward_flops)}")
            print(
                "Train FLOPs/batch "
                f"(batch={config.DATA.TRAIN_BATCH}, fwd+bwd est.): "
                f"{format_count(3 * forward_flops * config.DATA.TRAIN_BATCH)}"
            )
    finally:
        for tensor in tensors_to_free:
            del tensor
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        for (_, module), was_training in zip(named_modules, previous_training):
            if was_training:
                module.train()
