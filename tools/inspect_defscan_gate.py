import argparse
import os
import sys
from types import SimpleNamespace

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.config import get_config
from data.datasets import get_loader


def _checkpoint_stem(config):
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


def _load_config(config_name, project_path):
    args = SimpleNamespace(
        config_name=config_name,
        project_path=project_path,
        mode="test",
        model_config_path=os.path.join(
            project_path, "configs", "vssm", f"vssm_tiny_{config_name}.yaml"
        ),
        train_config_path=os.path.join(
            project_path, "configs", "train", f"vssm_tiny_{config_name}.yaml"
        ),
    )
    return get_config(args)


def _set_gate_mode(model, mode):
    for module in model.modules():
        if hasattr(module, "defscan_gate_mode"):
            module.defscan_gate_mode = mode


def _tensor_stats(t):
    t = t.detach()
    return {
        "mean": t.mean().item(),
        "std": t.std().item(),
        "min": t.min().item(),
        "max": t.max().item(),
        "finite": torch.isfinite(t).all().item(),
    }


def _gate_input_stats(y_base, y_def, snr):
    B, C, _ = y_base.shape
    dtype = y_base.dtype
    device = y_base.device

    if not torch.is_tensor(snr):
        snr = torch.tensor(snr, device=device, dtype=dtype)
    else:
        snr = snr.to(device=device, dtype=dtype)
    if snr.dim() == 0 or snr.numel() == 1:
        snr = snr.view(1).expand(B)
    else:
        snr = snr.view(B)
    snr_norm = (snr / 20.0).view(B, 1).expand(B, C)

    y_base_stat = y_base.detach()
    y_def_stat = y_def.detach()
    base_abs = y_base_stat.abs().mean(dim=2)
    def_abs = y_def_stat.abs().mean(dim=2)
    diff_abs = (y_def_stat - y_base_stat).abs().mean(dim=2)
    scale = base_abs + def_abs + 1e-6

    base_stat = base_abs / scale
    def_stat = def_abs / scale
    diff_stat = diff_abs / scale
    corr_stat = torch.nn.functional.cosine_similarity(
        y_base_stat, y_def_stat, dim=2, eps=1e-6
    )
    corr_stat = torch.nan_to_num((corr_stat + 1.0) * 0.5, nan=0.5, posinf=1.0, neginf=0.0)

    return {
        "base_stat": base_stat,
        "def_stat": def_stat,
        "diff_stat": diff_stat,
        "corr_stat": corr_stat,
        "snr_norm": snr_norm,
    }


def _summarize(records, input_records, snr):
    if not records:
        print(f"SNR {snr}: no gate outputs captured")
        return

    mean = sum(r["mean"] for r in records) / len(records)
    std = sum(r["std"] for r in records) / len(records)
    channel_std = sum(r["channel_std"] for r in records) / len(records)
    sat_low = sum(r["sat_low"] for r in records) / len(records)
    sat_high = sum(r["sat_high"] for r in records) / len(records)
    mid_ratio = sum(r["mid_ratio"] for r in records) / len(records)
    sample_means = torch.tensor([v for r in records for v in r["sample_means"]])
    sample_std = sample_means.std().item() if sample_means.numel() > 1 else 0.0

    print(f"\nSNR {snr}")
    print("num_gate_calls:", len(records))
    print("first_shape:", records[0]["shape"])
    print("gate_mean:", mean)
    print("gate_std:", std)
    print("gate_min:", min(r["min"] for r in records))
    print("gate_max:", max(r["max"] for r in records))
    print("channel_std:", channel_std)
    print("sample_std:", sample_std)
    print("sat_low:", sat_low)
    print("sat_high:", sat_high)
    print("mid_ratio:", mid_ratio)

    print("input_stats:")
    for key in ["base_stat", "def_stat", "diff_stat", "corr_stat", "snr_norm"]:
        vals = [item[key] for item in input_records]
        print(
            " ",
            key,
            "mean", sum(v["mean"] for v in vals) / len(vals),
            "std", sum(v["std"] for v in vals) / len(vals),
            "min", min(v["min"] for v in vals),
            "max", max(v["max"] for v in vals),
            "finite", all(v["finite"] for v in vals),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", required=True)
    parser.add_argument("--project_path", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--snrs", default="1,5,10,20")
    parser.add_argument("--num_batches", type=int, default=4)
    parser.add_argument("--gate_mode", default="learned", choices=["learned", "zero", "one", "mean"])
    parser.add_argument("--random", action="store_true")
    args = parser.parse_args()

    config = _load_config(args.config_name, os.path.abspath(args.project_path))
    device = torch.device(args.device)
    checkpoint = args.checkpoint
    if not checkpoint:
        checkpoint = os.path.join(config.TRAIN.ENCODER_PATH, _checkpoint_stem(config) + ".pt")

    model = torch.load(checkpoint, map_location=device, weights_only=False).to(device).eval()
    _set_gate_mode(model, args.gate_mode)

    hooks = []
    records = []
    input_records = []

    def make_hook(name):
        def hook(module, inp, out):
            gate = out.detach()
            y_base, y_def, snr = inp[:3]
            stats = _gate_input_stats(y_base, y_def, snr)
            input_records.append({k: _tensor_stats(v) for k, v in stats.items()})
            records.append(
                {
                    "module": name,
                    "shape": tuple(gate.shape),
                    "mean": gate.mean().item(),
                    "std": gate.std().item(),
                    "min": gate.min().item(),
                    "max": gate.max().item(),
                    "channel_std": gate.squeeze(-1).std(dim=1).mean().item(),
                    "sat_low": (gate < 0.05).float().mean().item(),
                    "sat_high": (gate > 0.95).float().mean().item(),
                    "mid_ratio": ((gate > 0.1) & (gate < 0.9)).float().mean().item(),
                    "sample_means": gate.squeeze(-1).mean(dim=1).detach().cpu().tolist(),
                }
            )
        return hook

    for name, module in model.named_modules():
        if name.endswith("defscan_gate"):
            hooks.append(module.register_forward_hook(make_hook(name)))

    snrs = [float(x) for x in args.snrs.split(",") if x]
    if args.random:
        batches = [(torch.randn(1, 3, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE), None)]
    else:
        _, loader = get_loader(config)
        batches = []
        for idx, batch in enumerate(loader):
            if idx >= args.num_batches:
                break
            batches.append(batch)

    with torch.no_grad():
        for snr in snrs:
            records.clear()
            input_records.clear()
            for batch in batches:
                x = batch[0].to(device)
                model(x, snr)
            _summarize(records, input_records, snr)

    for hook in hooks:
        hook.remove()


if __name__ == "__main__":
    main()
