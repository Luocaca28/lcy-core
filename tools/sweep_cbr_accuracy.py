"""CBR x SNR sweep -> rate-accuracy curves for the classification task.

The classification task (``tasks/classification/``) already sweeps SNR inside
``evaluate_classification``. The *missing* axis for a task-oriented-communication
paper is CBR (channel bandwidth ratio), which is set by ``MODEL.VSSM.OUT_CHANS``
-- an architectural knob, so each CBR needs its OWN trained encoder+head. This
driver loops over a list of OUT_CHANS values, trains (or just loads) one model
per CBR, runs the existing SNR sweep for each, and aggregates the results into a
2D rate-accuracy table + two plots.

It ADDS a new file only: it imports and reuses the classification task and the
shared engine untouched (same spirit as tasks/classification_pure/). Delete this
file to remove the experiment; nothing else references it.

Methodology (keep the numbers honest):
- Only ``OUT_CHANS`` changes across the sweep. Seed / epochs / aug / LR / SNR
  list are held identical (they come from the SAME base config), so the accuracy
  difference between CBRs is attributable to code rate alone.
- Each CBR trains for the same number of epochs. If a low CBR underfits, say so
  in the figure caption -- do not silently drop points.
- This is the "encoder optimized FOR the task" curve. A method-A baseline
  (reconstructed image -> frozen classifier) is a SEPARATE curve; note that
  feeding a clean-trained classifier reconstructed images adds a distribution-
  shift confound (fine-tune that classifier on reconstructions first).

Run (on the Linux+GPU server -- the CUDA core is a hard dependency):

    # train one model per CBR, then aggregate
    python tools/sweep_cbr_accuracy.py --mode train \
        --config tasks/classification/configs/CIFAR10_cls_finetune_encoder.yaml \
        --out_chans_list "8,16,32,48,64" \
        --out_dir /home/LYC/lcy/MambaJSCCcheckpoints/CBR_SWEEP/cifar10

    # checkpoints already exist -> aggregate only
    python tools/sweep_cbr_accuracy.py --mode eval \
        --config tasks/classification/configs/CIFAR10_cls_finetune_encoder.yaml \
        --out_chans_list "8,16,32,48,64" \
        --out_dir /home/LYC/lcy/MambaJSCCcheckpoints/CBR_SWEEP/cifar10
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    p = argparse.ArgumentParser(description="CBR x SNR rate-accuracy sweep (classification).")
    p.add_argument(
        "--config",
        default=str(REPO_ROOT / "tasks" / "classification" / "configs" / "CIFAR10_cls_finetune_encoder.yaml"),
        help="Classification task config yaml (same one main_cls.py takes).",
    )
    p.add_argument(
        "--model_config",
        default=str(REPO_ROOT / "configs" / "vssm" / "vssm_tiny_CIFAR10_multitask_defscan.yaml"),
        help="Shared backbone model config yaml.",
    )
    p.add_argument(
        "--out_chans_list",
        default="8,16,32,48,64",
        help="Comma-separated OUT_CHANS (CBR) sweep points.",
    )
    p.add_argument("--mode", default="train", choices=["train", "eval"],
                   help="train: train one model per CBR then aggregate. eval: load existing checkpoints and aggregate only.")
    p.add_argument("--out_dir", required=True, help="Directory for aggregated rate_accuracy.csv + plots and per-CBR run products.")
    p.add_argument("--stage", default=None, choices=["finetune_encoder", "from_scratch"],
                   help="Optional override for CLS.STAGE (applied to every CBR run).")
    p.add_argument("--pretrain_encoder", default=None, help="Optional override for CLS.PRETRAIN_ENCODER.")
    args = p.parse_args()
    args.model_config_path = os.path.abspath(args.model_config)
    args.train_config_path = os.path.abspath(args.config)
    return args


def _per_cbr_dir(out_dir, oc, kind):
    """Isolated per-CBR product directory (trailing sep, created on demand)."""
    path = os.path.join(os.path.abspath(out_dir), f"cbr{oc}", kind)
    os.makedirs(path, exist_ok=True)
    return path + os.sep


def _build_cbr_config(base_args, oc, out_dir, stage, pretrain_encoder):
    """Clone the base config, pin OUT_CHANS=oc, and redirect all run products to
    a per-CBR subtree so different CBRs never overwrite each other's checkpoints
    or curves (checkpoint_tag already encodes OUT_CHANS, but LOG_PATH does not)."""
    from configs.config import get_config

    cfg = get_config(base_args).clone()
    cfg.defrost()
    cfg.MODEL.VSSM.OUT_CHANS = int(oc)
    cfg.TRAIN.LOG_PATH = _per_cbr_dir(out_dir, oc, "logs")
    cfg.CLS.ENCODER_PATH = _per_cbr_dir(out_dir, oc, "cls_encoder")
    cfg.CLS.CLASSIFIER_PATH = _per_cbr_dir(out_dir, oc, "classifier")
    if stage:
        cfg.CLS.STAGE = stage
    if pretrain_encoder is not None:
        cfg.CLS.PRETRAIN_ENCODER = pretrain_encoder
    cfg.freeze()
    return cfg


def _load_cbr_models(cfg, device):
    from tasks.classification.eval_cls import (
        classification_checkpoint_name,
        classifier_save_dir,
        encoder_save_dir,
    )

    name = classification_checkpoint_name(cfg) + ".pt"
    encoder = torch.load(os.path.join(encoder_save_dir(cfg), name),
                         weights_only=False, map_location=device).to(device)
    classifier = torch.load(os.path.join(classifier_save_dir(cfg), name),
                            weights_only=False, map_location=device).to(device)
    return encoder, classifier


@torch.no_grad()
def _measure_cbr(cfg, encoder, test_loader, device):
    """Exact channel-bandwidth ratio from tensor shapes: apply_channel packs the
    real encoder output into complex symbols (cat(real, imag) -> /2 channel uses)
    over 3*H*W source pixels. Batch dim cancels between numels."""
    from tasks.classification.eval_cls import _prepare_input

    encoder.eval()
    input_image, _ = next(iter(test_loader))
    input_image = _prepare_input(cfg, input_image.to(device))
    snr0 = (cfg.CHANNEL.EVAL_SNR or cfg.CHANNEL.SNR)[0]
    feature = encoder(input_image, snr0)
    return (feature.numel() / 2.0) / input_image.numel()


def _run_one_cbr(base_args, oc, device):
    from tasks.classification.eval_cls import evaluate_classification
    from tasks.classification.train_cls import train_classification
    from data.datasets import get_loader
    from utils.utils import seed_torch

    cfg = _build_cbr_config(base_args, oc, base_args.out_dir, base_args.stage, base_args.pretrain_encoder)
    print(f"\n===== CBR sweep: OUT_CHANS={oc} (stage={cfg.CLS.STAGE}) =====")

    if base_args.mode == "train":
        seed_torch()
        train_classification(cfg)

    try:
        encoder, classifier = _load_cbr_models(cfg, device)
    except FileNotFoundError as exc:
        print(f"[skip OUT_CHANS={oc}] checkpoint missing: {exc}")
        return None

    _, test_loader = get_loader(cfg)
    cbr = _measure_cbr(cfg, encoder, test_loader, device)
    snr_list = list(cfg.CHANNEL.EVAL_SNR or cfg.CHANNEL.SNR)
    seed_torch()
    acc_all, _ = evaluate_classification(cfg, encoder, classifier, test_loader=test_loader, save_curves=False)
    print(f"OUT_CHANS={oc}  CBR={cbr:.5f}  SNRs={snr_list}  Acc={acc_all}")
    return {"out_chans": int(oc), "cbr": cbr, "snr_list": snr_list, "acc": acc_all}


def _write_csv(results, out_dir):
    csv_path = os.path.join(out_dir, "rate_accuracy.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["out_chans", "cbr", "snr", "acc"])
        for r in results:
            for snr, acc in zip(r["snr_list"], r["acc"]):
                writer.writerow([r["out_chans"], f"{r['cbr']:.6f}", snr, f"{acc:.6f}"])
    print(f"Wrote {csv_path}")


def _plot(results, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib unavailable, skipping plots: {exc}")
        return

    # accuracy vs SNR, one line per CBR
    plt.figure()
    for r in sorted(results, key=lambda x: x["cbr"]):
        plt.plot(r["snr_list"], r["acc"], marker="o", label=f"CBR={r['cbr']:.4f} (C={r['out_chans']})")
    plt.xlabel("SNR (dB)")
    plt.ylabel("acc")
    plt.title("Rate-accuracy: accuracy vs SNR")
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.legend(fontsize="small")
    plt.tight_layout()
    p1 = os.path.join(out_dir, "rate_accuracy_vs_snr.png")
    plt.savefig(p1, dpi=200)
    plt.close()
    print(f"Wrote {p1}")

    # accuracy vs CBR, one line per fixed SNR (classic rate-accuracy plot)
    all_snrs = sorted({s for r in results for s in r["snr_list"]})
    ordered = sorted(results, key=lambda x: x["cbr"])
    plt.figure()
    for snr in all_snrs:
        xs, ys = [], []
        for r in ordered:
            if snr in r["snr_list"]:
                xs.append(r["cbr"])
                ys.append(r["acc"][r["snr_list"].index(snr)])
        if xs:
            plt.plot(xs, ys, marker="o", label=f"SNR={snr} dB")
    plt.xlabel("CBR (channel uses / source pixel)")
    plt.ylabel("acc")
    plt.title("Rate-accuracy: accuracy vs CBR")
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.legend(fontsize="small")
    plt.tight_layout()
    p2 = os.path.join(out_dir, "rate_accuracy_vs_cbr.png")
    plt.savefig(p2, dpi=200)
    plt.close()
    print(f"Wrote {p2}")


def main():
    args = parse_args()
    os.makedirs(os.path.abspath(args.out_dir), exist_ok=True)
    out_chans_list = [int(x) for x in args.out_chans_list.split(",") if x.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = []
    for oc in out_chans_list:
        r = _run_one_cbr(args, oc, device)
        if r is not None:
            results.append(r)

    if not results:
        print("No CBR produced results (missing checkpoints?). Nothing to aggregate.")
        return

    out_dir = os.path.abspath(args.out_dir)
    _write_csv(results, out_dir)
    _plot(results, out_dir)


if __name__ == "__main__":
    main()
