"""Vertical bar chart comparing per-tier classification accuracy.

For the attribution ladder (tasks/classification_pure/MANIFEST.md):
  tier1 pure compression (ADAPTIVE='no', no SNR embed, no channel)
  tier2 SNR=20 no-channel (ADAPTIVE='ssm', SNR embed on, no channel)
  tier3 comm version      (real channel)  -- add it once you have the number

These tiers each yield a SINGLE scalar accuracy (no SNR axis for the no-channel
tiers), so a bar chart -- not an snr-acc curve -- is the right comparison view.

Standalone: no repo imports, only matplotlib. Run it on whichever machine has the
numbers; it just draws.

    # default: the two pure-baseline numbers already measured
    python tools/plot_tier_accuracy.py --out tier_accuracy_bar.png

    # add tier3 later
    python tools/plot_tier_accuracy.py \
        --labels "tier1 pure,tier2 snr20-nochan,tier3 comm@20dB" \
        --acc "0.739,0.7266,0.71" --out tier_accuracy_bar.png

    # or read label,acc rows from a csv (header: label,acc)
    python tools/plot_tier_accuracy.py --csv pure_test_results.csv --out bar.png
"""

import argparse
import csv
import os


def parse_args():
    p = argparse.ArgumentParser(description="Bar chart of per-tier classification accuracy.")
    p.add_argument("--labels", default="tier1 pure,tier2 snr20-nochan",
                   help="Comma-separated bar labels (ignored if --csv given).")
    p.add_argument("--acc", default="0.739,0.7266",
                   help="Comma-separated accuracies in [0,1] (ignored if --csv given).")
    p.add_argument("--csv", default=None,
                   help="Optional CSV with header 'label,acc'; overrides --labels/--acc.")
    p.add_argument("--title", default="Attribution ladder: classification accuracy")
    p.add_argument("--out", default="tier_accuracy_bar.png", help="Output PNG path.")
    return p.parse_args()


def _load(args):
    if args.csv:
        labels, accs = [], []
        with open(args.csv, newline="") as f:
            for row in csv.DictReader(f):
                labels.append(row["label"])
                accs.append(float(row["acc"]))
        return labels, accs
    labels = [s.strip() for s in args.labels.split(",") if s.strip()]
    accs = [float(x) for x in args.acc.split(",") if x.strip()]
    if len(labels) != len(accs):
        raise ValueError(f"{len(labels)} labels but {len(accs)} accuracies -- must match.")
    return labels, accs


def main():
    args = parse_args()
    labels, accs = _load(args)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # A stable palette; extra tiers cycle through it.
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
    fig, ax = plt.subplots(figsize=(1.6 * len(labels) + 2.0, 4.5))
    bars = ax.bar(labels, accs, color=[colors[i % len(colors)] for i in range(len(labels))],
                  width=0.6, edgecolor="black", linewidth=0.6)

    # Value label on top of each bar.
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.01, f"{acc:.4f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Fixed [0, 1] axis so bar heights are honest and comparable across runs.
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title(args.title)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()

    out = os.path.abspath(args.out)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Wrote {out}")
    for label, acc in zip(labels, accs):
        print(f"  {label}: {acc:.4f}")


if __name__ == "__main__":
    main()
