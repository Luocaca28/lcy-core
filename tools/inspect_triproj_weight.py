import argparse
import csv
import os
import sys
import torch


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _offdiag_abs_mean(matrix):
    diag = torch.diag(torch.diag(matrix))
    return (matrix - diag).abs().mean().item()


def inspect_model(model, label):
    found = False
    rows = []
    for name, module in model.named_modules():
        if "tri_merge" not in name or not hasattr(module, "proj"):
            continue
        found = True
        weight = module.proj.weight.detach().cpu()
        out_channels, in_channels, kernel = weight.shape
        print(f"\n[{label}] {name}")
        print("weight shape:", tuple(weight.shape))
        if kernel != 1:
            print("skip: expected kernel_size=1")
            continue

        if in_channels != 3 * out_channels:
            print("skip: unexpected projection shape")
            continue

        c = out_channels
        w0 = weight[:, 0 * c : 1 * c, 0]
        w1 = weight[:, 1 * c : 2 * c, 0]
        wd = weight[:, 2 * c : 3 * c, 0]
        for tag, block in [("W0", w0), ("W1", w1), ("Wd", wd)]:
            print(f"{tag} abs mean:", block.abs().mean().item())
            print(f"{tag} diag mean:", block.diag().mean().item())
            print(f"{tag} offdiag abs mean:", _offdiag_abs_mean(block))
        base_abs = 0.5 * (w0.abs().mean().item() + w1.abs().mean().item())
        wd_abs = wd.abs().mean().item()
        rows.append(
            {
                "checkpoint": label,
                "module": name,
                "W0_abs_mean": w0.abs().mean().item(),
                "W0_diag_mean": w0.diag().mean().item(),
                "W0_offdiag_abs_mean": _offdiag_abs_mean(w0),
                "W1_abs_mean": w1.abs().mean().item(),
                "W1_diag_mean": w1.diag().mean().item(),
                "W1_offdiag_abs_mean": _offdiag_abs_mean(w1),
                "Wd_abs_mean": wd_abs,
                "Wd_diag_mean": wd.diag().mean().item(),
                "Wd_offdiag_abs_mean": _offdiag_abs_mean(wd),
                "Wd_to_base_abs_ratio": wd_abs / (base_abs + 1e-12),
            }
        )

    if not found:
        print(f"[{label}] no tri_merge modules found")
    return rows


def save_rows(rows, output):
    if not output or not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    all_rows = []
    for path in args.checkpoints:
        model = torch.load(path, map_location=args.device, weights_only=False)
        all_rows.extend(inspect_model(model, path))
    save_rows(all_rows, args.output)


if __name__ == "__main__":
    main()
