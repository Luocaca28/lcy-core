import argparse
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

    if not found:
        print(f"[{label}] no tri_merge modules found")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    for path in args.checkpoints:
        model = torch.load(path, map_location=args.device, weights_only=False)
        inspect_model(model, path)


if __name__ == "__main__":
    main()
