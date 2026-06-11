import argparse
import csv
import glob
import os
import re


def _read_curve_csv(path):
    xs = []
    ys = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            xs.append(float(row[0]))
            ys.append(float(row[1]))
    return xs, ys


def _write_curve_csv(path, x_name, y_name, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([x_name, y_name])
        writer.writerows(rows)


def _plot_curve(xs, ys, xlabel, ylabel, title, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_multi_curve(curves, xlabel, ylabel, title, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure()
    for label, xs, ys in curves:
        plt.plot(xs, ys, marker="o", label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_train_val_loss(log_dir):
    combined_csv = os.path.join(log_dir, "train_val_loss_curve.csv")
    if os.path.exists(combined_csv):
        epochs = []
        train_losses = []
        val_epochs = []
        val_losses = []
        with open(combined_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                epoch = float(row["epoch"])
                epochs.append(epoch)
                train_losses.append(float(row["train_loss"]))
                val_loss = row.get("val_loss", "")
                if val_loss != "":
                    val_epochs.append(epoch)
                    val_losses.append(float(val_loss))

        out_png = os.path.join(log_dir, "train_val_loss_curve_final.png")
        curves = [("train", epochs, train_losses)]
        if val_losses:
            curves.append(("val", val_epochs, val_losses))
        _plot_multi_curve(curves, "Epoch", "Loss", "Train/Val Loss", out_png)
        print(f"Saved train/val loss curve: {out_png}")
        return epochs, train_losses

    train_csv = os.path.join(log_dir, "loss_curve.csv")
    if not os.path.exists(train_csv):
        print(f"Missing train loss CSV: {train_csv}")
        return None

    epochs, losses = _read_curve_csv(train_csv)
    out_png = os.path.join(log_dir, "train_loss_curve_final.png")
    _plot_curve(epochs, losses, "Epoch", "Loss", "Training Loss", out_png)
    print(f"Saved train loss curve: {out_png}")
    return epochs, losses


def _plot_eval_loss_if_available(log_dir, train_curve):
    eval_csv = os.path.join(log_dir, "eval_loss_curve.csv")
    if not os.path.exists(eval_csv):
        print(
            "No eval_loss_curve.csv found. Existing val_epoch_* files are PSNR/MS-SSIM "
            "metrics, not true eval loss."
        )
        return None

    epochs, losses = _read_curve_csv(eval_csv)
    out_png = os.path.join(log_dir, "eval_loss_curve_final.png")
    _plot_curve(epochs, losses, "Epoch", "Loss", "Eval Loss", out_png)
    print(f"Saved eval loss curve: {out_png}")

    if train_curve is not None:
        train_epochs, train_losses = train_curve
        combined_png = os.path.join(log_dir, "train_eval_loss_curve.png")
        _plot_multi_curve(
            [
                ("train", train_epochs, train_losses),
                ("eval", epochs, losses),
            ],
            "Epoch",
            "Loss",
            "Train/Eval Loss",
            combined_png,
        )
        print(f"Saved train/eval loss curve: {combined_png}")
    return epochs, losses


def _aggregate_val_metric(log_dir, metric):
    pattern = os.path.join(log_dir, f"val_epoch_*_snr_{metric}_curve.csv")
    regex = re.compile(r"val_epoch_(\d+)_snr_" + re.escape(metric) + r"_curve\.csv$")
    rows = []

    for path in glob.glob(pattern):
        match = regex.search(os.path.basename(path))
        if not match:
            continue
        epoch = int(match.group(1))
        _, values = _read_curve_csv(path)
        if not values:
            continue
        rows.append((epoch, sum(values) / len(values)))

    rows.sort(key=lambda item: item[0])
    if not rows:
        print(f"No validation {metric.upper()} CSV files found in {log_dir}")
        return

    y_name = f"avg_{metric}"
    out_csv = os.path.join(log_dir, f"eval_avg_{metric}_by_epoch.csv")
    out_png = os.path.join(log_dir, f"eval_avg_{metric}_by_epoch.png")
    _write_curve_csv(out_csv, "epoch", y_name, rows)
    _plot_curve(
        [row[0] for row in rows],
        [row[1] for row in rows],
        "Epoch",
        f"Average {metric.upper()}",
        f"Eval Average {metric.upper()}",
        out_png,
    )
    print(f"Saved eval average {metric.upper()} curve: {out_png}")


def _plot_test_snr_curve(log_dir, metric, ylabel):
    src_csv = os.path.join(log_dir, f"snr_{metric}_curve.csv")
    if not os.path.exists(src_csv):
        print(f"Missing final test curve CSV: {src_csv}")
        return

    snrs, values = _read_curve_csv(src_csv)
    out_png = os.path.join(log_dir, f"test_snr_{metric}_curve.png")
    _plot_curve(snrs, values, "SNR (dB)", ylabel, f"Test SNR-{ylabel}", out_png)
    print(f"Saved final test SNR-{ylabel} curve: {out_png}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default="")
    parser.add_argument("--experiment_dir", default="")
    args = parser.parse_args()

    if args.log_dir:
        log_dir = args.log_dir
    elif args.experiment_dir:
        log_dir = os.path.join(args.experiment_dir, "logs")
    else:
        raise ValueError("Use --log_dir or --experiment_dir")

    log_dir = os.path.abspath(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    train_curve = _plot_train_val_loss(log_dir)
    _plot_eval_loss_if_available(log_dir, train_curve)
    _aggregate_val_metric(log_dir, "psnr")
    _aggregate_val_metric(log_dir, "ms-ssim")
    _plot_test_snr_curve(log_dir, "psnr", "PSNR")
    _plot_test_snr_curve(log_dir, "ms-ssim", "MS-SSIM")


if __name__ == "__main__":
    main()
