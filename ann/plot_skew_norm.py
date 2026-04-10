# --------------------------------------------------
# Tech Aarvam
# Copyright (c) 2026 Tech Aarvam.
# Author: Ram (Ramasubramanian B)
# --------------------------------------------------

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_kv_line(line):
    parts = line.strip().split()
    label = parts[0]
    data = {}
    for part in parts[1:]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        data[key] = value
    return label, data


def maybe_number(value):
    if isinstance(value, (int, float, bool)):
        return value
    if value == "True":
        return True
    if value == "False":
        return False
    try:
        if "." in value or "e" in value or "E" in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_run(log_path):
    config = {}
    final = {}
    epoch_rows = []

    with log_path.open() as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("CONFIG "):
                _, data = parse_kv_line(line)
                config = {k: maybe_number(v) for k, v in data.items()}
            elif line.startswith("EPOCH "):
                _, data = parse_kv_line(line)
                epoch_rows.append({k: maybe_number(v) for k, v in data.items()})
            elif line.startswith("FINAL "):
                _, data = parse_kv_line(line)
                final = {k: maybe_number(v) for k, v in data.items()}

    last_epoch = epoch_rows[-1] if epoch_rows else {}
    return {"config": config, "final": final, "epochs": epoch_rows, "last_epoch": last_epoch}


def save_summary_csv(out_path, rows):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_dual(xs, ys_left, ys_right, xlabel, left_label, right_label, title, out_path):
    fig, ax1 = plt.subplots(figsize=(7.2, 4.6))
    ax1.plot(xs, ys_left, marker="o", color="tab:blue")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(left_label, color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(xs, ys_right, marker="s", color="tab:red")
    ax2.set_ylabel(right_label, color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    plt.title(title)
    fig.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_lines(xs, series, xlabel, ylabel, title, out_path):
    plt.figure(figsize=(7.2, 4.6))
    for label, ys in series:
        plt.plot(xs, ys, marker="o", label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_convergence(run_map, run_names, report_every, metric_key, ylabel, title, out_path):
    plt.figure(figsize=(7.4, 4.8))
    for run_name in run_names:
        rows = run_map[run_name]["epochs"]
        xs = [i * report_every for i in range(len(rows))]
        ys = [float(row[metric_key]) for row in rows]
        plt.plot(xs, ys, label=run_name)
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_bars(labels, values, ylabel, title, out_path, colors=None):
    plt.figure(figsize=(6.8, 4.6))
    plt.bar(labels, values, color=colors)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot weight skew and normalization sweep results.")
    parser.add_argument("--sweep-dir", required=True, help="Directory containing *.log files for the skew/normalization sweep.")
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    run_names = [
        "weight_skew_0.0",
        "weight_skew_0.25",
        "weight_skew_0.5",
        "weight_skew_0.75",
        "weight_skew_1.0",
        "normalize_0",
        "normalize_1",
    ]

    run_map = {name: load_run(sweep_dir / f"{name}.log") for name in run_names}

    summary_rows = []
    for name in run_names:
        row = {"run_name": name}
        row.update(run_map[name]["config"])
        row.update(run_map[name]["last_epoch"])
        row.update(run_map[name]["final"])
        summary_rows.append(row)

    save_summary_csv(sweep_dir / "summary.csv", summary_rows)

    skew_runs = [name for name in run_names if name.startswith("weight_skew_")]
    skew_rows = [
        {"run_name": name, **run_map[name]["config"], **run_map[name]["last_epoch"], **run_map[name]["final"]}
        for name in skew_runs
    ]
    skew_rows.sort(key=lambda row: float(row["weight_skew"]))
    skew_x = [float(row["weight_skew"]) for row in skew_rows]
    report_every = int(run_map["weight_skew_0.5"]["config"]["report_every"])

    plot_dual(
        skew_x,
        [float(row["validationAccuracy"]) for row in skew_rows],
        [float(row["validationLoss"]) for row in skew_rows],
        "weight skew",
        "validation accuracy",
        "validation loss",
        "Weight Skew: Validation Accuracy vs Loss",
        sweep_dir / "weight_skew_validation_accuracy_loss.png",
    )

    plot_dual(
        skew_x,
        [float(row["sigmoidDerivativeAvg"]) for row in skew_rows],
        [float(row["sigmoidSaturationFrac"]) for row in skew_rows],
        "weight skew",
        "sigmoid derivative avg",
        "sigmoid saturation frac",
        "Weight Skew: Sigmoid Derivative vs Saturation",
        sweep_dir / "weight_skew_sigmoid_dynamics.png",
    )

    plot_lines(
        skew_x,
        [
            ("layer1", [float(row["MeanWAgradientNorm_1"]) for row in skew_rows]),
            ("layer2", [float(row["MeanWAgradientNorm_2"]) for row in skew_rows]),
        ],
        "weight skew",
        "mean gradient norm",
        "Weight Skew: Mean Gradient Norm",
        sweep_dir / "weight_skew_gradient_norm.png",
    )

    plot_lines(
        skew_x,
        [
            ("layer1", [float(row["MeanWAupdateRatio_1"]) for row in skew_rows]),
            ("layer2", [float(row["MeanWAupdateRatio_2"]) for row in skew_rows]),
        ],
        "weight skew",
        "mean update ratio",
        "Weight Skew: Mean Update Ratio",
        sweep_dir / "weight_skew_mean_update_ratio.png",
    )

    plot_lines(
        skew_x,
        [
            ("layer1 max", [float(row["MaxWAupdateRatio_1"]) for row in skew_rows]),
            ("layer2 max", [float(row["MaxWAupdateRatio_2"]) for row in skew_rows]),
        ],
        "weight skew",
        "max update ratio",
        "Weight Skew: Max Update Ratio",
        sweep_dir / "weight_skew_max_update_ratio.png",
    )

    plot_lines(
        skew_x,
        [
            ("layer1 z>3 frac", [float(row["fracLargeZ_1"]) for row in skew_rows]),
            ("layer2 z>3 frac", [float(row["fracLargeZ_2"]) for row in skew_rows]),
        ],
        "weight skew",
        "fraction",
        "Weight Skew: Large Z Fraction",
        sweep_dir / "weight_skew_large_z_fraction.png",
    )

    plot_convergence(
        run_map,
        skew_runs,
        report_every,
        "trainingLoss",
        "training loss",
        "Weight Skew Convergence: Loss",
        sweep_dir / "weight_skew_convergence_loss.png",
    )

    plot_convergence(
        run_map,
        skew_runs,
        report_every,
        "sigmoidDerivativeAvg",
        "sigmoid derivative avg",
        "Weight Skew Convergence: Sigmoid Derivative",
        sweep_dir / "weight_skew_convergence_sigmoid_derivative.png",
    )

    plot_convergence(
        run_map,
        skew_runs,
        report_every,
        "sigmoidSaturationFrac",
        "sigmoid saturation frac",
        "Weight Skew Convergence: Sigmoid Saturation",
        sweep_dir / "weight_skew_convergence_sigmoid_saturation.png",
    )

    plot_convergence(
        run_map,
        skew_runs,
        report_every,
        "MeanWAupdateRatio_1",
        "mean update ratio layer1",
        "Weight Skew Convergence: Layer 1 Update Ratio",
        sweep_dir / "weight_skew_convergence_update_ratio_l1.png",
    )

    plot_convergence(
        run_map,
        skew_runs,
        report_every,
        "MeanWAupdateRatio_2",
        "mean update ratio layer2",
        "Weight Skew Convergence: Layer 2 Update Ratio",
        sweep_dir / "weight_skew_convergence_update_ratio_l2.png",
    )

    norm_rows = [
        {"run_name": name, **run_map[name]["config"], **run_map[name]["last_epoch"], **run_map[name]["final"]}
        for name in ["normalize_0", "normalize_1"]
    ]
    norm_rows.sort(key=lambda row: int(bool(row["normalize"])))
    norm_labels = ["off", "on"]

    plot_bars(
        norm_labels,
        [float(row["validationAccuracy"]) for row in norm_rows],
        "validation accuracy",
        "Normalization: Validation Accuracy",
        sweep_dir / "normalize_validation_accuracy.png",
        colors=["tab:gray", "tab:green"],
    )

    plot_bars(
        norm_labels,
        [float(row["sigmoidDerivativeAvg"]) for row in norm_rows],
        "sigmoid derivative avg",
        "Normalization: Sigmoid Derivative",
        sweep_dir / "normalize_sigmoid_derivative.png",
        colors=["tab:gray", "tab:green"],
    )

    plot_bars(
        norm_labels,
        [float(row["sigmoidSaturationFrac"]) for row in norm_rows],
        "sigmoid saturation frac",
        "Normalization: Sigmoid Saturation",
        sweep_dir / "normalize_sigmoid_saturation.png",
        colors=["tab:gray", "tab:green"],
    )

    plot_lines(
        [0, 1],
        [
            ("layer1 mean grad", [float(row["MeanWAgradientNorm_1"]) for row in norm_rows]),
            ("layer2 mean grad", [float(row["MeanWAgradientNorm_2"]) for row in norm_rows]),
            ("layer1 mean upd", [float(row["MeanWAupdateRatio_1"]) for row in norm_rows]),
            ("layer2 mean upd", [float(row["MeanWAupdateRatio_2"]) for row in norm_rows]),
        ],
        "normalization off/on",
        "value",
        "Normalization: Gradient and Update Ratios",
        sweep_dir / "normalize_gradient_update.png",
    )

    plot_convergence(
        run_map,
        ["normalize_0", "normalize_1"],
        report_every,
        "trainingLoss",
        "training loss",
        "Normalization Convergence: Loss",
        sweep_dir / "normalize_convergence_loss.png",
    )

    plot_convergence(
        run_map,
        ["normalize_0", "normalize_1"],
        report_every,
        "sigmoidDerivativeAvg",
        "sigmoid derivative avg",
        "Normalization Convergence: Sigmoid Derivative",
        sweep_dir / "normalize_convergence_sigmoid_derivative.png",
    )

    plot_convergence(
        run_map,
        ["normalize_0", "normalize_1"],
        report_every,
        "sigmoidSaturationFrac",
        "sigmoid saturation frac",
        "Normalization Convergence: Sigmoid Saturation",
        sweep_dir / "normalize_convergence_sigmoid_saturation.png",
    )

    plot_convergence(
        run_map,
        ["normalize_0", "normalize_1"],
        report_every,
        "MeanWAupdateRatio_1",
        "mean update ratio layer1",
        "Normalization Convergence: Layer 1 Update Ratio",
        sweep_dir / "normalize_convergence_update_ratio_l1.png",
    )

    plot_convergence(
        run_map,
        ["normalize_0", "normalize_1"],
        report_every,
        "MeanWAupdateRatio_2",
        "mean update ratio layer2",
        "Normalization Convergence: Layer 2 Update Ratio",
        sweep_dir / "normalize_convergence_update_ratio_l2.png",
    )

    print(f"PLOTS_DIR {sweep_dir}")


if __name__ == "__main__":
    main()
