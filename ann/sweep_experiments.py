# --------------------------------------------------
# Tech Aarvam
# Copyright (c) 2026 Tech Aarvam.
# Author: Ram (Ramasubramanian B)
# --------------------------------------------------

# This file and the argParser.py are the files that used AI code generation.
# All other code is hand-written with a learning objective in mind.

import csv
import os
import subprocess
from copy import deepcopy
from datetime import datetime

import matplotlib.pyplot as plt


MAIN = "/home/rambala/work/learn/aiml/ann/MainLayer.py"
OUT_ROOT = "/home/rambala/work/learn/aiml/ann/sweeps"


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


def run_one(name, params, out_dir):
    log_path = os.path.join(out_dir, f"{name}.log")
    cmd = ["python", MAIN]

    for key, value in params.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])

    with open(log_path, "w") as log_file:
        proc = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, check=True)

    config = {}
    final = {}
    last_epoch = {}
    epoch_rows = []
    with open(log_path) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("CONFIG "):
                _, data = parse_kv_line(line)
                config = {k: maybe_number(v) for k, v in data.items()}
            elif line.startswith("EPOCH "):
                _, data = parse_kv_line(line)
                last_epoch = {k: maybe_number(v) for k, v in data.items()}
                epoch_rows.append(last_epoch)
            elif line.startswith("FINAL "):
                _, data = parse_kv_line(line)
                final = {k: maybe_number(v) for k, v in data.items()}

    row = {"run_name": name, **config, **last_epoch, **final, "log_path": log_path}
    return row, epoch_rows


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_line(xs, ys_train, ys_val, xlabel, title, out_path):
    plt.figure(figsize=(7, 4.5))
    plt.plot(xs, ys_train, marker="o", label="train")
    plt.plot(xs, ys_val, marker="o", label="validation")
    plt.xlabel(xlabel)
    plt.ylabel("accuracy")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_dual_metric(xs, ys_left, ys_right, xlabel, left_label, right_label, title, out_path):
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
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


def plot_convergence(epoch_rows, run_names, report_every, metric_key, ylabel, title, out_path):
    plt.figure(figsize=(7, 4.5))
    for run_name in run_names:
        rows = [r for r in epoch_rows if r["run_name"] == run_name]
        xs = [int(r["epoch_index"]) * report_every for r in rows]
        ys = [float(r[metric_key]) for r in rows]
        plt.plot(xs, ys, label=run_name)
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(OUT_ROOT, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    baseline = {
        "epochs": 1000,
        "verbosity": 1,
        "report_every": 10,
        "hidden_size": 300,
        "lr1": 0.9,
        "lr2": 0.003,
        "weight_scale": 0.1,
        "normalize": False,
        "seed": 33,
    }

    experiments = []

    for value in [0.03, 0.1, 0.3, 0.9, 2.0]:
        params = deepcopy(baseline)
        params["lr1"] = value
        experiments.append((f"lr1_{value}", params))

    for value in [0.0003, 0.001, 0.003, 0.01, 0.03]:
        params = deepcopy(baseline)
        params["lr2"] = value
        experiments.append((f"lr2_{value}", params))

    for value in [50, 100, 300, 500]:
        params = deepcopy(baseline)
        params["hidden_size"] = value
        experiments.append((f"hidden_{value}", params))

    for value in [0.02, 0.05, 0.1, 0.2]:
        params = deepcopy(baseline)
        params["weight_scale"] = value
        experiments.append((f"weight_scale_{value}", params))

    for value in [False, True]:
        params = deepcopy(baseline)
        params["normalize"] = value
        experiments.append((f"normalize_{int(value)}", params))

    summary_rows = []
    all_epoch_rows = []

    for name, params in experiments:
        print(f"RUN {name}")
        row, epoch_rows = run_one(name, params, out_dir)
        summary_rows.append(row)
        for idx, epoch_row in enumerate(epoch_rows):
            all_epoch_rows.append({"run_name": name, "epoch_index": idx, **epoch_row})

    summary_csv = os.path.join(out_dir, "summary.csv")
    epochs_csv = os.path.join(out_dir, "epochs.csv")
    write_csv(summary_csv, summary_rows)
    write_csv(epochs_csv, all_epoch_rows)

    lr_rows = sorted(
        [r for r in summary_rows if str(r["run_name"]).startswith("lr1_")],
        key=lambda r: float(r["lr1"]),
    )
    plot_line(
        [float(r["lr1"]) for r in lr_rows],
        [float(r["trainingAccuracy"]) for r in lr_rows],
        [float(r["validationAccuracy"]) for r in lr_rows],
        "lr1",
        "Learning Rate Sensitivity",
        os.path.join(out_dir, "lr1_accuracy.png"),
    )
    plot_convergence(
        all_epoch_rows,
        [str(r["run_name"]) for r in lr_rows],
        int(baseline["report_every"]),
        "trainingLoss",
        "training loss",
        "Learning Rate Convergence: Loss",
        os.path.join(out_dir, "lr1_convergence_loss.png"),
    )
    plot_convergence(
        all_epoch_rows,
        [str(r["run_name"]) for r in lr_rows],
        int(baseline["report_every"]),
        "trainingAccuracy",
        "training accuracy",
        "Learning Rate Convergence: Accuracy",
        os.path.join(out_dir, "lr1_convergence_accuracy.png"),
    )

    lr2_rows = sorted(
        [r for r in summary_rows if str(r["run_name"]).startswith("lr2_")],
        key=lambda r: float(r["lr2"]),
    )
    plot_line(
        [float(r["lr2"]) for r in lr2_rows],
        [float(r["trainingAccuracy"]) for r in lr2_rows],
        [float(r["validationAccuracy"]) for r in lr2_rows],
        "lr2",
        "Output Layer Learning Rate Sensitivity",
        os.path.join(out_dir, "lr2_accuracy.png"),
    )
    plot_convergence(
        all_epoch_rows,
        [str(r["run_name"]) for r in lr2_rows],
        int(baseline["report_every"]),
        "trainingLoss",
        "training loss",
        "Output Layer Learning Rate Convergence: Loss",
        os.path.join(out_dir, "lr2_convergence_loss.png"),
    )
    plot_convergence(
        all_epoch_rows,
        [str(r["run_name"]) for r in lr2_rows],
        int(baseline["report_every"]),
        "trainingAccuracy",
        "training accuracy",
        "Output Layer Learning Rate Convergence: Accuracy",
        os.path.join(out_dir, "lr2_convergence_accuracy.png"),
    )

    hidden_rows = sorted(
        [r for r in summary_rows if str(r["run_name"]).startswith("hidden_")],
        key=lambda r: int(r["hidden_size"]),
    )
    plot_line(
        [int(r["hidden_size"]) for r in hidden_rows],
        [float(r["trainingAccuracy"]) for r in hidden_rows],
        [float(r["validationAccuracy"]) for r in hidden_rows],
        "hidden_size",
        "Hidden Size Sensitivity",
        os.path.join(out_dir, "hidden_size_accuracy.png"),
    )

    weight_rows = sorted(
        [r for r in summary_rows if str(r["run_name"]).startswith("weight_scale_")],
        key=lambda r: float(r["weight_scale"]),
    )
    plot_dual_metric(
        [float(r["weight_scale"]) for r in weight_rows],
        [float(r["sigmoidSaturationFrac"]) for r in weight_rows],
        [float(r["validationAccuracy"]) for r in weight_rows],
        "weight_scale",
        "sigmoid saturation frac",
        "validation accuracy",
        "Weight Scale Sensitivity",
        os.path.join(out_dir, "weight_scale_saturation_accuracy.png"),
    )

    norm_rows = sorted(
        [r for r in summary_rows if str(r["run_name"]).startswith("normalize_")],
        key=lambda r: int(bool(r["normalize"])),
    )
    xs = ["off" if not r["normalize"] else "on" for r in norm_rows]
    plt.figure(figsize=(6.5, 4.5))
    plt.bar(xs, [float(r["validationAccuracy"]) for r in norm_rows], color=["tab:gray", "tab:green"])
    plt.ylabel("validation accuracy")
    plt.title("Normalization Sensitivity")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "normalize_validation_accuracy.png"), dpi=150)
    plt.close()

    print(f"SWEEP_DIR {out_dir}")
    print(f"SUMMARY_CSV {summary_csv}")
    print(f"EPOCHS_CSV {epochs_csv}")


if __name__ == "__main__":
    main()
