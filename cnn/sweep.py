# --------------------------------------------------
# Tech Aarvam
# Copyright (c) 2026 Tech Aarvam.
# Author: Ram (Ramasubramanian B)
# --------------------------------------------------

# TBD: add lr shmoo
# TBD: add num_samples shmoo done below

import os
import subprocess
from datetime import datetime

import matplotlib.pyplot as plt

TRAINER  = "/home/rambala/work/learn/aiml/cnn/trainer.py"
OUT_ROOT = "/home/rambala/work/learn/aiml/cnn/sweeps"

BASELINE = {
    "epochs":      30,
    "batch_size":  100,
    "lr":          0.1,
    "seed":        73,
    "verbosity":   1,
    "hidden_size": 22,
    "num_samples": 1000,
}


def run_one(name, params, out_dir):
    log_path = os.path.join(out_dir, f"{name}.log")
    cmd = ["python", TRAINER]
    for key, value in params.items():
        cmd.extend([f"--{key.replace('_', '-')}", str(value)])

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    with open(log_path, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True, env=env)

    train_loss = train_acc = val_loss = val_acc = None
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if "Training Loss:" in line:
                parts = line.split(",")
                train_loss = float(parts[1].split(":")[1])
                train_acc  = float(parts[2].split(":")[1])
            elif "Validation Loss:" in line:
                parts = line.split(",")
                val_loss = float(parts[1].split(":")[1])
                val_acc  = float(parts[2].split(":")[1])

    return {"name": name, **params,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc}


def plot_shmoo(rows, x_key, xlabel, out_dir):
    rows = sorted(rows, key=lambda r: r[x_key])
    xs        = [r[x_key]    for r in rows]
    train_acc = [r["train_acc"] for r in rows]
    val_acc   = [r["val_acc"]   for r in rows]
    train_loss = [r["train_loss"] for r in rows]
    val_loss   = [r["val_loss"]   for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Shmoo: {xlabel}", fontsize=13, fontweight="bold")

    ax1.plot(xs, train_acc, marker="o", label="train")
    ax1.plot(xs, val_acc,   marker="o", label="validation")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("accuracy")
    ax1.set_title("Accuracy")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(xs, train_loss, marker="o", label="train")
    ax2.plot(xs, val_loss,   marker="o", label="validation")
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("loss")
    ax2.set_title("Loss")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"shmoo_{x_key}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Plot saved: {out_path}")


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(OUT_ROOT, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    hidden_sizes = list(range(2, 101, 10))
    rows = []
    for hs in hidden_sizes:
        params = {**BASELINE, "hidden_size": hs}
        name = f"hidden_{hs}"
        print(f"Running {name} ...")
        row = run_one(name, params, out_dir)
        rows.append(row)
        print(f"  train_acc={row['train_acc']:.4f}  val_acc={row['val_acc']:.4f}")

    plot_shmoo(rows, "hidden_size", "Hidden Size", out_dir)

    dataset_sizes = list(range(50, 1001, 100))
    rows = []
    for ns in dataset_sizes:
        params = {**BASELINE, "num_samples": ns}
        name = f"num_samples_{ns}"
        print(f"Running {name} ...")
        row = run_one(name, params, out_dir)
        rows.append(row)
        print(f"  train_acc={row['train_acc']:.4f}  val_acc={row['val_acc']:.4f}")

    plot_shmoo(rows, "num_samples", "Dataset Size (per class)", out_dir)


if __name__ == "__main__":
    main()
