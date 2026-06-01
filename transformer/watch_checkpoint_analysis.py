"""
Watch a training run and refresh matrix-analysis plots at epoch boundaries.

The trainer saves model.pth at the end of each epoch. This watcher observes
progress.txt; when the epoch number advances, it snapshots the previous epoch's
model.pth, appends that snapshot to analysis_checkpoints.json, then reruns:

  python3 analyze_checkpoints.py
  python3 plot_ffn_entropy.py

Usage:
  python3 watch_checkpoint_analysis.py runs/btm_d512_cont_local_20260530_185718

For a run that has already completed the current epoch but has not advanced to
the next progress epoch yet:
  python3 watch_checkpoint_analysis.py runs/... --snapshot-current-complete --once
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "analysis_checkpoints.json"
DEFAULT_SNAPSHOT_DIR = HERE / "btm_r2_backups" / "watch_checkpoints"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch progress.txt and refresh SVD/entropy plots.")
    parser.add_argument("run_dir", help="Run directory containing progress.txt and model.pth")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Analysis manifest JSON path")
    parser.add_argument("--snapshot-dir", default=str(DEFAULT_SNAPSHOT_DIR), help="Directory for copied checkpoint snapshots")
    parser.add_argument("--phase", default="d512s", help="Phase label written to the analysis manifest")
    parser.add_argument("--label-prefix", default="d512", help="Label prefix written to the analysis manifest")
    parser.add_argument("--poll-seconds", type=float, default=2.0, help="Polling interval")
    parser.add_argument("--settle-seconds", type=float, default=3.0, help="Require model.pth size/mtime to stay stable this long before copying")
    parser.add_argument("--once", action="store_true", help="Exit after one snapshot + analysis refresh")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing snapshot/manifest entry for the same epoch")
    parser.add_argument("--snapshot-current-complete", action="store_true", default=True,
        help="If progress.txt shows the current epoch at 100%% and train.log has the epoch loss, snapshot it immediately")
    parser.add_argument("--no-snapshot-current-complete", action="store_false", dest="snapshot_current_complete",
        help="Disable snapshotting the current epoch when progress.txt is already at 100%%")
    parser.add_argument("--snapshot-previous-complete", action="store_true", default=True,
        help="On startup, if progress.txt is already in a new epoch, snapshot the previous completed epoch when train.log has its loss")
    parser.add_argument("--no-snapshot-previous-complete", action="store_false", dest="snapshot_previous_complete",
        help="Disable startup snapshot of the previous completed epoch")
    parser.add_argument("--every-n-epochs", type=int, default=5,
        help="Only snapshot every N-th epoch (default 5)")
    return parser.parse_args()


def parse_progress(path: Path) -> dict[str, float | int] | None:
    try:
        text = path.read_text()
    except FileNotFoundError:
        return None

    epoch = re.search(r"Epoch\s*:\s*(\d+)\s*/\s*(\d+)", text)
    batch = re.search(r"Batch\s*:\s*(\d+)\s*/\s*(\d+)", text)
    loss = re.search(r"Loss\s*:\s*([0-9.]+)", text)
    if not epoch:
        return None

    info: dict[str, float | int] = {
        "epoch": int(epoch.group(1)),
        "epochs": int(epoch.group(2)),
    }
    if batch:
        info["batch"] = int(batch.group(1))
        info["batches"] = int(batch.group(2))
    if loss:
        info["loss"] = float(loss.group(1))
    return info


def parse_epoch_loss(log_path: Path, epoch: int) -> float | None:
    if not log_path.exists():
        return None
    pat = re.compile(rf"Epoch\s*{epoch}\s*:\s*Loss=([0-9.]+)")
    matches = pat.findall(log_path.read_text(errors="replace"))
    return float(matches[-1]) if matches else None


def load_run_args(run_dir: Path) -> dict:
    args_path = run_dir / "args.json"
    if not args_path.exists():
        return {}
    with args_path.open() as f:
        return json.load(f)


def wait_for_stable_file(path: Path, settle_seconds: float) -> None:
    last = None
    stable_since = None
    while True:
        stat = path.stat()
        cur = (stat.st_size, stat.st_mtime_ns)
        now = time.time()
        if cur == last:
            if stable_since is None:
                stable_since = now
            if now - stable_since >= settle_seconds:
                return
        else:
            last = cur
            stable_since = None
        time.sleep(0.5)


def load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return json.load(f)


def write_manifest(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(rows, f, indent=2)
        f.write("\n")
    tmp.replace(path)


def upsert_manifest(path: Path, row: dict, force: bool) -> bool:
    rows = load_manifest(path)
    for i, existing in enumerate(rows):
        if existing.get("label") == row["label"]:
            if not force:
                return False
            rows[i] = row
            write_manifest(path, rows)
            return True
    rows.append(row)
    rows.sort(key=lambda r: (int(r.get("epoch", 0)), r.get("label", "")))
    write_manifest(path, rows)
    return True


def manifest_has_epoch(path: Path, phase: str, epoch: int) -> bool:
    rows = load_manifest(path)
    return any(row.get("phase") == phase and int(row.get("epoch", -1)) == epoch for row in rows)


def run_analysis() -> None:
    for script in ("analyze_checkpoints.py", "plot_ffn_entropy.py"):
        print(f"spawning {script}", flush=True)
        subprocess.Popen([sys.executable, str(HERE / script)], cwd=HERE)


def snapshot_epoch(args: argparse.Namespace, run_dir: Path, epoch: int,
                   _skipped: set | None = None) -> bool:
    if epoch % args.every_n_epochs != 0:
        if _skipped is not None and epoch not in _skipped:
            print(f"skipping epoch {epoch} (every-n-epochs={args.every_n_epochs})", flush=True)
            _skipped.add(epoch)
        return False

    model_path = run_dir / "model.pth"
    if not model_path.exists():
        print(f"model not found: {model_path}", flush=True)
        return False

    run_args = load_run_args(run_dir)
    stride = run_args.get("data_stride", "na")
    label = f"{args.label_prefix} ep{epoch}s"
    snapshot_dir = Path(args.snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_name = f"{run_dir.name}_ep{epoch}_stride{stride}.pth"
    snapshot_path = snapshot_dir / snapshot_name

    if snapshot_path.exists() and not args.force:
        print(f"snapshot already exists for epoch {epoch}: {snapshot_path}", flush=True)
    else:
        wait_for_stable_file(model_path, args.settle_seconds)
        shutil.copy2(model_path, snapshot_path)
        print(f"snapshot saved: {snapshot_path}", flush=True)

    loss = parse_epoch_loss(run_dir / "train.log", epoch)
    rel_path = snapshot_path.relative_to(HERE).as_posix()
    row = {
        "label": label,
        "phase": args.phase,
        "epoch": epoch,
        "loss": loss,
        "rel_path": rel_path,
    }
    changed = upsert_manifest(Path(args.manifest), row, args.force)
    if not changed:
        print(f"manifest already has {label}; analysis refresh skipped", flush=True)
        return False

    #run_analysis()
    return True


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    progress_path = run_dir / "progress.txt"
    log_path = run_dir / "train.log"

    last_epoch = None
    skipped_epochs: set = set()
    print(f"watching {progress_path}", flush=True)

    while True:
        info = parse_progress(progress_path)
        if info is None:
            time.sleep(args.poll_seconds)
            continue

        epoch = int(info["epoch"])
        batch = int(info.get("batch", 0))
        batches = int(info.get("batches", 0))

        if last_epoch is None:
            last_epoch = epoch
            prev_epoch = epoch - 1
            if args.snapshot_previous_complete and prev_epoch > 0:
                if not manifest_has_epoch(Path(args.manifest), args.phase, prev_epoch):
                    if parse_epoch_loss(log_path, prev_epoch) is not None:
                        print(f"startup: snapshotting previous completed epoch {prev_epoch}", flush=True)
                        snapshot_epoch(args, run_dir, prev_epoch, skipped_epochs)
                        if args.once:
                            return
        if args.snapshot_current_complete and batches and batch >= batches:
            if not manifest_has_epoch(Path(args.manifest), args.phase, epoch):
                if parse_epoch_loss(log_path, epoch) is not None:
                    if epoch not in skipped_epochs:
                        print(f"epoch {epoch} complete; snapshotting", flush=True)
                    snapshot_epoch(args, run_dir, epoch, skipped_epochs)
                    if args.once:
                        return

        if epoch > last_epoch:
            completed_epoch = epoch - 1
            print(f"epoch advanced {last_epoch} -> {epoch}; snapshotting epoch {completed_epoch}", flush=True)
            snapshot_epoch(args, run_dir, completed_epoch, skipped_epochs)
            if args.once:
                return
            last_epoch = epoch
        elif epoch < last_epoch:
            print(f"epoch moved backwards {last_epoch} -> {epoch}; updating baseline", flush=True)
            last_epoch = epoch

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
