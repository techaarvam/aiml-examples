# Training Runner — Tech Aarvam Transformer

## Overview

`runner.py` is the single entry point for all training runs. It reads a TOML profile file, creates a timestamped output directory, writes reproducibility artifacts, and launches training with combined stdout/stderr logging.

---

## Quick Start

```bash
# List available profiles
python runner.py runs.toml

# Run a profile
python runner.py runs.toml smoke     # 5-epoch sanity check (~5 min)
python runner.py runs.toml mid       # 10-epoch local trial
python runner.py runs.toml server    # full cloud run

# Override any arg inline
python runner.py runs.toml server --epochs 20 --batch_size 512
```

---

## Run Profiles (runs.toml)

The TOML file defines run profiles. A `[defaults]` section applies to all profiles unless overridden.

### Profile: `smoke`
- **Purpose**: verify the full code path runs end-to-end
- **Dataset**: `raw_data/combined.txt` (~12MB, Gutenberg novels)
- **Model**: 2 layers, 2 heads, vecDims=32, window=16, batch=16384, vocab=3K
- **Expected time**: a few minutes on local GPU
- **Use when**: any code change, before committing, on a new machine

### Profile: `mid`
- **Purpose**: local architecture iteration with fast feedback
- **Dataset**: `raw_data/combined.txt`
- **Model**: 6 layers, 8 heads, vecDims=128, window=64, batch=512, vocab=30K
- **Expected time**: ~1–2 hours on local GPU
- **Use when**: tuning hyperparameters, testing new features

### Profile: `server`
- **Purpose**: full quality training run on cloud GPU (A100 40GB)
- **Dataset**: `raw_data/wikitext103.txt` (~514MB, Wikipedia)
- **Model**: 8 layers, 8 heads, vecDims=256, window=128, batch=1024, vocab=50K
- **Expected time**: ~27 hours/epoch on A100 40GB — plan for 5–8 epochs (~₹1,500–2,000)
- **Use when**: renting cloud GPU (GCP `a2-highgpu-1g` spot / Vast.ai 4090)
- **Note**: 100 epochs on full WikiText-103 is not feasible in one budget run — use 5–10 epochs

---

## Adding a Custom Profile

Add a new section to `runs.toml`:

```toml
[myrun]
input          = "raw_data/wikitext50m.txt"
vecDims        = 192
num_heads      = 8
num_layers     = 6
window_size    = 64
batch_size     = 128
epochs         = 30
lr             = 0.0003
max_vocab_size = 30000
```

Any key not specified falls back to `[defaults]`.

---

## TOML Key Reference

| Key | Description | Typical values |
|-----|-------------|----------------|
| `input` | Path to training text file | `raw_data/*.txt` |
| `vecDims` | Embedding / model dimension | 32, 64, 128, 256 |
| `num_heads` | Number of attention heads | 2, 4, 8 |
| `num_layers` | Number of transformer layers | 2, 6, 8, 12 |
| `window_size` | Context window (tokens) | 16, 64, 128 |
| `batch_size` | Training batch size | 64, 128, 192, 256 |
| `epochs` | Number of training epochs | 5, 10, 100 |
| `lr` | Learning rate | 0.001, 0.0003 |
| `lr_schedule` | LR scheduler | `none`, `plateau`, `cosine` |
| `max_vocab_size` | Vocabulary cap (0 = unlimited) | 3000, 30000, 50000 |
| `embedding_type` | Embedding mode | `learned`, `glove-fixed` |
| `output_type` | Loss / output mode | `indices` (CrossEntropy) |
| `seed` | Random seed | 42 |
| `float_type` | Model precision | `float32`, `bfloat16` |
| `optimizer` | Optimiser | `adam`, `sgd` |
| `quantize` | INT8 ONNX quantization | `true`, `false` |

---

## web_infer.html Templating

The source `web_infer.html` uses `63` (= `window_size - 1` for the default window=64) throughout its JavaScript. When runner.py copies it to a run directory, it substitutes all window-size-dependent values with the profile's actual `window_size - 1`. This means:

- A `server` profile run (window=128) gets `seqLen = 127` in its copy
- A `smoke` profile run (window=16) gets `seqLen = 15` in its copy
- The source file stays runnable as-is for window=64 runs

---

## Output Directory Structure

Each run creates `runs/<profile>_<YYYYMMDD_HHMMSS>/`:

```
runs/
└── server_20260512_063045/
    ├── args.json           # all hyperparameters for this run
    ├── train.log           # combined stdout+stderr from training
    ├── model.pth           # latest checkpoint (saved every epoch)
    ├── model.onnx          # ONNX export (created by cmd_pth2onnx.sh)
    ├── vocab.json          # vocabulary mapping (word → index)
    ├── web_infer.html      # browser inference UI for this run
    ├── cmd_train.sh        # exact command used to start this run
    ├── cmd_resume.sh       # command to resume from latest checkpoint
    ├── cmd_infer.sh        # interactive text inference against model.pth
    ├── cmd_pth2onnx.sh     # converts model.pth → model.onnx
    ├── cmd_netron.sh       # opens Netron graph visualiser
    └── cmd_webserver.sh    # serves browser inference on port 9090
```

---

## Generated Scripts

### `cmd_infer.sh`
Runs the interactive inference prompt against the latest `model.pth`. Loads the model in inference-only mode (no training). Use this after (or during) training to test the model.
```bash
./runs/server_20260512_063045/cmd_infer.sh
```

### `cmd_train.sh`
Exact replica of the command that launched this run. Useful for restarting from scratch with identical settings.
```bash
./runs/server_20260512_063045/cmd_train.sh
```

### `cmd_resume.sh`
Resumes training from the latest `model.pth` checkpoint in this run directory. Appends to `train.log`.
```bash
./runs/server_20260512_063045/cmd_resume.sh
```

### `cmd_pth2onnx.sh`
Converts `model.pth` to `model.onnx` in the same run directory. Run this after training (or mid-run on a saved checkpoint).
```bash
./runs/server_20260512_063045/cmd_pth2onnx.sh
```

### `cmd_netron.sh`
Starts Netron on port 8081 to visualise the ONNX graph. Requires `model.onnx` to exist first.
```bash
./runs/server_20260512_063045/cmd_netron.sh
# Open http://<machine-ip>:8081 in browser
```

### `cmd_webserver.sh`
Serves the run directory on port 9090. The `web_infer.html` page loads `vocab.json` and `model.onnx` from the same directory and runs inference in the browser via ONNX Runtime Web (WebAssembly).
```bash
./runs/server_20260512_063045/cmd_webserver.sh
# Open http://<machine-ip>:9090/web_infer.html in browser
```

---

## Monitoring Training

Training stdout and stderr are written simultaneously to the terminal and to `train.log` via `tee`. Python runs with `-u` (unbuffered) so output appears immediately.

```bash
# Follow live log from another terminal
tail -f runs/server_20260512_063045/train.log

# Check last epoch result
grep "Epoch" runs/server_20260512_063045/train.log | tail -5
```

When piped to a log file (non-interactive), tqdm is disabled and a heartbeat line is printed every ~10% of batches per epoch:

```
[5040/50407] loss=4.2813
[10080/50407] loss=4.1204
...
Epoch7: Loss=3.9514
```

---

## Resuming a Run

```bash
# Resume using the generated script (recommended)
./runs/server_20260512_063045/cmd_resume.sh

# Or manually with start_epoch override (for old-format checkpoints)
python trainer.py --resume --model_file runs/server_20260512_063045/model.pth \
    --start_epoch 7 --vocab_file runs/server_20260512_063045/vocab.json \
    [... all other original args ...]
```

New-format checkpoints (saved by runner) store epoch and optimizer state automatically, so `--start_epoch` is not needed.

---

## Reproducibility

`args.json` captures everything needed to reproduce a run:

```json
{
  "vecDims": 256,
  "num_heads": 8,
  "num_layers": 8,
  "window_size": 128,
  "batch_size": 256,
  "epochs": 100,
  "lr": 0.0003,
  "seed": 42,
  "profile": "server",
  "launched_at": "20260512_063045",
  ...
}
```

---

## Cloud Setup (GCP / Vast.ai)

```bash
# On cloud machine
git clone <repo>
cd transformer
pip install -r requirements.txt
python raw_data/download_wiki.py       # downloads wikitext103.txt + wikitext50m.txt

python runner.py runs.toml smoke       # verify environment works
python runner.py runs.toml server      # start full run
```
