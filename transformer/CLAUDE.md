# CLAUDE.md — Tech Aarvam Transformer Project

## Running Notes Guidelines

`running_notes.md` is a data log, not an analysis document.

- **Data only**: loss values, step counts, config params, token counts, measured outcomes
- **No claims without evidence**: if something is unverified or a hypothesis, keep it in conversation — do not write it to the file
- **No conjectures**: do not explain *why* something happened unless there is direct evidence
- **Humble**: do not overstate results — "loss descended" not "model learned X"; "loss 5.73" not "strong recovery"
- **To the point**: one row per data point, one line per fact — no padding, no narrative

---

## Operational Rules

**scp, rsync, git** — run these directly via Bash tool without asking.  
**Training jobs** — never run. cmd_run*.sh, runner.py, trainer.py invocations go in a code block for the user to execute. This includes anything that starts a training loop.

Always use `scp -C` (compressed) and `rsync -z` for all server transfers.

---

## Server

| Field | Value |
|---|---|
| Host | 207.102.87.207 |
| Port | 52853 |
| User | root |
| Python | `/venv/main/bin/activate` |
| Code root | `/root/transformer/` |
| Run output dirs | `/root/transformer/runs/` |
| Machine3 data | `/root/transformer/raw_data/btm_machines/machine3/` |

Typical scp pattern:
```bash
scp -C -P 52853 local_file root@207.102.87.207:/root/transformer/path
scp -C -P 52853 root@207.102.87.207:/root/transformer/path local_file
```

---

## Training Structure

**Runner invocation** (always use runner.py + runs.toml, never call trainer.py directly):
```bash
python /root/transformer/runner.py /root/transformer/runs.toml <profile> \
    --input_list /root/transformer/raw_data/btm_machines/machine3/input_list.json \
    --resume --start_epoch <N> \
    --model_file /root/transformer/runs/<run_dir>/model.pth \
    2>&1 | tee -a /root/transformer/raw_data/btm_machines/machine3/train_<name>.log
```

**Key flags:**
- `--resume` — required when loading a pretrained model for continuation (without it, trainer.py treats --model_file as inference-only)
- `--start_epoch N` — sets the epoch counter
- `--model_file` — path to checkpoint to load
- `--input_list` — JSON array of shard file paths to cycle through

**Output per run:** `runs/<profile>_<timestamp>/model.pth` and `train.log`

---

## Data / Slicing

**Machine3 input_list.json** cycles through 4 shard files:
```
gs12_m3_s2.txt → gs13_m3_s3.txt → gs14_m3_s4.txt → gs15_m3_s5.txt
```

**Slice offset algorithm** — each shard file has ~457M tokens. `max_tokens=20M` caps per file. The slice is controlled by offset into the source file:
| Slice | Offset | Epochs |
|---|---|---|
| 0 | 0 | 0–4 |
| 1 | 20M | 5–9 |
| 2 | 40M | 10–14 |
| 3 | 60M | 15–19 |

Current runs use **slice 2, offset 40M**.

**Steps per epoch:** 124,999 (at batch=160, window=256, max_tokens=20M, machine3 4-file input_list)  
**Tokens per epoch:** ~20M

---

## Run 4 — Current State (as of May 22, 2026)

### Model lineage
```
w64 (7 ep, 4 machines) → BTM merge → w128 cont (ep11-12) → extend_context 128→256
→ w256 ep13 (loss 5.6670) → extend_dims 256→384 → d384 ep14 (loss 5.7343, killed)
```

### Key checkpoints (server)
| Checkpoint | Path | Loss | Notes |
|---|---|---|---|
| btm_w256_cont ep13 | `runs/btm_w256_cont_20260521_054026/model.pth` | 5.6670 | pre-expansion baseline |
| model_d384.pth | `runs/btm_w256_cont_20260521_054026/model_d384.pth` | — | extend_dims output, epoch tag 13 |
| d384 ep14 | `runs/btm_d384_cont_20260521_142729/model.pth` | 5.7343 | warm-start ep14 done |

### Key checkpoints (local backups)
| File | Loss |
|---|---|
| `btm_r2_backups/btm_w256_cont_ep13.pth` | 5.6670 |

### Active experiments
| Run | Profile | cmd | Goal |
|---|---|---|---|
| d384 warm-start | btm_d384_cont | cmd_run_d384.sh | ep14+ from model_d384.pth |
| d384 cold start | btm_d384_cont | cmd_run_d384_scratch.sh | random init, same arch+data — control |

---

## Architecture

### Standard w256 (32M params)
- vecDims=256, num_heads=4, num_layers=8, window=256
- Embedding [50257×256] + Output [50257×256] = 25.8M frozen
- Transformer core = ~6.3M

### d384 expanded (40M params)
- vecDims=256, **inner_dims=384**
- Sandwich: embed[V,256] → upscale[256→384] → 8× transformer[384] → downscale[384→256] → output[V,256]
- Embeddings + output: 25.8M frozen
- Trainable: upscale + transformer blocks + downscale = ~14.4M

### extend_dims.py warm-start
Identity matrix in first 256×256 block, zeros for extra 128 dims. Preserves pre-extension function at init; initial loss bump of ~0.83 observed at ep14 start.

---

## runs.toml Profiles (local + server in sync)

Key profiles: `btm_w64`, `btm_w128`, `btm_w128_cont`, `btm_w256_cont`, `btm_d384_cont`

**Never write runs.toml content over SSH heredoc** — inner double-quotes get stripped. Always write locally and upload with scp.

---

## Loss Reference

| Phase | Epoch | Loss |
|---|---|---|
| w64 saturated (M1) | 7 | 6.0920 |
| w128 branch (M3) | 1 | 5.9133 |
| BTM merge | — | — |
| post-merge | 12 | 5.7632 |
| w256 | 13 | 5.6670 |
| d384 warm-start | 14 | 5.7343 |
| d384 cold start | 1+ | in progress |

---

## Tokenizer / Vocab

- tiktoken gpt2, vocab size 50,257
- vocab_server (OWT-rebuilt) — used by all server runs from BTM onward
- vocab_local (WikiText-103) — local runs only
- These are NOT interchangeable (99.6% of shared tokens have different indices)
