#!/bin/bash
# BTM d1024 local — ep31–201 from d512 ep30 (diagonal-only noise init, ~100% Chinchilla at ep201)
# Arch : vecDims=256, inner_dims=1024, 8 layers, 4 heads, window=256
# Model: 127M total (101M trainable + 26M frozen embedding/output)
# VRAM : batch=256 estimated safe for 12 GB — bump to 320/384 if headroom allows
# Watch: python3 watch_checkpoint_analysis.py <run_dir> --every-n-epochs 10 --phase d1024s --label-prefix d1024
cd /home/rambala/work/learn/aiml/transformer
systemd-inhibit --what=idle:sleep --why="d1024 local training" \
python runner.py runs.toml btm_d1024_local \
    --resume \
    --start_epoch 43 \
    --model_file runs/btm_d1024_local_20260531_170625/model.pth
