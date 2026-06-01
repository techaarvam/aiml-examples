#!/bin/bash
# BTM d512 local continuation — ep16+ from ep15 backup
# Arch : vecDims=256, inner_dims=512, 8 layers, 4 heads, window=256
# Data : machine3 shards gs11–gs15 (local)
# VRAM : batch=256 ~3 GB — safe for 12GB; bump to 512 if headroom allows
#    --reset_optimizer_every_epoch \
cd /home/rambala/work/learn/aiml/transformer
systemd-inhibit --what=idle:sleep --why="d512 local training" \
python runner.py runs.toml btm_d512_cont_local \
    --resume \
    --start_epoch 21 \
    --model_file /home/rambala/work/learn/aiml/transformer/runs/btm_d512_cont_local_20260529_181714/model.pth
