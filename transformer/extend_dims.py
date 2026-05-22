
# --------------------------------------------------
# Tech Aarvam
# Copyright (c) 2026 Tech Aarvam.
# Author: Ram (Ramasubramanian B)
# --------------------------------------------------
# Expands the transformer inner dimension (vecDims → inner_dims) while keeping
# embedding and outputLinear weights frozen-ready.
#
# Architecture after expansion:
#   embedding [V, d]  (frozen)  →  upscale [d→d']  →  transformer @ d'
#   →  downscale [d'→d]  →  outputLinear [d, V]  (frozen)
#
# Warm-start strategy:
#   upscale   : identity in first-d rows, zeros below  → signal passes through unchanged on day 0
#   downscale : identity in first-d cols, zeros right  → only first-d transformer dims feed output
#   attn/Wo/FFN: old weights padded into top-left block, new dims initialised to zero
#   LayerNorm weights: old values kept, new dims set to 1.0 (LN default)
#
# Usage:
#   python extend_dims.py input.pth output.pth 512
#
# Then train with:
#   python trainer.py ... --inner_dims 512 --model_file output.pth --vecDims <original_d>
# --------------------------------------------------

import torch
import sys
import re


def extend_dims(input_path, output_path, inner_dims):
    ckpt  = torch.load(input_path, map_location='cpu')
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt

    d  = state['embedding.weight'].shape[1]
    dp = inner_dims

    if dp <= d:
        print(f"ERROR: inner_dims ({dp}) must be greater than current vecDims ({d})")
        sys.exit(1)

    num_heads  = state['attentionHeads.0.keys'].shape[0]
    head_dim   = d  // num_heads
    head_dim_p = dp // num_heads

    print(f"vecDims:   {d}  →  inner_dims: {dp}")
    print(f"num_heads: {num_heads},  head_dim: {head_dim} → {head_dim_p}")

    FROZEN = {'embedding.weight', 'posEmbedding.weight', 'outputLinear.weight', 'outputLinear.bias'}

    def pad(t, new_shape, fill=0.0):
        out = torch.full(new_shape, fill, dtype=t.dtype)
        out[tuple(slice(0, s) for s in t.shape)] = t
        return out

    new_state = {}

    # frozen weights — copy unchanged
    for k in FROZEN:
        if k in state:
            new_state[k] = state[k].clone()

    # upscale weight [dp, d]  (Linear(d, dp, bias=False) stores [out, in])
    up = torch.zeros(dp, d)
    up[:d, :] = torch.eye(d)
    new_state['upscale.weight'] = up

    # downscale weight [d, dp]  (Linear(dp, d, bias=False) stores [out, in])
    down = torch.zeros(d, dp)
    down[:, :d] = torch.eye(d)
    new_state['downscale.weight'] = down

    for key, val in state.items():
        if key in new_state:
            continue

        if 'mask' in key:
            new_state[key] = val
            continue

        # attention Q / K / V: [H, d, head_dim] → [H, dp, head_dim_p]
        if re.search(r'attentionHeads\.\d+\.(keys|query|value)$', key):
            new_state[key] = pad(val, (num_heads, dp, head_dim_p))
            continue

        # Wo: [d, d] → [dp, dp]
        if re.search(r'^Wo\.\d+$', key):
            new_state[key] = pad(val, (dp, dp))
            continue

        # LayerNorm weight (default 1.0 for new dims)
        if re.search(r'norm[12]\.\d+\.weight$', key):
            new_state[key] = pad(val, (dp,), fill=1.0)
            continue

        # LayerNorm bias (default 0.0)
        if re.search(r'norm[12]\.\d+\.bias$', key):
            new_state[key] = pad(val, (dp,))
            continue

        # MLP first linear weight [4d, d] → [4dp, dp]
        if re.search(r'mlp\.\d+\.0\.weight$', key):
            new_state[key] = pad(val, (4 * dp, dp))
            continue

        # MLP first linear bias [4d] → [4dp]
        if re.search(r'mlp\.\d+\.0\.bias$', key):
            new_state[key] = pad(val, (4 * dp,))
            continue

        # MLP second linear weight [d, 4d] → [dp, 4dp]
        if re.search(r'mlp\.\d+\.2\.weight$', key):
            new_state[key] = pad(val, (dp, 4 * dp))
            continue

        # MLP second linear bias [d] → [dp]
        if re.search(r'mlp\.\d+\.2\.bias$', key):
            new_state[key] = pad(val, (dp,))
            continue

        # custom norm params (use_custom_norm=True path, rarely used)
        if re.search(r'learned(Mean|Std)', key):
            fill = 1.0 if 'Scale' in key else 0.0
            new_state[key] = pad(val.reshape(-1), (dp,), fill=fill).reshape(1, 1, dp)
            continue

        print(f"WARNING: unhandled key '{key}' shape={list(val.shape)} — copying as-is")
        new_state[key] = val

    epoch = ckpt.get('epoch', 0) if isinstance(ckpt, dict) else 0
    torch.save({'model': new_state, 'epoch': epoch}, output_path)

    orig = sum(v.numel() for v in state.values()     if v.dtype != torch.bool)
    new  = sum(v.numel() for v in new_state.values() if v.dtype != torch.bool)
    frozen = sum(state[k].numel() for k in FROZEN if k in state)
    print(f"Params (total):    {orig:>12,}  →  {new:>12,}")
    print(f"Params (frozen):                      {frozen:>12,}  (embedding + outputLinear)")
    print(f"Params (trainable):                   {new - frozen:>12,}")
    print(f"Saved → {output_path}  (epoch={epoch})")
    print(f"\nTrain with:  --inner_dims {dp} --model_file {output_path} --vecDims {d}")


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python extend_dims.py input.pth output.pth inner_dims")
        sys.exit(1)
    extend_dims(sys.argv[1], sys.argv[2], int(sys.argv[3]))
