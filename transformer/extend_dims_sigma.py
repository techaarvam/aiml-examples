
# --------------------------------------------------
# Tech Aarvam
# Copyright (c) 2026 Tech Aarvam.
# Author: Ram (Ramasubramanian B)
# --------------------------------------------------
# Expands transformer inner dimension using SVD sigma-based noise.
#
# Strategy:
#   Old weights are preserved EXACTLY in the top-left block (rotations intact).
#   New-new diagonal entries are initialised to 0.01 × median(σ) of the source
#   matrix — proportional to its actual spectral scale, not weight std.
#
#   W [m, n]  →  W' [mp, np]:
#     W'[:m, :n]       = W                  (old block, exact)
#     W'[m+i, n+i]     = 0.01 × median(σ)  (new-new diagonal)
#     all cross blocks  = 0
#
#   Upscale/downscale: carry trained weights, zeros for new rows/cols.
#   LayerNorm: replicate old values, new dims get 1.0/0.0 defaults.
#   Biases: carry old values, new dims get tiny sigma-based noise.
#
# Usage:
#   python extend_dims_sigma.py input.pth output.pth 512
# --------------------------------------------------

import torch
import sys
import re
import numpy as np


def _sigma_noise(val):
    """0.01 × median non-zero singular value of val."""
    s = np.linalg.svd(val.float().numpy(), compute_uv=False)
    s = s[s > 0]
    return float(np.median(s)) * 0.01 if len(s) else 1e-4


def _diag_expand(val, new_rows, new_cols):
    """
    Expand 2D weight val [r, c] → [new_rows, new_cols].
    Top-left [r, c] = val exactly.
    New-new diagonal = 0.01 × median(sigma).
    """
    r, c = val.shape
    out  = torch.zeros(new_rows, new_cols, dtype=val.dtype)
    out[:r, :c] = val
    ns   = _sigma_noise(val)
    n_new = min(new_rows - r, new_cols - c)
    if n_new > 0:
        idx = torch.arange(n_new)
        out[r + idx, c + idx] = ns
    return out


def extend_dims_sigma(input_path, output_path, inner_dims, reset_epoch=True):
    ckpt  = torch.load(input_path, map_location='cpu')
    raw   = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    state = {k.replace('_orig_mod.', ''): v for k, v in raw.items()}

    vec_d = state['embedding.weight'].shape[1]

    # Detect current inner_dims from actual weight shapes
    if 'Wo.0' in state:
        d = state['Wo.0'].shape[0]
    elif 'mlp.0.fc1.weight' in state:
        d = state['mlp.0.fc1.weight'].shape[1]
    elif 'mlp.0.0.weight' in state:
        d = state['mlp.0.0.weight'].shape[1]
    else:
        raise RuntimeError("Cannot detect current inner_dims from checkpoint")

    dp = inner_dims

    if dp <= d:
        print(f"ERROR: inner_dims ({dp}) must be > current inner_dims ({d})")
        sys.exit(1)

    fused_qkv = 'attentionHeads.0.qkv' in state

    print(f"vec_d:    {vec_d}  (embedding dim, frozen)")
    print(f"inner_d:  {d}  →  {dp}")
    print(f"QKV mode: {'fused' if fused_qkv else 'unfused'}")
    print(f"Strategy: SVD sigma-based diagonal noise (0.01 × median σ per matrix)")

    FROZEN = {'embedding.weight', 'posEmbedding.weight', 'outputLinear.weight', 'outputLinear.bias'}

    new_state = {}

    for k in FROZEN:
        if k in state:
            new_state[k] = state[k].clone()

    # upscale [d, vec_d] → [dp, vec_d]: carry trained rows, zeros for new rows
    old_up = state.get('upscale.weight')
    if old_up is not None:
        up = torch.zeros(dp, vec_d, dtype=old_up.dtype)
        up[:d] = old_up
    else:
        up = torch.zeros(dp, vec_d)
        up[:vec_d] = torch.eye(vec_d)
    new_state['upscale.weight'] = up

    # downscale [vec_d, d] → [vec_d, dp]: carry trained cols, zeros for new cols
    old_down = state.get('downscale.weight')
    if old_down is not None:
        down = torch.zeros(vec_d, dp, dtype=old_down.dtype)
        down[:, :d] = old_down
    else:
        down = torch.zeros(vec_d, dp)
        down[:, :vec_d] = torch.eye(vec_d)
    new_state['downscale.weight'] = down

    for key, val in state.items():
        if key in new_state:
            continue
        if 'mask' in key:
            new_state[key] = val
            continue

        # ── Fused QKV [d, 3d] → [dp, 3dp] ─────────────────────────────────
        if re.search(r'attentionHeads\.\d+\.qkv$', key):
            new_state[key] = _diag_expand(val, dp, 3 * dp)
            continue

        # ── Unfused Q/K/V [H, d, hd] → [H, dp, hdp] ───────────────────────
        if re.search(r'attentionHeads\.\d+\.(keys|query|value)$', key):
            num_heads = val.shape[0]
            head_d    = d  // num_heads
            head_dp   = dp // num_heads
            new_val   = torch.zeros(num_heads, dp, head_dp, dtype=val.dtype)
            for h in range(num_heads):
                new_val[h] = _diag_expand(val[h], dp, head_dp)
            new_state[key] = new_val
            continue

        # ── Wo [d, d] → [dp, dp] ───────────────────────────────────────────
        if re.search(r'^Wo\.\d+$', key):
            new_state[key] = _diag_expand(val, dp, dp)
            continue

        # ── LayerNorm weight [d] → [dp]: old values + 1.0 for new dims ─────
        if re.search(r'norm[12]\.\d+\.weight$', key):
            nv      = torch.ones(dp, dtype=val.dtype)
            nv[:d]  = val
            new_state[key] = nv
            continue

        # ── LayerNorm bias [d] → [dp]: old values + 0.0 for new dims ───────
        if re.search(r'norm[12]\.\d+\.bias$', key):
            nv      = torch.zeros(dp, dtype=val.dtype)
            nv[:d]  = val
            new_state[key] = nv
            continue

        # ── MLP fc1 weight [4d, d] → [4dp, dp] ─────────────────────────────
        if re.search(r'mlp\.\d+\.fc1\.weight$', key):
            new_state[key] = _diag_expand(val, 4 * dp, dp)
            continue

        # ── MLP fc1 bias [4d] → [4dp] ───────────────────────────────────────
        if re.search(r'mlp\.\d+\.fc1\.bias$', key):
            nv        = torch.zeros(4 * dp, dtype=val.dtype)
            nv[:4*d]  = val
            new_state[key] = nv
            continue

        # ── MLP fc2 weight [d, 4d] → [dp, 4dp] ─────────────────────────────
        if re.search(r'mlp\.\d+\.fc2\.weight$', key):
            new_state[key] = _diag_expand(val, dp, 4 * dp)
            continue

        # ── MLP fc2 bias [d] → [dp] ──────────────────────────────────────────
        if re.search(r'mlp\.\d+\.fc2\.bias$', key):
            nv      = torch.zeros(dp, dtype=val.dtype)
            nv[:d]  = val
            new_state[key] = nv
            continue

        # ── custom norm (rare) ───────────────────────────────────────────────
        if re.search(r'learned(Mean|Std)', key):
            fill = 1.0 if 'Scale' in key else 0.0
            v    = val.reshape(-1)
            nv   = torch.full((dp,), fill, dtype=v.dtype)
            nv[:d] = v
            new_state[key] = nv.reshape(1, 1, dp)
            continue

        print(f"WARNING: unhandled key '{key}' shape={list(val.shape)} — copying as-is")
        new_state[key] = val

    saved_epoch = 0 if reset_epoch else (ckpt.get('epoch', 0) if isinstance(ckpt, dict) else 0)
    torch.save({'model': new_state, 'epoch': saved_epoch}, output_path)

    orig   = sum(v.numel() for v in state.values()     if v.dtype != torch.bool)
    nw     = sum(v.numel() for v in new_state.values() if v.dtype != torch.bool)
    frozen = sum(state[k].numel() for k in FROZEN if k in state)
    print(f"Params (total):     {orig:>12,}  →  {nw:>12,}")
    print(f"Params (frozen):                       {frozen:>12,}")
    print(f"Params (trainable):                    {nw - frozen:>12,}")
    print(f"Saved → {output_path}  (epoch={saved_epoch})")
    print(f"\nTrain with:  --inner_dims {dp} --model_file {output_path} --vecDims {vec_d}")


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python extend_dims_sigma.py input.pth output.pth inner_dims [--keep-epoch]")
        sys.exit(1)
    reset = '--keep-epoch' not in sys.argv
    extend_dims_sigma(sys.argv[1], sys.argv[2], int(sys.argv[3]), reset_epoch=reset)
