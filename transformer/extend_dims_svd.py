
# --------------------------------------------------
# Tech Aarvam
# Copyright (c) 2026 Tech Aarvam.
# Author: Ram (Ramasubramanian B)
# --------------------------------------------------
# Expands transformer inner dimension using SVD block-diagonal init.
#
# Strategy (exact doubling d → 2d):
#   W [m, n]  →  [[W/√2,   0  ],    shape [2m, 2n]
#                 [  0,  W/√2 ]]
#
#   This replicates the singular directions of W in both blocks with σ
#   scaled by 1/√2, so:
#     - Frobenius norm is preserved: 2×(σ/√2)² = σ²
#     - Both halves of all matrices are active from step 1
#     - Condition number of Wo/FFN is identical to the original (no inflation)
#     - Gradient flows through all new dims immediately
#
#   Biases: replicated exactly [b; b] for exact doubling.
#   LayerNorm: replicated [w; w] (both halves same normalisation).
#   Upscale/downscale: replicated column-stacked (both halves see input signal).
#
# Usage:
#   python extend_dims_svd.py input.pth output.pth 1024
#
# Then train with:
#   python runner.py runs.toml <profile> --resume --start_epoch N \
#       --model_file output.pth
# --------------------------------------------------

import torch
import sys
import re


def block_expand_2d(val, new_rows, new_cols):
    """
    Expand 2D weight val [r, c] → [new_rows, new_cols].
    Exact doubling (new_rows==2r, new_cols==2c): block-diagonal [[val/√2, 0], [0, val/√2]].
    Non-doubling fallback: top-left copy + diagonal noise in new-new block.
    """
    r, c = val.shape
    out = torch.zeros(new_rows, new_cols, dtype=val.dtype)
    if new_rows == 2 * r and new_cols == 2 * c:
        scaled = val * (2.0 ** -0.5)
        out[:r, :c] = scaled
        out[r:,  c:] = scaled
    else:
        out[:r, :c] = val
        ns  = val.std().item() * 0.01
        idx = torch.arange(min(new_rows - r, new_cols - c))
        out[r + idx, c + idx] = ns
    return out


def replicate_rows(val, new_rows):
    """
    Expand matrix val [r, c] → [new_rows, c] by replicating rows (scaled 1/√2).
    Used for upscale weight where only row dim doubles.
    """
    r, c = val.shape
    out = torch.zeros(new_rows, c, dtype=val.dtype)
    if new_rows == 2 * r:
        scaled = val * (2.0 ** -0.5)
        out[:r] = scaled
        out[r:] = scaled
    else:
        out[:r] = val
        ns = val.std().item() * 0.01
        out[r:] = torch.randn(new_rows - r, c, dtype=val.dtype) * ns
    return out


def replicate_cols(val, new_cols):
    """
    Expand matrix val [r, c] → [r, new_cols] by replicating cols (scaled 1/√2).
    Used for downscale weight where only col dim doubles.
    """
    r, c = val.shape
    out = torch.zeros(r, new_cols, dtype=val.dtype)
    if new_cols == 2 * c:
        scaled = val * (2.0 ** -0.5)
        out[:, :c] = scaled
        out[:, c:] = scaled
    else:
        out[:, :c] = val
        ns = val.std().item() * 0.01
        out[:, c:] = torch.randn(r, new_cols - c, dtype=val.dtype) * ns
    return out


def replicate_1d(val, new_len):
    """
    Expand 1D tensor val [n] → [new_len] by replicating (exact doubling: [v; v]).
    """
    n = val.shape[0]
    out = torch.zeros(new_len, dtype=val.dtype)
    out[:n] = val
    if new_len == 2 * n:
        out[n:] = val
    else:
        ns = (val.std().item() or 0.01) * 0.01
        out[n:] = torch.randn(new_len - n, dtype=val.dtype) * ns
    return out


def extend_dims_svd(input_path, output_path, inner_dims):
    ckpt  = torch.load(input_path, map_location='cpu')
    raw   = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    state = {k.replace('_orig_mod.', ''): v for k, v in raw.items()}

    vec_d = state['embedding.weight'].shape[1]     # embedding dim (frozen)

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
    exact     = (dp == 2 * d)

    print(f"vec_d:     {vec_d}  (embedding dim, frozen)")
    print(f"inner_d:   {d}  →  {dp}")
    print(f"QKV mode:  {'fused' if fused_qkv else 'unfused'}")
    print(f"Strategy:  {'block-diagonal SVD (exact ×2)' if exact else 'top-left + diagonal noise (non-doubling)'}")
    if not exact:
        print(f"WARNING: {dp} ≠ 2×{d}, falling back to diagonal noise for non-doubled dims")

    FROZEN = {'embedding.weight', 'posEmbedding.weight', 'outputLinear.weight', 'outputLinear.bias'}

    new_state = {}

    for k in FROZEN:
        if k in state:
            new_state[k] = state[k].clone()

    # upscale [d, vec_d] → [dp, vec_d]: replicate rows (both halves see input)
    new_state['upscale.weight'] = replicate_rows(state['upscale.weight'], dp)

    # downscale [vec_d, d] → [vec_d, dp]: replicate cols (both halves contribute)
    new_state['downscale.weight'] = replicate_cols(state['downscale.weight'], dp)

    for key, val in state.items():
        if key in new_state:
            continue

        if 'mask' in key:
            new_state[key] = val
            continue

        # ── Fused QKV [d, 3d] → [dp, 3dp] ─────────────────────────────────
        if re.search(r'attentionHeads\.\d+\.qkv$', key):
            new_state[key] = block_expand_2d(val, dp, 3 * dp)
            continue

        # ── Unfused Q/K/V [H, d, hd] → [H, dp, hdp] ───────────────────────
        if re.search(r'attentionHeads\.\d+\.(keys|query|value)$', key):
            num_heads = val.shape[0]
            head_d    = d  // num_heads
            head_dp   = dp // num_heads
            new_val   = torch.zeros(num_heads, dp, head_dp, dtype=val.dtype)
            for h in range(num_heads):
                new_val[h] = block_expand_2d(val[h], dp, head_dp)
            new_state[key] = new_val
            continue

        # ── Wo [d, d] → [dp, dp] ───────────────────────────────────────────
        if re.search(r'^Wo\.\d+$', key):
            new_state[key] = block_expand_2d(val, dp, dp)
            continue

        # ── LayerNorm weight [d] → [dp]: replicate ─────────────────────────
        if re.search(r'norm[12]\.\d+\.weight$', key):
            new_state[key] = replicate_1d(val, dp)
            continue

        # ── LayerNorm bias [d] → [dp]: replicate (near zero) ───────────────
        if re.search(r'norm[12]\.\d+\.bias$', key):
            new_state[key] = replicate_1d(val, dp)
            continue

        # ── MLP fc1 weight [4d, d] → [4dp, dp] ─────────────────────────────
        if re.search(r'mlp\.\d+\.fc1\.weight$', key):
            new_state[key] = block_expand_2d(val, 4 * dp, dp)
            continue

        # ── MLP fc1 bias [4d] → [4dp] ───────────────────────────────────────
        if re.search(r'mlp\.\d+\.fc1\.bias$', key):
            new_state[key] = replicate_1d(val, 4 * dp)
            continue

        # ── MLP fc2 weight [d, 4d] → [dp, 4dp] ─────────────────────────────
        if re.search(r'mlp\.\d+\.fc2\.weight$', key):
            new_state[key] = block_expand_2d(val, dp, 4 * dp)
            continue

        # ── MLP fc2 bias [d] → [dp] ──────────────────────────────────────────
        if re.search(r'mlp\.\d+\.fc2\.bias$', key):
            new_state[key] = replicate_1d(val, dp)
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

    epoch = ckpt.get('epoch', 0) if isinstance(ckpt, dict) else 0
    torch.save({'model': new_state, 'epoch': epoch}, output_path)

    orig   = sum(v.numel() for v in state.values()     if v.dtype != torch.bool)
    nw     = sum(v.numel() for v in new_state.values() if v.dtype != torch.bool)
    frozen = sum(state[k].numel() for k in FROZEN if k in state)
    print(f"Params (total):     {orig:>12,}  →  {nw:>12,}")
    print(f"Params (frozen):                       {frozen:>12,}  (embedding + outputLinear)")
    print(f"Params (trainable):                    {nw - frozen:>12,}")
    print(f"Saved → {output_path}  (epoch={epoch})")
    print(f"\nTrain with:  --inner_dims {dp} --model_file {output_path} --vecDims {vec_d}")


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python extend_dims_svd.py input.pth output.pth inner_dims")
        sys.exit(1)
    extend_dims_svd(sys.argv[1], sys.argv[2], int(sys.argv[3]))
