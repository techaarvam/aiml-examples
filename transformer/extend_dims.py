
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
#   FFN_down (mlp.2): new columns/rows get small noise (0.01 × std) to break zero-gradient deadlock
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
    raw   = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    state = {k.replace('_orig_mod.', ''): v for k, v in raw.items()}

    vec_d = state['embedding.weight'].shape[1]   # embedding dim (frozen, not inner dim)
    dp    = inner_dims

    # Detect current inner_dims from actual weight shapes, not embedding dim
    if 'Wo.0' in state:
        d = state['Wo.0'].shape[0]
    elif 'mlp.0.fc1.weight' in state:
        d = state['mlp.0.fc1.weight'].shape[1]
    elif 'mlp.0.0.weight' in state:
        d = state['mlp.0.0.weight'].shape[1]
    else:
        raise RuntimeError("Cannot detect current inner_dims from checkpoint keys")

    if dp <= d:
        print(f"ERROR: inner_dims ({dp}) must be greater than current inner_dims ({d})")
        sys.exit(1)

    # Detect fused vs unfused QKV
    fused_qkv = 'attentionHeads.0.qkv' in state

    if fused_qkv:
        num_heads  = None   # not needed for fused path
        head_dim_p = None
    else:
        num_heads  = state['attentionHeads.0.keys'].shape[0]
        head_dim   = d  // num_heads
        head_dim_p = dp // num_heads

    print(f"vec_d:     {vec_d}  (embedding dim, frozen)")
    print(f"inner_d:   {d}  →  {dp}")
    print(f"QKV mode:  {'fused' if fused_qkv else 'unfused'}")
    if not fused_qkv:
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

    # upscale [d, vec_d] → [dp, vec_d]: carry existing trained rows, zeros for new dims
    old_up = state.get('upscale.weight')
    if old_up is not None:
        up = torch.zeros(dp, vec_d, dtype=old_up.dtype)
        up[:d] = old_up
    else:
        up = torch.zeros(dp, vec_d)
        up[:vec_d] = torch.eye(vec_d)
    new_state['upscale.weight'] = up

    # downscale [vec_d, d] → [vec_d, dp]: carry existing trained cols, zeros for new dims
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

        # fused QKV: [d, 3*d] → [dp, 3*dp]
        # Diagonal of new-new block for Q, K, V independently.
        if re.search(r'attentionHeads\.\d+\.qkv$', key):
            new_val = pad(val, (dp, 3 * dp))
            ns  = val.std().item() * 0.01
            idx = torch.arange(dp - d)
            new_val[d + idx,      d + idx] = ns   # Q new-new diagonal
            new_val[d + idx, dp + d + idx] = ns   # K new-new diagonal
            new_val[d + idx, 2*dp + d + idx] = ns # V new-new diagonal
            new_state[key] = new_val
            continue

        # unfused Q / K / V: [H, d, head_dim] → [H, dp, head_dim_p]
        if re.search(r'attentionHeads\.\d+\.(keys|query|value)$', key):
            new_val = pad(val, (num_heads, dp, head_dim_p))
            ns  = val.std().item() * 0.01
            idx = torch.arange(min(dp - d, head_dim_p - head_dim))
            for h in range(num_heads):
                new_val[h, d + idx, head_dim + idx] = ns
            new_state[key] = new_val
            continue

        # Wo: [d, d] → [dp, dp]
        if re.search(r'^Wo\.\d+$', key):
            new_val = pad(val, (dp, dp))
            ns  = val.std().item() * 0.01
            idx = torch.arange(dp - d)
            new_val[d + idx, d + idx] = ns   # new-new diagonal only
            new_state[key] = new_val
            continue

        # LayerNorm weight: new dims → 1.0 + tiny noise (1D, no matrix diagonal)
        if re.search(r'norm[12]\.\d+\.weight$', key):
            new_val     = pad(val, (dp,), fill=1.0)
            ns          = val.std().item() * 0.01
            new_val[d:] = 1.0 + torch.randn(dp - d) * ns
            new_state[key] = new_val
            continue

        # LayerNorm bias: new dims → tiny noise (1D)
        if re.search(r'norm[12]\.\d+\.bias$', key):
            new_val     = pad(val, (dp,))
            ns          = (val.std().item() or 0.01) * 0.01
            new_val[d:] = torch.randn(dp - d) * ns
            new_state[key] = new_val
            continue

        # MLP fc1 weight [4d, d] → [4dp, dp]: diagonal of new-new block
        if re.search(r'mlp\.\d+\.fc1\.weight$', key):
            new_val = pad(val, (4 * dp, dp))
            ns  = val.std().item() * 0.01
            idx = torch.arange(dp - d)
            new_val[4*d + idx, d + idx] = ns
            new_state[key] = new_val
            continue

        # MLP fc1 bias [4d] → [4dp] (1D)
        if re.search(r'mlp\.\d+\.fc1\.bias$', key):
            new_val       = pad(val, (4 * dp,))
            ns            = val.std().item() * 0.01
            new_val[4*d:] = torch.randn(4 * (dp - d)) * ns
            new_state[key] = new_val
            continue

        # MLP fc2 weight [d, 4d] → [dp, 4dp]: diagonal of new-new block
        if re.search(r'mlp\.\d+\.fc2\.weight$', key):
            new_val = pad(val, (dp, 4 * dp))
            ns  = val.std().item() * 0.01
            idx = torch.arange(dp - d)
            new_val[d + idx, 4*d + idx] = ns
            new_state[key] = new_val
            continue

        # MLP fc2 bias [d] → [dp] (1D)
        if re.search(r'mlp\.\d+\.fc2\.bias$', key):
            new_val     = pad(val, (dp,))
            ns          = val.std().item() * 0.01
            new_val[d:] = torch.randn(dp - d) * ns
            new_state[key] = new_val
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

    orig   = sum(v.numel() for v in state.values()     if v.dtype != torch.bool)
    new    = sum(v.numel() for v in new_state.values() if v.dtype != torch.bool)
    frozen = sum(state[k].numel() for k in FROZEN if k in state)
    print(f"Params (total):    {orig:>12,}  →  {new:>12,}")
    print(f"Params (frozen):                      {frozen:>12,}  (embedding + outputLinear)")
    print(f"Params (trainable):                   {new - frozen:>12,}")
    print(f"Saved → {output_path}  (epoch={epoch})")
    print(f"\nTrain with:  --inner_dims {dp} --model_file {output_path} --vecDims {vec_d}")


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python extend_dims.py input.pth output.pth inner_dims")
        sys.exit(1)
    extend_dims(sys.argv[1], sys.argv[2], int(sys.argv[3]))
