"""
convert_unfused_to_fused.py
---------------------------
Converts a checkpoint saved with unfused QKV (separate keys/query/value params)
to the fused QKV layout (single qkv param) used by flash_attn.

Unfused layout per attention layer:
    attentionHeads.N.keys   [H, D, head_dim]
    attentionHeads.N.query  [H, D, head_dim]
    attentionHeads.N.value  [H, D, head_dim]

Fused layout:
    attentionHeads.N.qkv    [D, 3*D]   (D = H * head_dim = innerDims)

Usage:
    python convert_unfused_to_fused.py <input.pth> <output.pth>
"""

import sys
import torch

def convert(src_path, dst_path):
    ckpt = torch.load(src_path, map_location='cpu')
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt

    # Strip torch.compile prefix if present
    state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}

    new_state = {}
    converted = 0

    # Collect all attention layer indices
    layer_indices = set()
    for k in state:
        if k.startswith('attentionHeads.') and k.endswith('.keys'):
            idx = k.split('.')[1]
            layer_indices.add(idx)

    for idx in sorted(layer_indices, key=int):
        prefix = f'attentionHeads.{idx}'
        query = state[f'{prefix}.query']   # [H, D, head_dim]
        keys  = state[f'{prefix}.keys']    # [H, D, head_dim]
        value = state[f'{prefix}.value']   # [H, D, head_dim]

        H, D, head_dim = query.shape
        innerDims = H * head_dim
        assert innerDims == D, f"Layer {idx}: H*head_dim ({innerDims}) != D ({D})"

        # permute [H, D, head_dim] → [D, H, head_dim] → reshape [D, H*head_dim]
        Q_block = query.permute(1, 0, 2).reshape(D, H * head_dim)
        K_block = keys.permute(1, 0, 2).reshape(D, H * head_dim)
        V_block = value.permute(1, 0, 2).reshape(D, H * head_dim)

        new_state[f'{prefix}.qkv'] = torch.cat([Q_block, K_block, V_block], dim=1)
        converted += 1
        print(f"  Layer {idx}: query/keys/value [{H},{D},{head_dim}] → qkv [{D},{3*innerDims}]")

    # Copy all other keys unchanged
    skip = {'query', 'keys', 'value'}
    for k, v in state.items():
        parts = k.split('.')
        if not (parts[0] == 'attentionHeads' and parts[-1] in skip):
            new_state[k] = v

    # Preserve checkpoint envelope; drop optimizer state (incompatible after layout change)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        ckpt['model'] = new_state
        had_optimizer = 'optimizer' in ckpt
        ckpt.pop('optimizer', None)
        if had_optimizer:
            print("  Dropped optimizer state (incompatible with new param layout — will start fresh)")
        torch.save(ckpt, dst_path)
    else:
        torch.save(new_state, dst_path)

    print(f"\nConverted {converted} attention layers.")
    print(f"Saved → {dst_path}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python convert_unfused_to_fused.py <input.pth> <output.pth>")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
