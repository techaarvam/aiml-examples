import torch
import torch.nn.functional as F
import sys

# Usage:
#   python extend_context.py input.pth output.pth 1024
#
# Converts a checkpoint trained with window_size=W to window_size=NEW_W.
# All weights transfer directly except posEmbedding, which is interpolated.

def extend_context(input_path, output_path, new_window):
    ckpt = torch.load(input_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    old_pos = state["posEmbedding.weight"]          # [old_window, vecDims]
    old_window, vecDims = old_pos.shape
    print(f"posEmbedding: [{old_window}, {vecDims}] → [{new_window}, {vecDims}]")

    # Interpolate along position axis: reshape to [1, vecDims, old_window] for F.interpolate
    new_pos = F.interpolate(
        old_pos.float().T.unsqueeze(0),             # [1, vecDims, old_window]
        size=new_window,
        mode="linear",
        align_corners=True,
    ).squeeze(0).T                                  # [new_window, vecDims]

    new_mask = torch.triu(torch.ones(new_window - 1, new_window - 1), diagonal=1).bool()
    new_state = dict(state)
    for k in state:
        if 'mask' in k:
            new_state[k] = new_mask
    new_state["posEmbedding.weight"] = new_pos

    epoch = ckpt.get("epoch", 0) if isinstance(ckpt, dict) else 0
    torch.save({"model": new_state, "epoch": epoch}, output_path)
    print(f"Saved to {output_path}  (epoch={epoch}, window {old_window}→{new_window})")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python extend_context.py input.pth output.pth new_window_size")
        sys.exit(1)
    extend_context(sys.argv[1], sys.argv[2], int(sys.argv[3]))
