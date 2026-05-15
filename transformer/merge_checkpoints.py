import torch
import sys

# Usage:
#   python merge_checkpoints.py out.pth model_A.pth model_B.pth model_C.pth model_D.pth
#
# Averages model weights from N checkpoints saved by trainer.py.
# Optimizer state is reset — Adam moments are inconsistent across
# machines that trained on different data, so starting fresh is safer.

def merge(output_path, *input_paths):
    print(f"Merging {len(input_paths)} checkpoints → {output_path}")

    states = []
    last_epoch = 0
    for path in input_paths:
        ckpt = torch.load(path, map_location="cpu")
        model_state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        states.append({k: v.float() for k, v in model_state.items()})
        if isinstance(ckpt, dict):
            last_epoch = max(last_epoch, ckpt.get("epoch", 0))
        print(f"  loaded {path}")

    merged = {}
    for key in states[0].keys():
        merged[key] = sum(s[key] for s in states) / len(states)

    torch.save({"model": merged, "epoch": last_epoch}, output_path)
    print(f"Saved merged checkpoint to {output_path}  (epoch={last_epoch}, optimizer reset)")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python merge_checkpoints.py out.pth a.pth b.pth [c.pth ...]")
        sys.exit(1)
    merge(sys.argv[1], *sys.argv[2:])
