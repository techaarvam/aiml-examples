# --------------------------------------------------
# Tech Aarvam
# Copyright (c) 2026 Tech Aarvam.
# Authors: Ram (Ramasubramanian B), Claude Code (Anthropic)
# --------------------------------------------------
import numpy as np
import random
from debug import *
from seeder import *
from validate import validate_dataset

# ── Constants ──────────────────────────────────────────────────────────────────

SEQ_LEN      = 128    # eighth-note timesteps per sample (8 bars of 4/4)
BLOCK_LEN    = 16     # eighths per rhythm block (2 bars of 4/4)
MELODY_CLAMP = 12     # ± clamp on raw cumulative sum
NUM_SAMPLES  = 2000

SCALE_NAMES  = ['C', 'D', 'E', 'F', 'G', 'A', 'B']

# ── Rhythm patterns ────────────────────────────────────────────────────────────

PATTERNS = {
    'straight_quarters': [(2,1),(2,1),(2,1),(2,1)],
    'marching':          [(2,2),(2,1),(2,2),(2,1)],
    'swing_feel':        [(3,2),(1,0),(3,2),(1,0)],
    'syncopated':        [(1,1),(2,2),(1,0),(2,2),(1,1),(1,0)],
    'long_short':        [(3,2),(1,1),(3,2),(1,1)],
    'gallop':            [(1,2),(1,0),(2,1),(1,2),(1,0),(2,1)],
    'held_release':      [(4,2),(2,1),(2,2)],
    'dense_run':         [(1,1),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1)],
    't332_run':          [(1,2),(1,1),(1,1),(1,2),(1,1),(1,1),(1,2),(1,1)],
    'slow332_run':       [(2,2),(2,1),(2,1),(2,2),(2,1),(2,1),(2,2),(2,1)],
    'sparse':            [(4,2),(4,1)],
    'push_pull':         [(1,0),(3,2),(1,0),(3,2)],
    'cadence':           [(2,1),(2,1),(4,2)],
    'ornament_run':      [(1,2),(1,1),(1,0),(1,1),(2,2),(2,1)],
    'thathimidhim':      [(2,1),(1,1),(1,1),(1,1),(2,1),(1,1),(2,1),(1,1)],
}

MAIN_PATTERNS = [k for k in PATTERNS if k != 'cadence']

# ── Melody generation ──────────────────────────────────────────────────────────

def generate_melody(seq_len, note_boundary):
    """
    Rhythm-coupled melody generator.

    Pitch advances by ±1 or 0 only at rhythm note boundaries; held constant
    within a note. This guarantees melody and rhythm are always in lockstep and
    consecutive melody_raw values never differ by more than 1.

    Returns (melody_raw_arr, scale_degree_arr).
    """
    running_pos      = 0
    melody_raw_arr   = np.zeros(seq_len, dtype=int)
    scale_degree_arr = np.zeros(seq_len, dtype=int)

    for t in range(seq_len):
        if note_boundary[t]:
            delta       = random.choice([-1, 0, 1])
            running_pos = int(np.clip(running_pos + delta, -MELODY_CLAMP, MELODY_CLAMP))
        melody_raw_arr[t]   = running_pos
        scale_degree_arr[t] = running_pos % 7

    return melody_raw_arr, scale_degree_arr

# ── Rhythm generation ──────────────────────────────────────────────────────────

def _fill_with_cadence(remaining):
    """Fill exactly `remaining` eighths: walk cadence notes, truncate last if needed, pad rests."""
    result = []
    used   = 0
    for (dur, acc) in PATTERNS['cadence']:
        if used >= remaining:
            break
        if used + dur > remaining:
            dur = remaining - used          # truncate this note to fit
        result.append((dur, acc))
        used += dur
    while used < remaining:                 # cadence ran short — pad with rests
        result.append((1, 0))
        used += 1
    return result


def generate_rhythm(seq_len=SEQ_LEN):
    """
    Generate accent, beat_pos, note_boundary, and note_dur arrays.

    note_boundary_arr : bool, True at the first eighth-note of each rhythm note
    note_dur_arr      : int,  duration in eighths of the note active at each timestep

    Both are needed by generate_melody to implement boundary-aware pitch selection.
    note_dur is also stored in the dataset as a useful RNN training feature.
    """
    assert seq_len % BLOCK_LEN == 0, f"seq_len {seq_len} must be divisible by BLOCK_LEN {BLOCK_LEN}"
    n_blocks = seq_len // BLOCK_LEN

    accent_arr        = np.zeros(seq_len, dtype=int)
    beat_pos_arr      = np.zeros(seq_len, dtype=float)
    note_boundary_arr = np.zeros(seq_len, dtype=bool)
    note_dur_arr      = np.zeros(seq_len, dtype=int)

    t = 0
    for block_idx in range(n_blocks):
        key     = random.choice(MAIN_PATTERNS)
        pattern = list(PATTERNS[key])
        pat_len = sum(d for d, _ in pattern)

        if pat_len > BLOCK_LEN:
            warn(f"Block {block_idx}: pattern '{key}' length {pat_len} > BLOCK_LEN {BLOCK_LEN}")

        remaining  = BLOCK_LEN - pat_len
        fill       = _fill_with_cadence(remaining) if remaining > 0 else []
        full_block = pattern + fill
        block_dur  = sum(d for d, _ in full_block)

        if block_dur != BLOCK_LEN:
            warn(f"Block {block_idx}: '{key}' total duration {block_dur} != {BLOCK_LEN}")

        for (dur, acc) in full_block:
            note_boundary_arr[t] = True          # first step of this note
            for step in range(dur):
                accent_arr[t]   = acc
                beat_pos_arr[t] = (t % 2) * 0.5
                note_dur_arr[t] = dur
                t += 1

    return accent_arr, beat_pos_arr, note_boundary_arr, note_dur_arr

# ── Sample / dataset generation ────────────────────────────────────────────────

def generate_sample(seq_len=SEQ_LEN):
    accent, beat_pos, note_boundary, note_dur = generate_rhythm(seq_len)
    melody_raw, scale_degree                  = generate_melody(seq_len, note_boundary)
    return {
        'scale_degree':  scale_degree,
        'melody_raw':    melody_raw,
        'accent':        accent,
        'beat_pos':      beat_pos,
        'note_dur':      note_dur,
        'note_boundary': note_boundary,
    }


def generate_dataset(num_samples=NUM_SAMPLES, seq_len=SEQ_LEN):
    info(f"Generating {num_samples} samples  seq_len={seq_len} ...")

    all_scale_degree  = np.zeros((num_samples, seq_len), dtype=int)
    all_melody_raw    = np.zeros((num_samples, seq_len), dtype=int)
    all_accent        = np.zeros((num_samples, seq_len), dtype=int)
    all_beat_pos      = np.zeros((num_samples, seq_len), dtype=float)
    all_note_dur      = np.zeros((num_samples, seq_len), dtype=int)
    all_note_boundary = np.zeros((num_samples, seq_len), dtype=bool)

    for i in range(num_samples):
        if checkVerbosity(INFO) and i % 200 == 0:
            info(f"  {i}/{num_samples}")
        s = generate_sample(seq_len)
        all_scale_degree[i]  = s['scale_degree']
        all_melody_raw[i]    = s['melody_raw']
        all_accent[i]        = s['accent']
        all_beat_pos[i]      = s['beat_pos']
        all_note_dur[i]      = s['note_dur']
        all_note_boundary[i] = s['note_boundary']

    info(f"Done.")
    return {
        'scale_degree':  all_scale_degree,
        'melody_raw':    all_melody_raw,
        'accent':        all_accent,
        'beat_pos':      all_beat_pos,
        'note_dur':      all_note_dur,
        'note_boundary': all_note_boundary,
    }

# validate_dataset is imported from validate.py

# ── Visualisation ──────────────────────────────────────────────────────────────

# Accent display characters: soft='.', medium='-', strong='|'
_ACCENT_CHAR = ['.', '-', '|']
_BAND        = 32   # timesteps per display row

def visualise_sample(sample):
    """ASCII piano-roll: melody, note boundaries, and accent markers."""
    seq_len      = len(sample['scale_degree'])
    melody_line  = ''.join(SCALE_NAMES[d]  for d in sample['scale_degree'])
    boundary_line= ''.join('|' if b else ' ' for b in sample['note_boundary'])
    accent_line  = ''.join(_ACCENT_CHAR[a] for a in sample['accent'])

    dbg_output(f"\n{'─'*70}")
    dbg_output(f"  Melody (Mayamalavagowla degree)   | = note boundary")
    dbg_output(f"  Accent: . soft   - medium   | strong")
    dbg_output(f"  melody_raw range: [{sample['melody_raw'].min():+3d} .. {sample['melody_raw'].max():+3d}]")
    dbg_output(f"{'─'*70}")

    for start in range(0, seq_len, _BAND):
        end = min(start + _BAND, seq_len)
        raw_slice = sample['melody_raw'][start:end]
        dbg_output(f"t={start:03d}  {melody_line[start:end]}   raw=[{raw_slice.min():+3d}..{raw_slice.max():+3d}]")
        dbg_output(f"bound= {boundary_line[start:end]}")
        dbg_output(f"       {accent_line[start:end]}")

    dbg_output(f"{'─'*70}")

# ── Save / Load ────────────────────────────────────────────────────────────────

def save_dataset(dataset, path='music_dataset.npz'):
    np.savez(path, **dataset)
    dbg_output(f"Saved → {path}  shape={dataset['scale_degree'].shape}")

# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate music training dataset')
    parser.add_argument('--seq-len', type=int, default=SEQ_LEN,
                        help=f'sequence length in eighth-note timesteps '
                             f'(must be a multiple of {BLOCK_LEN}, default {SEQ_LEN})')
    parser.add_argument('--n',       type=int, default=NUM_SAMPLES,
                        help=f'number of samples to generate (default {NUM_SAMPLES})')
    parser.add_argument('--out',     default='music_dataset.npz',
                        help='output path (default music_dataset.npz)')
    args = parser.parse_args()

    if args.seq_len % BLOCK_LEN != 0:
        import sys
        print(f"Error: --seq-len {args.seq_len} is not a multiple of BLOCK_LEN ({BLOCK_LEN})")
        sys.exit(1)

    set_verbosity(INFO)
    set_seed(42)

    dataset = generate_dataset(num_samples=args.n, seq_len=args.seq_len)
    validate_dataset(dataset)
    save_dataset(dataset, args.out)

    # Visualise 3 samples chosen reproducibly
    vis_rng = np.random.default_rng(seed=7)
    indices = vis_rng.choice(args.n, size=min(3, args.n), replace=False)
    for idx in indices:
        dbg_output(f"\n{'═'*70}")
        dbg_output(f"  Sample {idx}")
        visualise_sample({k: v[idx] for k, v in dataset.items()})
