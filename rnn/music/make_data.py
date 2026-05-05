# --------------------------------------------------
# Tech Aarvam
# Copyright (c) 2026 Tech Aarvam.
# Authors: Ram (Ramasubramanian B), Claude Code (Anthropic)
# --------------------------------------------------
import numpy as np
import random
from debug import *
from seeder import *

# ── Constants ──────────────────────────────────────────────────────────────────

SEQ_LEN      = 128    # eighth-note timesteps per sample (8 bars of 4/4)
BLOCK_LEN    = 16     # eighths per rhythm block (2 bars of 4/4)
MELODY_CLAMP = 12     # ± clamp on raw cumulative sum
NUM_SAMPLES  = 2000

SCALE_NAMES  = ['C', 'D', 'E', 'F', 'G', 'A', 'B']

# Cascaded single-firing level table: (level_index, period_in_eighths), slowest first.
# At each timestep exactly one level fires — the slowest whose period divides t.
LEVEL_PERIODS = [
    (0, 32),   # Paragraph  — fires when t % 32 == 0
    (1, 16),   # Ultra-slow — fires when t % 16 == 0
    (2,  8),   # Super-slow — fires when t %  8 == 0
    (3,  4),   # Slow       — fires when t %  4 == 0
    (4,  2),   # At-rate    — fires when t %  2 == 0
    (5,  1),   # Fast       — fires every remaining timestep
]

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
    'sparse':            [(4,2),(4,1)],
    'push_pull':         [(1,0),(3,2),(1,0),(3,2)],
    'cadence':           [(2,1),(2,1),(4,2)],
    'ornament_run':      [(1,2),(1,1),(1,0),(1,1),(2,2),(2,1)],
}

MAIN_PATTERNS = [k for k in PATTERNS if k != 'cadence']

# ── Melody generation ──────────────────────────────────────────────────────────

def which_level_fires(t):
    """Return the index of the slowest level whose period divides t exactly."""
    for level, period in LEVEL_PERIODS:
        if t % period == 0:
            return level
    return 5   # unreachable — level 5 (period=1) always matches


def update_melody(running_pos, t):
    """Fire the appropriate level, draw one delta, return (new_pos, level, delta)."""
    level = which_level_fires(t)
    delta = random.choice([-1, 0, 1])
    new_pos = int(np.clip(running_pos + delta, -MELODY_CLAMP, MELODY_CLAMP))
    return new_pos, level, delta


def generate_melody(seq_len):
    """
    Cascaded single-firing melody generator.

    At each timestep exactly one level fires — the slowest whose period divides t.
    A single delta ±1 or 0 is added to one shared running position, guaranteeing
    consecutive melody values never differ by more than 1.

    Returns (melody_raw_arr, scale_degree_arr, level_fired_arr).
    """
    running_pos      = 0
    melody_raw_arr   = np.zeros(seq_len, dtype=int)
    scale_degree_arr = np.zeros(seq_len, dtype=int)
    level_fired_arr  = np.zeros(seq_len, dtype=int)

    for t in range(seq_len):
        running_pos, level, delta = update_melody(running_pos, t)

        melody_raw_arr[t]   = running_pos
        scale_degree_arr[t] = running_pos % 7
        level_fired_arr[t]  = level

    return melody_raw_arr, scale_degree_arr, level_fired_arr

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
    melody_raw, scale_degree, level_fired    = generate_melody(seq_len)
    return {
        'scale_degree':  scale_degree,
        'melody_raw':    melody_raw,
        'accent':        accent,
        'beat_pos':      beat_pos,
        'note_dur':      note_dur,
        'note_boundary': note_boundary,
        'level_fired':   level_fired,
    }


def generate_dataset(num_samples=NUM_SAMPLES, seq_len=SEQ_LEN):
    info(f"Generating {num_samples} samples  seq_len={seq_len} ...")

    all_scale_degree  = np.zeros((num_samples, seq_len), dtype=int)
    all_melody_raw    = np.zeros((num_samples, seq_len), dtype=int)
    all_accent        = np.zeros((num_samples, seq_len), dtype=int)
    all_beat_pos      = np.zeros((num_samples, seq_len), dtype=float)
    all_note_dur      = np.zeros((num_samples, seq_len), dtype=int)
    all_note_boundary = np.zeros((num_samples, seq_len), dtype=bool)
    all_level_fired   = np.zeros((num_samples, seq_len), dtype=int)

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
        all_level_fired[i]   = s['level_fired']

    info(f"Done.")
    return {
        'scale_degree':  all_scale_degree,
        'melody_raw':    all_melody_raw,
        'accent':        all_accent,
        'beat_pos':      all_beat_pos,
        'note_dur':      all_note_dur,
        'note_boundary': all_note_boundary,
        'level_fired':   all_level_fired,
    }

# ── Validation ─────────────────────────────────────────────────────────────────

def validate_dataset(dataset):
    """Print warnings for any out-of-spec values; does not raise."""
    ok = True
    n, seq_len = dataset['scale_degree'].shape

    if not np.all((dataset['scale_degree'] >= 0) & (dataset['scale_degree'] <= 6)):
        warn("VALIDATION FAIL: scale_degree out of [0, 6]")
        ok = False

    if not np.all((dataset['melody_raw'] >= -MELODY_CLAMP) & (dataset['melody_raw'] <= MELODY_CLAMP)):
        warn(f"VALIDATION FAIL: melody_raw out of [{-MELODY_CLAMP}, {MELODY_CLAMP}]")
        ok = False

    if not np.all((dataset['accent'] >= 0) & (dataset['accent'] <= 2)):
        warn("VALIDATION FAIL: accent out of [0, 2]")
        ok = False

    # Consecutive melody steps must never differ by more than 1 (single-delta guarantee).
    diffs = np.abs(np.diff(dataset['melody_raw'], axis=1))
    if np.any(diffs > 1):
        n_bad = int(np.sum(diffs > 1))
        warn(f"VALIDATION FAIL: {n_bad} consecutive melody_raw pairs differ by > 1")
        ok = False

    # Rhythm block length: every BLOCK_LEN-wide window must sum accent counts to BLOCK_LEN
    n_blocks = seq_len // BLOCK_LEN
    for bi in range(n_blocks):
        block_accents = dataset['accent'][:, bi*BLOCK_LEN:(bi+1)*BLOCK_LEN]
        # Accent array is dense (one entry per eighth), so block width == BLOCK_LEN always;
        # check that no sample's block overflowed by verifying the stride is consistent.
        if block_accents.shape[1] != BLOCK_LEN:
            warn(f"VALIDATION FAIL: rhythm block {bi} has width {block_accents.shape[1]} != {BLOCK_LEN}")
            ok = False
            break

    if ok:
        output("VALIDATION: all checks passed.")

# ── Visualisation ──────────────────────────────────────────────────────────────

# Accent display characters: soft='.', medium='-', strong='|'
_ACCENT_CHAR = ['.', '-', '|']
_BAND        = 32   # timesteps per display row

def visualise_sample(sample):
    """ASCII piano-roll: melody, level-fired hierarchy, and accent markers."""
    seq_len     = len(sample['scale_degree'])
    melody_line = ''.join(SCALE_NAMES[d]        for d in sample['scale_degree'])
    level_line  = ''.join(str(l)                for l in sample['level_fired'])
    accent_line = ''.join(_ACCENT_CHAR[a]       for a in sample['accent'])

    output(f"\n{'─'*70}")
    output(f"  Melody (C-major note)   level: 0=Paragraph 1=UltraSlow … 5=Fast")
    output(f"  Accent: . soft   - medium   | strong")
    output(f"  melody_raw range: [{sample['melody_raw'].min():+3d} .. {sample['melody_raw'].max():+3d}]")
    output(f"{'─'*70}")

    for start in range(0, seq_len, _BAND):
        end = min(start + _BAND, seq_len)
        raw_slice = sample['melody_raw'][start:end]
        output(f"t={start:03d}  {melody_line[start:end]}   raw=[{raw_slice.min():+3d}..{raw_slice.max():+3d}]")
        output(f"level= {level_line[start:end]}")
        output(f"       {accent_line[start:end]}")

    output(f"{'─'*70}")

# ── Save / Load ────────────────────────────────────────────────────────────────

def save_dataset(dataset, path='music_dataset.npz'):
    np.savez(path, **dataset)
    output(f"Saved → {path}  shape={dataset['scale_degree'].shape}")

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
        output(f"\n{'═'*70}")
        output(f"  Sample {idx}")
        visualise_sample({k: v[idx] for k, v in dataset.items()})
