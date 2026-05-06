# --------------------------------------------------
# Tech Aarvam
# Copyright (c) 2026 Tech Aarvam.
# Author: Ram (Ramasubramanian B)
# --------------------------------------------------
import numpy as np
from debug import *

MELODY_CLAMP = 12

def validate_sample(melody_raw, accent, note_boundary):
    """
    Validate one sample against all data rules.

    Rules:
      1. melody_raw in [-12, +12]
      2. accent in {0, 1, 2}
      3. note_boundary[0] must be True
      4. melody_raw changes only at note_boundary
      5. accent changes only at note_boundary
      6. consecutive melody_raw steps differ by at most 1
      7. melody_raw is constant within each note (held value identical across all steps)
      8. accent is constant within each note

    Returns True if all rules pass, False otherwise.
    Prints a warning for each violation found.
    """
    ok = True
    melody_raw   = np.asarray(melody_raw,   dtype=int)
    accent       = np.asarray(accent,       dtype=int)
    note_boundary= np.asarray(note_boundary, dtype=bool)
    seq_len      = len(melody_raw)

    # Rule 1: melody_raw range
    if not np.all((melody_raw >= -MELODY_CLAMP) & (melody_raw <= MELODY_CLAMP)):
        warn(f"FAIL rule 1: melody_raw out of [{-MELODY_CLAMP}, {MELODY_CLAMP}]")
        ok = False

    # Rule 2: accent values
    if not np.all((accent >= 0) & (accent <= 2)):
        warn(f"FAIL rule 2: accent has values outside {{0,1,2}}")
        ok = False

    # Rule 3: sequence starts with a boundary
    if not note_boundary[0]:
        warn("FAIL rule 3: note_boundary[0] is not True")
        ok = False

    # Rules 4 & 5: changes only at note_boundary
    mid_note         = ~note_boundary[1:]          # timesteps that are NOT a boundary
    melody_changes   = np.diff(melody_raw) != 0
    accent_changes   = np.diff(accent)    != 0

    bad_melody = np.where(mid_note & melody_changes)[0] + 1
    if len(bad_melody):
        warn(f"FAIL rule 4: melody_raw changed mid-note at {len(bad_melody)} position(s): {bad_melody[:5]}")
        ok = False

    bad_accent = np.where(mid_note & accent_changes)[0] + 1
    if len(bad_accent):
        warn(f"FAIL rule 5: accent changed mid-note at {len(bad_accent)} position(s): {bad_accent[:5]}")
        ok = False

    # Rule 6: consecutive diff <= 1
    diffs = np.abs(np.diff(melody_raw))
    if np.any(diffs > 1):
        n_bad = int(np.sum(diffs > 1))
        warn(f"FAIL rule 6: {n_bad} consecutive melody_raw pair(s) differ by > 1")
        ok = False

    # Rules 7 & 8: held value is identical within each note (not just non-incrementing)
    # Build note segments by splitting at boundaries and check uniformity within each.
    boundary_positions = np.where(note_boundary)[0].tolist() + [seq_len]
    for i in range(len(boundary_positions) - 1):
        start, end = boundary_positions[i], boundary_positions[i + 1]
        if np.any(melody_raw[start:end] != melody_raw[start]):
            warn(f"FAIL rule 7: melody_raw not constant in note [{start}:{end}]")
            ok = False
            break
        if np.any(accent[start:end] != accent[start]):
            warn(f"FAIL rule 8: accent not constant in note [{start}:{end}]")
            ok = False
            break

    # Rule 9: melody_raw must change at least once
    if np.all(melody_raw == melody_raw[0]):
        warn("FAIL rule 9: melody_raw never changes across the sequence")
        ok = False

    return ok


def validate_dataset(dataset):
    """
    Validate all samples in a dataset dict with arrays of shape (n_samples, seq_len).
    Returns True if all samples pass, False otherwise.
    """
    n = dataset['melody_raw'].shape[0]
    n_fail = 0
    for i in range(n):
        if not validate_sample(
            dataset['melody_raw'][i],
            dataset['accent'][i],
            dataset['note_boundary'][i],
        ):
            warn(f"  Sample {i} failed validation")
            n_fail += 1

    if n_fail == 0:
        dbg_output(f"VALIDATION: all {n} samples passed.")
    else:
        warn(f"VALIDATION: {n_fail}/{n} samples failed.")
    return n_fail == 0
