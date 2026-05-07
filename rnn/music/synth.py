# --------------------------------------------------
# Tech Aarvam
# Copyright (c) 2026 Tech Aarvam.
# Authors: Ram (Ramasubramanian B), Claude Code (Anthropic)
# --------------------------------------------------
import argparse
import os
import subprocess
import tempfile
import numpy as np
from scipy.io import wavfile
from debug import *
from seeder import *

# ── Constants ──────────────────────────────────────────────────────────────────

SAMPLE_RATE      = 44100
TEMPO_BPM        = 240                      # quarter note = 0.25 s
BEAT_DURATION    = 60.0 / TEMPO_BPM        # seconds per quarter note
EIGHTH_DURATION  = BEAT_DURATION / 2.0     # seconds per eighth note
MIN_NOTE_SAMPLES = 256                     # warn if a note is shorter than this

TARGET_LUFS      = -14.0                   # average loudness target (dBFS RMS)

OUTPUT_DIR     = 'wav_output'
DATASET_PATH   = 'music_dataset.npz'
DEFAULT_N_SAMPLES = 5                       # synthesise this many unless --all

# Mayamalavagowla scale — one octave from C4
MAYAMALAVAGOWLA_C4 = {
    0: 261.63,   # C4  – S
    1: 277.18,   # Db4 – R1
    2: 329.63,   # E4  – G3
    3: 349.23,   # F4  – M1
    4: 392.00,   # G4  – P
    5: 415.30,   # Ab4 – D1
    6: 493.88,   # B4  – N3
}

# Harmonic series: (harmonic_number, relative_amplitude)
# Full series — trimmed per timbre and accent at synthesis time.
HARMONICS = [
    (1, 1.00),
    (2, 0.50),
    (3, 0.25),
    (4, 0.12),
    (5, 0.06),
]

# Timbre presets: name → how many harmonics from HARMONICS to allow at maximum
TIMBRE_MAX_HARMONICS = {
    'flute':    2,
    'strings':  4,
    'sawtooth': 5,
}

TIMBRE = 'strings'   # active timbre: 'flute' | 'strings' | 'sawtooth'

# Accent → (amplitude_scale, max_harmonics_allowed)
# Accent further limits richness on top of TIMBRE ceiling.
ACCENT_AMPLITUDE  = {0: 0.4, 1: 0.7, 2: 1.0}
ACCENT_MAX_HARM   = {0: 2,   1: 3,   2: 5  }

# Envelope fractions (must sum to 1.0)
ATTACK_FRAC  = 0.10
SUSTAIN_FRAC = 0.70
DECAY_FRAC   = 0.20

CROSSFADE_MS = 25     # overlap between consecutive notes to eliminate boundary clicks

# ── Drone (sequentially strummed tanpura) ─────────────────────────────────────
# Stroke cycle for Mayamalavagowla: Pa(low) Sa Sa Sa(low)
# Each entry: (scale_degree, octave_offset)
DRONE_STROKE_SEQUENCE = [
    (4, -1, 2),   # Pa  – G3  (low Pa, one octave below middle)
    (0,  0, 4),   # Sa  – C4
    (0, -1, 2),   # Sa  – C3  (low Sa, one octave below middle)
]
DRONE_STROKE_EIGHTHS = 2      # eighths between successive plucks
DRONE_CYCLE_EIGHTHS  = 16     # full cycle length (3 strokes × dur_mult: 2+4+2 × 2 eighths each)
DRONE_DECAY_TIME_S   = 0.6    # exponential decay time constant — string rings ~3× this

# Tanpura harmonic profile: brighter and richer than the melody harmonics
DRONE_HARMONICS = [
    (1, 1.00),
    (2, 0.60),
    (3, 0.35),
    (4, 0.20),
    (5, 0.12),
    (6, 0.07),
    (7, 0.04),
]

DRONE_AMPLITUDE = 0.30   # relative weight of drone vs melody before RMS normalisation
DRONE_FADE_MS   = 20     # fade-in/out at edges to avoid clicks (milliseconds)

# ── Dataset loading ────────────────────────────────────────────────────────────

def load_dataset(path=DATASET_PATH):
    data = np.load(path)
    n = data['melody_raw'].shape[0]
    samples = []
    for i in range(n):
        mr = data['melody_raw'][i]
        samples.append({
            'melody_raw':    mr,
            'scale_degree':  mr % 7,
            'accent':        data['accent'][i],
            'note_boundary': data['note_boundary'][i],
        })
    info(f"Loaded {n} samples from {path}")
    return samples

# ── Frequency lookup ───────────────────────────────────────────────────────────

def get_frequency(scale_degree, melody_raw):
    """Return Hz for this scale degree with octave shift derived from melody_raw."""
    octave_shift = int(melody_raw) // 7      # gives -1, 0, or +1 for raw in [-12,12]
    base_freq    = MAYAMALAVAGOWLA_C4[int(scale_degree)]
    return base_freq * (2.0 ** octave_shift)

# ── Envelope ───────────────────────────────────────────────────────────────────

def apply_envelope(wave, attack_frac=ATTACK_FRAC,
                   sustain_frac=SUSTAIN_FRAC, decay_frac=DECAY_FRAC):
    """Multiply wave by an attack-sustain-decay envelope in-place and return it."""
    n      = len(wave)
    env    = np.ones(n, dtype=float)

    n_atk  = max(1, int(n * attack_frac))
    n_sus  = int(n * sustain_frac)
    n_dec  = n - n_atk - n_sus          # absorb rounding into decay

    if n_atk > 0:
        env[:n_atk] = np.linspace(0.0, 1.0, n_atk)

    # sustain stays at 1.0 — already set above

    if n_dec > 0:
        env[n_atk + n_sus:] = np.linspace(1.0, 0.0, n_dec)

    return wave * env

# ── Drone synthesis ────────────────────────────────────────────────────────────

def drone_envelope(n_samples):
    """Exponential decay envelope for a single plucked tanpura stroke."""
    t = np.linspace(0.0, DRONE_DECAY_TIME_S * 3.0, n_samples)
    return np.exp(-t / DRONE_DECAY_TIME_S)


def synthesise_drone(duration_s, eighth_duration=EIGHTH_DURATION):
    """
    Synthesise the tanpura drone as sequentially strummed strokes.

    Each stroke is plucked at its cycle position, synthesised with an exponential
    decay, then added into the output buffer — overlapping decays accumulate to
    create the characteristic tanpura wash.
    """
    total_samples  = int(duration_s * SAMPLE_RATE)
    drone          = np.zeros(total_samples, dtype=float)
    fade_samps     = int(DRONE_FADE_MS * SAMPLE_RATE / 1000)
    stroke_samples = int(DRONE_STROKE_EIGHTHS * eighth_duration * SAMPLE_RATE)
    decay_samples  = int(DRONE_DECAY_TIME_S * 3.0 * SAMPLE_RATE)

    if stroke_samples < MIN_NOTE_SAMPLES:
        warn(f"Drone stroke too short: {stroke_samples} samples < {MIN_NOTE_SAMPLES}")

    stroke_idx = 0
    t_stroke   = 0
    while t_stroke < total_samples:
        cycle_pos  = stroke_idx % len(DRONE_STROKE_SEQUENCE)
        sd, octave, dur_mult = DRONE_STROKE_SEQUENCE[cycle_pos]
        freq       = MAYAMALAVAGOWLA_C4[sd] * (2.0 ** octave)

        if freq < 40.0 or freq > 4000.0:
            warn(f"Drone stroke {stroke_idx}: freq {freq:.1f} Hz out of [40,4000]")

        sw_len = min(decay_samples, total_samples - t_stroke)
        t      = np.arange(sw_len) / SAMPLE_RATE

        stroke_wave = np.zeros(sw_len, dtype=float)
        for (h_num, rel_amp) in DRONE_HARMONICS:
            h_freq = freq * h_num
            if h_freq > SAMPLE_RATE / 2:
                continue
            phase_offset = 2.0 * np.pi * h_freq * t_stroke / SAMPLE_RATE
            stroke_wave += rel_amp * np.sin(2.0 * np.pi * h_freq * t + phase_offset)

        peak = np.max(np.abs(stroke_wave))
        if peak > 0:
            stroke_wave /= peak
        stroke_wave *= drone_envelope(sw_len)

        attack_s = min(fade_samps, sw_len // 4)
        if attack_s > 0:
            stroke_wave[:attack_s] *= np.linspace(0.0, 1.0, attack_s)

        if sw_len < decay_samples:
            fade = min(fade_samps, sw_len // 2)
            if fade > 0:
                stroke_wave[-fade:] *= np.linspace(1.0, 0.0, fade)

        drone[t_stroke : t_stroke + sw_len] += stroke_wave

        if checkVerbosity(DEBUG) and not checkVerbosity(INFO):
            debug(f"  drone stroke {stroke_idx}  cycle={cycle_pos}  "
                  f"freq={freq:.1f}Hz  t={t_stroke}")

        stroke_idx += 1
        t_stroke   += stroke_samples * dur_mult

    # Normalise and scale to amplitude — fade is handled in synthesise_sample
    peak = np.max(np.abs(drone))
    if peak > 0:
        drone /= peak

    drone *= DRONE_AMPLITUDE
    return drone


# ── Note synthesis ─────────────────────────────────────────────────────────────

def synthesise_note(frequency, duration_s, accent, timbre=TIMBRE):
    """
    Frequency-domain note synthesis via IFFT.
    Returns a float64 array in approximately [-1, 1] (envelope applied, amplitude scaled).
    """
    n_samples = int(duration_s * SAMPLE_RATE)

    if n_samples < MIN_NOTE_SAMPLES:
        warn(f"Note at {frequency:.1f} Hz: {n_samples} samples < {MIN_NOTE_SAMPLES} — "
             f"consider lowering TEMPO_BPM")

    # Decide how many harmonics to use: minimum of timbre ceiling and accent ceiling
    h_limit  = min(TIMBRE_MAX_HARMONICS[timbre], ACCENT_MAX_HARM[accent])
    active_h = HARMONICS[:h_limit]

    # Build one-sided FFT spectrum
    fft_len  = n_samples // 2 + 1
    spectrum = np.zeros(fft_len, dtype=complex)

    for (h_num, rel_amp) in active_h:
        bin_idx = int(round(frequency * h_num * n_samples / SAMPLE_RATE))
        if bin_idx == 0 or bin_idx >= fft_len:
            warn(f"Harmonic {h_num} of {frequency:.1f} Hz → bin {bin_idx} "
                 f"out of [1, {fft_len-1}], skipping")
            continue
        # Scale so irfft produces amplitude ≈ rel_amp (2/N factor from irfft normalization)
        spectrum[bin_idx] = rel_amp * (n_samples / 2.0)

    wave = np.fft.irfft(spectrum, n=n_samples)

    # Normalize harmonics to [-1, 1] before envelope so relative amplitudes stay intact
    peak = np.max(np.abs(wave))
    if peak > 0:
        wave /= peak

    wave = apply_envelope(wave)

    # Scale loudness by accent
    wave *= ACCENT_AMPLITUDE[accent]

    return wave

# ── Note merging ───────────────────────────────────────────────────────────────

def merge_consecutive(sample):
    """
    Merge consecutive timesteps that share (scale_degree, octave_shift, accent).
    Accent boundaries are kept as note boundaries to preserve rhythmic articulation.
    Returns list of (scale_degree, melody_raw_first, accent, duration_eighths).
    """
    sd = sample['scale_degree']
    mr = sample['melody_raw']
    ac = sample['accent']
    T  = len(sd)

    notes = []
    i     = 0
    while i < T:
        cur_sd = int(sd[i])
        cur_os = int(mr[i]) // 7
        cur_ac = int(ac[i])
        j      = i + 1
        while j < T:
            if (int(sd[j]) == cur_sd and
                    int(mr[j]) // 7 == cur_os and
                    int(ac[j]) == cur_ac):
                j += 1
            else:
                break
        notes.append((cur_sd, int(mr[i]), cur_ac, j - i))
        i = j

    return notes

# ── Crossfade concat ───────────────────────────────────────────────────────────

def find_zero_crossing(seg, max_search):
    """Return index of first zero crossing within max_search samples, or 0."""
    for i in range(1, min(max_search, len(seg))):
        if seg[i-1] * seg[i] <= 0:
            return i
    return 0


def crossfade_concat(segments, cf_samples):
    """Blend consecutive note segments with a linear crossfade overlap."""
    if len(segments) == 0:
        return np.array([])
    if cf_samples == 0:
        return np.concatenate(segments)

    out = segments[0].copy()
    for seg in segments[1:]:
        if len(out) < cf_samples or len(seg) < cf_samples:
            out = np.concatenate([out, seg])
            continue
        zc       = find_zero_crossing(seg, cf_samples)
        fade_out = np.linspace(1.0, 0.0, cf_samples)
        fade_in  = np.linspace(0.0, 1.0, cf_samples)
        overlap  = out[-cf_samples:] * fade_out + seg[zc:zc+cf_samples] * fade_in
        out      = np.concatenate([out[:-cf_samples], overlap, seg[zc+cf_samples:]])
    return out


# ── Full sample synthesis ──────────────────────────────────────────────────────

def synthesise_sample(sample, timbre=TIMBRE, eighth_duration=EIGHTH_DURATION):
    """Synthesise one dataset sample: melody + Sa-Pa-Sa drone, returned as float64 array."""
    notes    = merge_consecutive(sample)
    segments = []

    for (sd, mr, ac, dur_eighths) in notes:
        duration_s = dur_eighths * eighth_duration
        frequency  = get_frequency(sd, mr)
        seg = synthesise_note(frequency, duration_s, ac, timbre)
        segments.append(seg)

    # Validate segment boundaries — both ends should be silent after decay-to-zero
    for seg in segments:
        if abs(seg[0]) > 0.01 or abs(seg[-1]) > 0.01:
            warn(f"Segment boundary not near zero: start={seg[0]:.4f} end={seg[-1]:.4f}")

    cf_samples   = int(CROSSFADE_MS * SAMPLE_RATE / 1000)
    melody       = crossfade_concat(segments, cf_samples)

    lead_samples = int(DRONE_CYCLE_EIGHTHS * eighth_duration * SAMPLE_RATE)  # 1 full tanpura cycle

    # 2 cycles lead-in (cycle 1 fades up, cycle 2 at full), 1 cycle lead-out (fades down)
    drone_duration_s = (len(melody) + 3 * lead_samples) / SAMPLE_RATE
    drone            = synthesise_drone(drone_duration_s, eighth_duration)

    # Pad melody: 2 cycles of silence before, 1 cycle after
    melody_padded = np.concatenate([
        np.zeros(2 * lead_samples),
        melody,
        np.zeros(lead_samples),
    ])

    # Fade up over first cycle only; second cycle and beyond stay at 1.0
    fade_in_env  = np.ones(len(drone))
    fade_out_env = np.ones(len(drone))
    fade_in_env[:lead_samples]   = np.linspace(0.0, 1.0, lead_samples)
    fade_out_env[-lead_samples:] = np.linspace(1.0, 0.0, lead_samples)
    drone = drone * fade_in_env * fade_out_env

    min_len = min(len(melody_padded), len(drone))
    mixed   = melody_padded[:min_len] + drone[:min_len]
    peak    = np.max(np.abs(mixed))
    if peak > 0:
        mixed /= peak
    return mixed

# ── Save wav ───────────────────────────────────────────────────────────────────

def save_wav(array, path, target_lufs=TARGET_LUFS):
    """RMS-normalise to target_lufs (dBFS), clip peaks, write 16-bit PCM wav."""
    rms = np.sqrt(np.mean(array ** 2))
    if rms == 0:
        warn(f"save_wav: silent signal for {path}")
        pcm = np.zeros(len(array), dtype=np.int16)
        wavfile.write(path, SAMPLE_RATE, pcm)
        return

    target_rms = 10.0 ** (target_lufs / 20.0)   # linear RMS target (~0.1995 for -14 dB)
    array = array * (target_rms / rms)

    # Hard-clip any transient peaks that exceed ±1 after RMS scaling
    peak = np.max(np.abs(array))
    if peak > 1.0:
        array = np.clip(array, -1.0, 1.0)
        info(f"  Peak clipped from {peak:.3f} to 1.0 for {path}")

    actual_db = 20.0 * np.log10(np.sqrt(np.mean(array ** 2)))
    pcm = (array * 32767).astype(np.int16)
    wavfile.write(path, SAMPLE_RATE, pcm)
    dbg_output(f"  Wrote {path}  ({len(array)/SAMPLE_RATE:.2f}s  RMS={actual_db:.1f} dBFS)")

# ── Stitch to MP4 ─────────────────────────────────────────────────────────────

def wav_to_mp3(wav_path, mp3_path, bitrate='128k'):
    """Convert a single wav to MP3 and remove the source wav."""
    cmd = ['ffmpeg', '-y', '-i', wav_path, '-c:a', 'libmp3lame', '-b:a', bitrate, mp3_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        warn(f"ffmpeg failed for {wav_path} (exit {result.returncode}):")
        warn(result.stderr[-400:])
    else:
        os.remove(wav_path)
        size_kb = os.path.getsize(mp3_path) / 1024
        dbg_output(f"  → {mp3_path}  ({size_kb:.0f} KB)")


def stitch_to_mp4(wav_paths, out_path):
    """
    Concatenate wav_paths into a single MP4 (AAC audio, no video) using ffmpeg.
    Writes a temporary concat list, calls ffmpeg, then removes the list.
    """
    if not wav_paths:
        warn("stitch_to_mp4: no wav files to stitch")
        return

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                     delete=False) as flist:
        for p in wav_paths:
            # ffmpeg concat demuxer requires absolute paths or paths relative to
            # the list file; use absolute to be safe.
            flist.write(f"file '{os.path.abspath(p)}'\n")
        list_path = flist.name

    cmd = [
        'ffmpeg', '-y',
        '-f', 'concat', '-safe', '0',
        '-i', list_path,
        '-c:a', 'aac', '-b:a', '192k',
        '-vn',
        out_path,
    ]

    dbg_output(f"Stitching {len(wav_paths)} files → {out_path} ...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            warn(f"ffmpeg failed (exit {result.returncode}):")
            warn(result.stderr[-400:])   # last 400 chars of stderr is usually the error
        else:
            size_mb = os.path.getsize(out_path) / 1_048_576
            dbg_output(f"  Done → {out_path}  ({size_mb:.1f} MB)")
    finally:
        os.unlink(list_path)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Synthesise wav files from music dataset')
    parser.add_argument('--all',      action='store_true', help='synthesise all samples')
    parser.add_argument('--dataset',  default=DATASET_PATH, help='path to .npz dataset')
    parser.add_argument('--outdir',   default=OUTPUT_DIR,   help='output directory')
    parser.add_argument('--timbre',   default=TIMBRE,
                        choices=list(TIMBRE_MAX_HARMONICS.keys()), help='timbre preset')
    parser.add_argument('--n',        type=int,   default=DEFAULT_N_SAMPLES,
                        help=f'number of samples (default {DEFAULT_N_SAMPLES}; ignored with --all)')
    parser.add_argument('--beat',     type=float, default=BEAT_DURATION,
                        help=f'quarter-note duration in seconds (default {BEAT_DURATION})')
    args = parser.parse_args()

    set_verbosity(INFO)
    set_seed(42)
    os.makedirs(args.outdir, exist_ok=True)

    eighth_duration = args.beat / 2.0
    bpm             = 60.0 / args.beat

    samples  = load_dataset(args.dataset)
    n_synth  = len(samples) if args.all else min(args.n, len(samples))
    timbre   = args.timbre

    dbg_output(f"Synthesising {n_synth} sample(s)  timbre={timbre}  "
           f"beat={args.beat}s ({bpm:.0f} BPM)  outdir={args.outdir}")

    mp3_paths = []
    for i in range(n_synth):
        info(f"Sample {i}/{n_synth} ...")
        wav      = synthesise_sample(samples[i], timbre=timbre, eighth_duration=eighth_duration)
        wav_path = os.path.join(args.outdir, f"sample_{i:04d}.wav")
        mp3_path = os.path.join(args.outdir, f"sample_{i:04d}.mp3")
        save_wav(wav, wav_path)
        wav_to_mp3(wav_path, mp3_path)
        mp3_paths.append(mp3_path)

    dbg_output(f"Done. {n_synth} file(s) written to {args.outdir}/")

    if args.all:
        stitch_to_mp4(mp3_paths, os.path.join(args.outdir, 'all_samples.mp4'))


if __name__ == "__main__":
    main()
