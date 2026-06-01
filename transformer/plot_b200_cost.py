"""
plot_b200_cost.py — B200-only training: progressive expansion vs direct for 100M and 1B.

Question: if you committed to B200 from day 1 (no GPU ladder), does progressive
expansion still save cost vs training at the full model size from the start?

Answer: yes — because early stages have fewer parameters → fewer FLOPs per token
→ more tokens processed per dollar at the same hourly rate.

Two charts:
  b200_100m_cost.png  — 100M model  (d256→d384→d512→d640→d768→d896 all on B200)
  b200_1b_cost.png    — 1B model    (100M→200M→600M→1B all on B200)

B200 spec: 179 GB VRAM, 2250 TFLOPS fp16, $4.24/hr (vast.ai May 2026).
VRAM check (conservative 120 bytes/param):
  100M (103M): 103M × 120 = 12.4 GB  ✓ fits easily
  1B:          1B   × 120 = 120 GB   ✓ fits (179 GB > 120 GB)
75% Chinchilla: D = 15N tokens.
Recovery: 2M tokens per expansion = 10% of one 20M-token epoch.
MFU: two scenarios — 0.35 (optimistic) and 0.15 (realistic).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import os

HERE   = os.path.dirname(os.path.abspath(__file__))
OUT    = os.path.join(HERE, 'output')
os.makedirs(OUT, exist_ok=True)
DARK_BG = '#1a1a2e'

CHIN  = 15      # 75% Chinchilla (D = 20N × 0.75)
FPT   = 6       # FLOPs per token (2N forward + 4N backward)
RECOV = 2e6     # 10% of 20M-token epoch per expansion

B200_VRAM   = 179    # GB
B200_TFLOPS = 2250   # fp16 dense
B200_CPH    = 4.24   # $/hr (vast.ai May 2026, listing #33945597)
B200_COLOR  = '#991100'

MFU_SCENARIOS = [
    (0.35, 'MFU=0.35  (optimistic — compute-bound, large batch)'),
    (0.15, 'MFU=0.15  (realistic — small batch, memory-constrained)'),
]

def b200_compute(n, tokens, mfu):
    tps   = B200_TFLOPS * 1e12 * mfu / (FPT * n)
    hours = tokens / tps / 3600
    return hours * B200_CPH, hours

# ── 100M architecture ─────────────────────────────────────────────────────────
N_EMBED = 25.8e6
def n_total_100m(d):
    core = 96 * d**2 + (2 * d * 256 if d > 256 else 0)
    return core + N_EMBED

DIMS_100M = [256, 384, 512, 640, 768, 896]
N_100M    = {d: n_total_100m(d) for d in DIMS_100M}
N_TARG_100M = N_100M[896]   # ~103M

def make_stages_100m():
    stages    = []
    chin_done = 0.0
    for i, d in enumerate(DIMS_100M):
        n           = N_100M[d]
        chin_target = CHIN * n
        new_toks    = chin_target - chin_done
        recovery    = RECOV if i > 0 else 0.0
        stages.append((f'd{d}', n, new_toks + recovery))
        chin_done = chin_target
    return stages

STAGES_100M = make_stages_100m()

# ── 1B architecture ───────────────────────────────────────────────────────────
CHECKPOINTS_1B = [100e6, 200e6, 600e6, 1e9]   # same ladder as 1B chart

def make_stages_1b():
    stages    = []
    chin_done = 0.0
    for i, n in enumerate(CHECKPOINTS_1B):
        chin_target = CHIN * n
        new_toks    = chin_target - chin_done
        recovery    = RECOV if i > 0 else 0.0
        stages.append((f'{int(n/1e6)}M', n, new_toks + recovery))
        chin_done = chin_target
    return stages

STAGES_1B = make_stages_1b()

# ── Summary printout ──────────────────────────────────────────────────────────
print('── B200 100M ──────────────────────────────────────────')
print(f'  Direct 103M:  {CHIN * N_TARG_100M / 1e9:.2f}B tokens')
for mfu_val, lbl in [(0.35, 'MFU=0.35'), (0.15, 'MFU=0.15')]:
    prog_costs  = [b200_compute(n, t, mfu_val) for _, n, t in STAGES_100M]
    prog_c      = sum(c for c, _ in prog_costs)
    prog_h      = sum(h for _, h in prog_costs)
    prog_toks   = sum(t for _, _, t in STAGES_100M)
    dc, dh      = b200_compute(N_TARG_100M, CHIN * N_TARG_100M, mfu_val)
    print(f'  {lbl}:  Progressive {prog_h:.1f}h ${prog_c:.2f}  '
          f'({prog_toks/1e9:.2f}B tok)    Direct {dh:.1f}h ${dc:.2f}  '
          f'(saves ${dc-prog_c:.2f}, {(dc-prog_c)/dc*100:.0f}%)')

print()
print('── B200 1B ─────────────────────────────────────────────')
print(f'  Direct 1B:  {CHIN * 1e9 / 1e9:.0f}B tokens')
for mfu_val, lbl in [(0.35, 'MFU=0.35'), (0.15, 'MFU=0.15')]:
    prog_costs  = [b200_compute(n, t, mfu_val) for _, n, t in STAGES_1B]
    prog_c      = sum(c for c, _ in prog_costs)
    prog_h      = sum(h for _, h in prog_costs)
    prog_toks   = sum(t for _, _, t in STAGES_1B)
    dc, dh      = b200_compute(1e9, CHIN * 1e9, mfu_val)
    print(f'  {lbl}:  Progressive {prog_h:.0f}h ${prog_c:.0f}  '
          f'({prog_toks/1e9:.2f}B tok)    Direct {dh:.0f}h ${dc:.0f}  '
          f'(saves ${dc-prog_c:.0f}, {(dc-prog_c)/dc*100:.0f}%)')
print()

# ── Shared plot helpers ───────────────────────────────────────────────────────
def style_ax(ax):
    ax.set_facecolor(DARK_BG)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(colors='#888', labelsize=8)
    ax.grid(axis='y', alpha=0.08, color='#555')

def draw_comparison(ax, stages, mfu_val, n_direct, n_direct_label,
                    ylabel, fmt, value_key, shades, title):
    """Draw stacked progressive bar + direct bar side by side."""
    # Progressive stacked
    bottom   = 0.0
    prog_tot = 0.0
    results  = []
    for (lbl, n, toks), shade in zip(stages, shades):
        c, h = b200_compute(n, toks, mfu_val)
        val  = c if value_key == 'cost' else h
        results.append(val)
        prog_tot += val

    for (lbl, n, toks), shade, val in zip(stages, shades, results):
        ax.bar('Progressive', val, bottom=bottom,
               color=shade, alpha=0.90, width=0.55, label=lbl)
        if val > prog_tot * 0.05:
            ax.text(0, bottom + val / 2, fmt.format(val),
                    ha='center', va='center', color='white', fontsize=8)
        bottom += val
    ax.text(0, prog_tot * 1.05, fmt.format(prog_tot),
            ha='center', color='white', fontsize=10, fontweight='bold')

    # Direct bar
    dc, dh = b200_compute(n_direct, CHIN * n_direct, mfu_val)
    dval   = dc if value_key == 'cost' else dh
    ax.bar(n_direct_label, dval, color=B200_COLOR, alpha=0.85, width=0.55)
    ax.text(n_direct_label, dval * 1.05, fmt.format(dval),
            ha='center', color='#ccc', fontsize=10, fontweight='bold')

    # Savings annotation
    savings = dval - prog_tot
    pct     = savings / dval * 100
    if savings > 0:
        ax.text(0.98, 0.96,
                f'saves {fmt.format(savings)} ({pct:.0f}%)',
                transform=ax.transAxes, ha='right', va='top',
                color='#88ff88', fontsize=8.5,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a2a1a', alpha=0.7))

    ax.set_ylabel(ylabel, color='#aaa', fontsize=8.5)
    ax.set_title(title, color='white', fontsize=8.5)
    ax.tick_params(axis='x', rotation=0, colors='#999', labelsize=9)
    style_ax(ax)

def draw_flop_breakdown(ax, stages, mfu_val, n_direct, n_direct_label,
                        shades, title):
    """Stacked bar of FLOPs (PetaFLOPs) per stage vs direct."""
    # Progressive stacked
    bottom   = 0.0
    prog_tot = 0.0
    for (lbl, n, toks), shade in zip(stages, shades):
        pf = FPT * n * toks / 1e15
        prog_tot += pf

    for (lbl, n, toks), shade in zip(stages, shades):
        pf = FPT * n * toks / 1e15
        ax.bar('Progressive', pf, bottom=bottom,
               color=shade, alpha=0.90, width=0.55, label=lbl)
        if pf > prog_tot * 0.05:
            ax.text(0, bottom + pf / 2, f'{pf:.0f}',
                    ha='center', va='center', color='white', fontsize=8)
        bottom += pf
    ax.text(0, prog_tot * 1.05, f'{prog_tot:.0f} PF',
            ha='center', color='white', fontsize=10, fontweight='bold')

    # Direct bar
    dpf = FPT * n_direct * CHIN * n_direct / 1e15
    ax.bar(n_direct_label, dpf, color=B200_COLOR, alpha=0.85, width=0.55)
    ax.text(n_direct_label, dpf * 1.05, f'{dpf:.0f} PF',
            ha='center', color='#ccc', fontsize=10, fontweight='bold')

    savings = dpf - prog_tot
    pct     = savings / dpf * 100
    if savings > 0:
        ax.text(0.98, 0.96,
                f'saves {savings:.0f} PF ({pct:.0f}%)',
                transform=ax.transAxes, ha='right', va='top',
                color='#88ff88', fontsize=8.5,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a2a1a', alpha=0.7))

    ax.set_ylabel('Compute  (PetaFLOPs)', color='#aaa', fontsize=8.5)
    ax.set_title(title, color='white', fontsize=8.5)
    ax.tick_params(axis='x', rotation=0, colors='#999', labelsize=9)
    style_ax(ax)

# ─────────────────────────────────────────────────────────────────────────────
# CHART 1: B200 100M model
# ─────────────────────────────────────────────────────────────────────────────
SHADES_100M = ['#2255cc', '#3366dd', '#4477ee', '#5588ff', '#66aaff', '#77bbff']

chin_toks_100m = CHIN * N_TARG_100M
flop_prog_100m = sum(FPT * n * t for _, n, t in STAGES_100M)
flop_dir_100m  = FPT * N_TARG_100M * chin_toks_100m

fig1 = plt.figure(figsize=(14, 12), facecolor=DARK_BG)
gs1  = GridSpec(2, 3, figure=fig1, hspace=0.48, wspace=0.38,
                top=0.90, bottom=0.07, left=0.07, right=0.97)

for row, (mfu_val, mfu_label) in enumerate(MFU_SCENARIOS):
    ax_cost = fig1.add_subplot(gs1[row, 0])
    ax_time = fig1.add_subplot(gs1[row, 1])
    ax_flop = fig1.add_subplot(gs1[row, 2])

    row_note = (f'{mfu_label}\n'
                f'{chin_toks_100m/1e9:.2f}B tokens  ·  '
                f'Recovery {int(RECOV/1e6)}M tok/expansion')

    draw_comparison(ax_cost, STAGES_100M, mfu_val,
                    N_TARG_100M, 'Direct\n(full d896)',
                    'Total cost  ($)', '${:.2f}', 'cost',
                    SHADES_100M,
                    f'Cost — B200 100M\n{row_note}')

    draw_comparison(ax_time, STAGES_100M, mfu_val,
                    N_TARG_100M, 'Direct\n(full d896)',
                    'Wall-clock time  (hours)', '{:.1f}h', 'time',
                    SHADES_100M,
                    f'Time — B200 100M\n{row_note}')

    draw_flop_breakdown(ax_flop, STAGES_100M, mfu_val,
                        N_TARG_100M, 'Direct\n(full d896)',
                        SHADES_100M,
                        f'FLOPs — independent of MFU\n(same both rows)')

    if row == 0:
        patches = [mpatches.Patch(color=c, label=s[0])
                   for c, s in zip(SHADES_100M, STAGES_100M)]
        ax_cost.legend(handles=patches, fontsize=7, framealpha=0.25,
                       facecolor='#222240', labelcolor='white',
                       title='Expansion stages', title_fontsize=7,
                       loc='upper left')

fig1.suptitle(
    'B200-only Training  —  100M Model  (d256→d384→d512→d640→d768→d896)\n'
    f'B200: {B200_TFLOPS} TFLOPS  ·  {B200_VRAM} GB VRAM  ·  ${B200_CPH}/hr  '
    f'(vast.ai May 2026)  ·  75% Chinchilla = {chin_toks_100m/1e9:.2f}B tokens  ·  '
    f'Progressive compute: {flop_prog_100m/1e15:.0f} PF  '
    f'vs Direct: {flop_dir_100m/1e15:.0f} PF  '
    f'(saves {(flop_dir_100m-flop_prog_100m)/flop_dir_100m*100:.0f}% FLOPs)',
    color='white', fontsize=9.5, y=0.975)

fig1.text(0.5, 0.01,
          'FLOPs saved because early stages process tokens at smaller model size  ·  '
          'Cost & time savings are proportional to FLOP savings at fixed $/hr  ·  '
          'FLOPs panel is MFU-independent (same for both rows)',
          ha='center', color='#555577', fontsize=8)

out1 = os.path.join(OUT, 'b200_100m_cost.png')
plt.savefig(out1, dpi=140, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print(f'Saved → {out1}')

# ─────────────────────────────────────────────────────────────────────────────
# CHART 2: B200 1B model
# ─────────────────────────────────────────────────────────────────────────────
SHADES_1B = ['#2255cc', '#3377ee', '#5599ff', '#77bbff']

chin_toks_1b = CHIN * 1e9
flop_prog_1b = sum(FPT * n * t for _, n, t in STAGES_1B)
flop_dir_1b  = FPT * 1e9 * chin_toks_1b

print(f'\n1B FLOPs:  Progressive {flop_prog_1b/1e15:.0f} PF  '
      f'vs Direct {flop_dir_1b/1e15:.0f} PF  '
      f'(saves {(flop_dir_1b-flop_prog_1b)/flop_dir_1b*100:.0f}%)')

fig2 = plt.figure(figsize=(14, 12), facecolor=DARK_BG)
gs2  = GridSpec(2, 3, figure=fig2, hspace=0.48, wspace=0.38,
                top=0.90, bottom=0.07, left=0.07, right=0.97)

for row, (mfu_val, mfu_label) in enumerate(MFU_SCENARIOS):
    ax_cost = fig2.add_subplot(gs2[row, 0])
    ax_time = fig2.add_subplot(gs2[row, 1])
    ax_flop = fig2.add_subplot(gs2[row, 2])

    row_note = (f'{mfu_label}\n'
                f'{chin_toks_1b/1e9:.0f}B tokens  ·  '
                f'Recovery {int(RECOV/1e6)}M tok/expansion')

    draw_comparison(ax_cost, STAGES_1B, mfu_val,
                    1e9, 'Direct\n(full 1B)',
                    'Total cost  ($)', '${:.0f}', 'cost',
                    SHADES_1B,
                    f'Cost — B200 1B\n{row_note}')

    draw_comparison(ax_time, STAGES_1B, mfu_val,
                    1e9, 'Direct\n(full 1B)',
                    'Wall-clock time  (hours)', '{:.0f}h', 'time',
                    SHADES_1B,
                    f'Time — B200 1B\n{row_note}')

    draw_flop_breakdown(ax_flop, STAGES_1B, mfu_val,
                        1e9, 'Direct\n(full 1B)',
                        SHADES_1B,
                        f'FLOPs — independent of MFU\n(same both rows)')

    if row == 0:
        patches = [mpatches.Patch(color=c, label=s[0])
                   for c, s in zip(SHADES_1B, STAGES_1B)]
        ax_cost.legend(handles=patches, fontsize=7.5, framealpha=0.25,
                       facecolor='#222240', labelcolor='white',
                       title='Expansion stages', title_fontsize=7.5,
                       loc='upper left')

fig2.suptitle(
    'B200-only Training  —  1B Model  (100M→200M→600M→1B all on B200)\n'
    f'B200: {B200_TFLOPS} TFLOPS  ·  {B200_VRAM} GB VRAM  ·  ${B200_CPH}/hr  '
    f'(vast.ai May 2026)  ·  75% Chinchilla = {chin_toks_1b/1e9:.0f}B tokens  ·  '
    f'Progressive compute: {flop_prog_1b/1e15:.0f} PF  '
    f'vs Direct: {flop_dir_1b/1e15:.0f} PF  '
    f'(saves {(flop_dir_1b-flop_prog_1b)/flop_dir_1b*100:.0f}% FLOPs)',
    color='white', fontsize=9.5, y=0.975)

fig2.text(0.5, 0.01,
          'VRAM check: 1B × 120 bytes/param = 120 GB < 179 GB B200  ✓  all stages fit on one B200  ·  '
          'FLOPs saved because 100M+200M+600M stages are processed at smaller model size  ·  '
          'FLOPs panel is MFU-independent (same for both rows)',
          ha='center', color='#555577', fontsize=8)

out2 = os.path.join(OUT, 'b200_1b_cost.png')
plt.savefig(out2, dpi=140, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print(f'Saved → {out2}')
