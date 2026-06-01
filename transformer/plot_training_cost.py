"""
plot_training_cost.py — Progressive expansion vs direct training for 1B parameter model.

VRAM model — two scenarios, both calibrated from empirical data:
  Conservative: ~120 bytes/param  (12 GB per 100M params)
    Calibrated from user's 51M model at batch=160, seq=256 filling 12 GB.
    Dominant term is activation memory (~11.2 GB), not weights (~0.8 GB).

  Optimistic:   ~40 bytes/param   (4 GB per 100M params)
    Aggressive gradient checkpointing + ~3x smaller physical batch.

MFU: two scenarios — 0.35 (optimistic) and 0.15 (realistic small-batch consumer GPU).
Calibrated baseline: 0.9 batches/sec × batch=512 × window=256 = 117,965 tok/s
for d512 (51M model) on RTX 5070 (100 TFLOPS) → MFU ≈ 36% at batch=512.
At smaller effective batches (memory-constrained large models) MFU is lower.
0.15 represents realistic throughput when VRAM limits batch size heavily.

Recovery: 2M tokens per expansion = 10% of one 20M-token epoch.
75% Chinchilla: D = 15 × N.

GPU pricing: dedicated on-demand (not spot), vast.ai May 2026.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, 'output')
os.makedirs(OUT, exist_ok=True)
DARK_BG = '#1a1a2e'

FPT      = 6      # FLOPs per token (forward 2N + backward 4N)
CHIN     = 15     # 75% of Chinchilla optimal (D = 20N × 0.75)
RECOV    = 2e6    # 10% of 20M-token epoch = 2M tokens recovery per expansion
N_TARGET = 1e9    # goal: 1B parameters

BYTES_PER_PARAM_CONSERVATIVE = 120   # 12 GB per 100M: practical batch, no grad ckpt
BYTES_PER_PARAM_OPTIMISTIC   = 40    # 4 GB per 100M: aggressive grad ckpt + small batch

# ── GPU table: (VRAM_GB, fp16_TFLOPS_dense, dedicated_$/hr, hex_color) ──────────
GPU_DATA = [
    # name         vram  TFLOPS  $/hr  color
    ('RTX 5070',    12,   100,  0.08, '#4488ff'),  # vast.ai search: $0.08/hr
    ('RTX 3090',    24,    71,  0.17, '#5599ee'),  # vast.ai search: $0.17/hr
    ('RTX 4090',    24,   165,  0.48, '#66aadd'),  # vast.ai search: $0.48/hr
    ('RTX 5090',    32,   400,  0.77, '#77bbcc'),  # vast.ai search: $0.77/hr
    ('A100-40',     40,   312,  1.05, '#ff9944'),  # listing #36449303: $1.003/hr
    ('A100-80',     80,   312,  1.20, '#ff7733'),  # listing #32304443: $1.205/hr
    ('H100-SXM',    80,   989,  4.35, '#ff5522'),  # listing #38248504: ~$4.35/hr
    ('H200',       140,  1000,  4.00, '#dd3311'),  # listing #38231216: $3.951/hr
    ('B200',       179,  2250,  4.24, '#991100'),  # listing #33945597: $4.236/hr
]
GPU       = {r[0]: r[1:] for r in GPU_DATA}
GPU_ORDER = [r[0] for r in GPU_DATA]

def max_params(name, bytes_per_param):
    return GPU[name][0] * 1e9 / bytes_per_param

def compute(name, n, tokens, mfu):
    tf  = GPU[name][1]
    cph = GPU[name][2]
    tps = tf * 1e12 * mfu / (FPT * n)
    h   = tokens / tps / 3600
    return h * cph, h

def chinchilla_stages(sizes_and_gpus):
    """Returns list of (N_params, GPU_name, tokens_this_stage)."""
    stages    = []
    chin_done = 0.0
    for i, (n, gpu_name) in enumerate(sizes_and_gpus):
        chin_target = CHIN * n
        new_tokens  = chin_target - chin_done
        recovery    = RECOV if i > 0 else 0.0
        stages.append((n, gpu_name, new_tokens + recovery))
        chin_done = chin_target
    return stages

# ── Scenario A: Conservative (120 bytes/param) ──────────────────────────────────
# 1B needs 120 GB → minimum GPU: H200 (140 GB)
STAGES_CONS = chinchilla_stages([
    (100e6, 'RTX 5070'),   # 12 GB  fits 100M  ✓
    (200e6, 'RTX 4090'),   # 24 GB  fits 200M  ✓
    (600e6, 'H100-SXM'),   # 80 GB  fits 667M  ✓
    (1e9,   'H200'),       # 140 GB fits 1.2B  ✓
])

# ── Scenario B: Optimistic (40 bytes/param, grad ckpt + small batch) ────────────
# 1B needs 40 GB → minimum GPU: A100-40 (40 GB)
STAGES_OPT = chinchilla_stages([
    (300e6, 'RTX 5070'),   # 12 GB  fits 300M  ✓
    (600e6, 'RTX 4090'),   # 24 GB  fits 600M  ✓
    (1e9,   'A100-40'),    # 40 GB  fits 1.0B  ✓
])

def total_cost_hours(stages, mfu):
    costs, hours = [], []
    for n, gpu_name, tokens in stages:
        c, h = compute(gpu_name, n, tokens, mfu)
        costs.append(c)
        hours.append(h)
    return costs, hours, sum(costs), sum(hours)

def direct_options(bytes_per_param, mfu):
    out = {}
    for name in GPU_ORDER:
        if max_params(name, bytes_per_param) >= N_TARGET:
            c, h = compute(name, N_TARGET, CHIN * N_TARGET, mfu)
            out[name] = (c, h)
    return out

# ── Summary printout ──────────────────────────────────────────────────────────
chin_tokens = CHIN * N_TARGET
print(f'Target: 1B params   75% Chinchilla: {chin_tokens/1e9:.0f}B tokens')
print(f'Recovery: {int(RECOV/1e6)}M tokens per expansion (10% of 20M epoch)')
print()

for mfu_val, lbl in [(0.35, 'MFU=0.35 (optimistic)'), (0.15, 'MFU=0.15 (realistic)')]:
    print(f'── {lbl} ──')
    for bpp, bpp_lbl, stages in [
        (BYTES_PER_PARAM_CONSERVATIVE, 'Conservative', STAGES_CONS),
        (BYTES_PER_PARAM_OPTIMISTIC,   'Optimistic',   STAGES_OPT),
    ]:
        _, _, tc, th = total_cost_hours(stages, mfu_val)
        direct = direct_options(bpp, mfu_val)
        prog_toks = sum(s[2] for s in stages)
        print(f'  {bpp_lbl}: progressive  {th:>6.0f}h  ${tc:>6.0f}  '
              f'({prog_toks/1e9:.2f}B tokens total)')
        for name, (c, h) in list(direct.items())[:3]:
            print(f'    Direct {name}: {h:.0f}h  ${c:.0f}')
    print()

# ── Plot — two rows for MFU scenarios, 3 cols per row ─────────────────────────
MFU_SCENARIOS = [
    (0.35, 'MFU=0.35  (optimistic — compute-bound, large batch)'),
    (0.15, 'MFU=0.15  (realistic — small batch, memory-constrained)'),
]

fig = plt.figure(figsize=(18, 17), facecolor=DARK_BG)
gs  = GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.35,
               top=0.92, bottom=0.05, left=0.06, right=0.97,
               height_ratios=[1.2, 1.1, 1.1])

# ── VRAM capacity panel (top, spans full width) ───────────────────────────────
ax_cap = fig.add_subplot(gs[0, :])
ax_cap.set_facecolor(DARK_BG)

cap_cons = [max_params(g, BYTES_PER_PARAM_CONSERVATIVE) / 1e9 for g in GPU_ORDER]
cap_opt  = [max_params(g, BYTES_PER_PARAM_OPTIMISTIC)   / 1e9 for g in GPU_ORDER]
cols     = [GPU[g][3] for g in GPU_ORDER]
y        = np.arange(len(GPU_ORDER))
bar_h    = 0.35

bars_c = ax_cap.barh(y - bar_h/2, cap_cons, bar_h, color=cols, alpha=0.75,
                     label='Conservative (120 bytes/param — no grad ckpt, practical batch)',
                     hatch='///', edgecolor='none')
bars_o = ax_cap.barh(y + bar_h/2, cap_opt, bar_h, color=cols, alpha=0.95,
                     label='Optimistic (40 bytes/param — grad ckpt + 3× smaller batch)')

ax_cap.axvline(N_TARGET / 1e9, color='white', lw=1.8, ls='--', alpha=0.9)
ax_cap.text(N_TARGET / 1e9 + 0.04, len(GPU_ORDER) - 0.45,
            '← 1B\ntarget', color='white', fontsize=8, va='top', linespacing=1.4)

for bar, val in zip(bars_c, cap_cons):
    if val >= 0.08:
        ax_cap.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                    f'{val:.2f}B', color='#888', fontsize=7, va='center')
for bar, val in zip(bars_o, cap_opt):
    if val >= 0.08:
        ax_cap.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                    f'{val:.2f}B', color='#ccc', fontsize=7, va='center')

ax_cap.set_yticks(y)
ax_cap.set_yticklabels(GPU_ORDER, fontsize=8.5)
ax_cap.set_xlabel(
    'Max trainable parameters   '
    '[Conservative: 120 bytes/param (calibrated: 51M fills 12 GB at batch=160, seq=256)   '
    '|   Optimistic: 40 bytes/param (grad ckpt + 3× smaller batch)]',
    color='#aaa', fontsize=7.5)
ax_cap.set_title('GPU VRAM capacity — max trainable model size for 1B target',
                 color='white', fontsize=10)
ax_cap.tick_params(colors='#888', labelsize=8.5)
ax_cap.legend(fontsize=8, framealpha=0.25, facecolor='#222240', labelcolor='white',
              loc='lower right')
ax_cap.set_xlim(0, max(cap_opt) * 1.15)
for sp in ax_cap.spines.values(): sp.set_visible(False)
ax_cap.grid(axis='x', alpha=0.10, color='#555')
ax_cap.axhspan(3.5, len(GPU_ORDER), alpha=0.04, color='#ff7733')
ax_cap.axhspan(-0.5, 3.5, alpha=0.04, color='#4488ff')
ax_cap.text(max(cap_opt) * 1.12, 1.5, 'Consumer',   color='#4488ff', fontsize=7.5,
            ha='right', va='center', style='italic')
ax_cap.text(max(cap_opt) * 1.12, 6.5, 'Datacenter', color='#ff7733', fontsize=7.5,
            ha='right', va='center', style='italic')

# ── Cost / time panels per MFU scenario ──────────────────────────────────────
STAGE_SHADES_C = ['#2255cc', '#3366dd', '#4477ee', '#5588ff']   # cons stages
STAGE_SHADES_O = ['#2244aa', '#3366cc', '#4488dd']              # opt stages
bar_w = 0.48

def draw_stacked(ax, stages, vals, stage_shades, total, fmt):
    bottom = 0.0
    for (n, gpu_name, _), v, shade in zip(stages, vals, stage_shades):
        label = f'{int(n/1e6)}M\n{gpu_name}'
        ax.bar('Prog\n(cons)' if len(stages) == 4 else 'Prog\n(opt)',
               v, bottom=bottom, color=shade, alpha=0.90, width=bar_w, label=label)
        if v > total * 0.05:
            ax.text(0 if len(stages) == 4 else 1, bottom + v / 2,
                    fmt.format(v), ha='center', va='center',
                    color='white', fontsize=7.5)
        bottom += v
    x_label = 0 if len(stages) == 4 else 1
    ax.text(x_label, total * 1.05, fmt.format(total),
            ha='center', color='white', fontsize=9, fontweight='bold')

def draw_direct(ax, direct_dict, fmt, value_key):
    for name, (c, h) in direct_dict.items():
        val = c if value_key == 'cost' else h
        col = GPU[name][3]
        ax.bar(name, val, color=col, alpha=0.85, width=bar_w)
        ax.text(name, val * 1.04, fmt.format(val),
                ha='center', color='#ccc', fontsize=7)

def annotate_savings(ax, prog_val, direct_dict, fmt, value_key):
    if not direct_dict:
        return
    cheapest = min(v[0] if value_key == 'cost' else v[1]
                   for v in direct_dict.values())
    savings = cheapest - prog_val
    pct = savings / cheapest * 100
    if savings > 0:
        ax.text(0.98, 0.97,
                f'saves {fmt.format(savings)} ({pct:.0f}%)\nvs cheapest direct',
                transform=ax.transAxes, ha='right', va='top', color='#88ff88',
                fontsize=7.5,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a2a1a', alpha=0.7))

for row, (mfu_val, mfu_label) in enumerate(MFU_SCENARIOS):
    costs_c, hours_c, tc_c, th_c = total_cost_hours(STAGES_CONS, mfu_val)
    costs_o, hours_o, tc_o, th_o = total_cost_hours(STAGES_OPT, mfu_val)
    d_cons = direct_options(BYTES_PER_PARAM_CONSERVATIVE, mfu_val)
    d_opt  = direct_options(BYTES_PER_PARAM_OPTIMISTIC,   mfu_val)

    ax_cc = fig.add_subplot(gs[row + 1, 0])   # cost conservative
    ax_co = fig.add_subplot(gs[row + 1, 1])   # cost optimistic
    ax_tc = fig.add_subplot(gs[row + 1, 2])   # time conservative + scatter ref

    for ax in (ax_cc, ax_co, ax_tc):
        ax.set_facecolor(DARK_BG)

    row_title = (f'{mfu_label}\n'
                 f'{chin_tokens/1e9:.0f}B tokens  ·  '
                 f'Recovery {int(RECOV/1e6)}M tok/expansion  ·  '
                 f'75% Chinchilla')

    # ── Cost conservative ────────────────────────────────────────────────────
    draw_stacked(ax_cc, STAGES_CONS, costs_c, STAGE_SHADES_C, tc_c, '${:.0f}')
    draw_direct(ax_cc, d_cons, '${:.0f}', 'cost')
    annotate_savings(ax_cc, tc_c, d_cons, '${:.0f}', 'cost')
    ax_cc.set_ylabel('Total cost  ($)', color='#aaa', fontsize=8.5)
    ax_cc.set_title(f'Cost — Conservative  (120 bytes/param)\n{row_title}',
                    color='white', fontsize=7.5)
    ax_cc.tick_params(colors='#888', labelsize=7.5)
    ax_cc.tick_params(axis='x', rotation=28)
    for sp in ax_cc.spines.values(): sp.set_visible(False)
    ax_cc.grid(axis='y', alpha=0.08, color='#555')
    if row == 0:
        patches = [mpatches.Patch(color=c, label=f'{int(s[0]/1e6)}M {s[1]}')
                   for c, s in zip(STAGE_SHADES_C, STAGES_CONS)]
        ax_cc.legend(handles=patches, fontsize=6.5, framealpha=0.25,
                     facecolor='#222240', labelcolor='white',
                     title='Cons. stages', title_fontsize=6.5, loc='upper left')

    # ── Cost optimistic ──────────────────────────────────────────────────────
    # Show both progressive paths on same panel as separate bars
    # Optimistic progressive
    bottom = 0.0
    x_opt = 1
    for (n, gpu_name, _), v, shade in zip(STAGES_OPT, costs_o, STAGE_SHADES_O):
        label = f'{int(n/1e6)}M\n{gpu_name}'
        ax_co.bar(x_opt, v, bottom=bottom, color=shade, alpha=0.90, width=bar_w, label=label)
        if v > tc_o * 0.05:
            ax_co.text(x_opt, bottom + v / 2, f'${v:.0f}',
                       ha='center', va='center', color='white', fontsize=7.5)
        bottom += v
    ax_co.text(x_opt, tc_o * 1.05, f'${tc_o:.0f}',
               ha='center', color='white', fontsize=9, fontweight='bold')
    ax_co.set_xticks([x_opt] + list(range(x_opt + 1, x_opt + 1 + len(d_opt))))
    # Direct on opt
    x_d = x_opt + 1
    for name, (c, h) in d_opt.items():
        ax_co.bar(x_d, c, color=GPU[name][3], alpha=0.85, width=bar_w)
        ax_co.text(x_d, c * 1.04, f'${c:.0f}', ha='center', color='#ccc', fontsize=7)
        x_d += 1
    labels = ['Prog\n(opt)'] + list(d_opt.keys())
    ax_co.set_xticks(range(x_opt, x_opt + len(labels)))
    ax_co.set_xticklabels(labels, fontsize=7.5, rotation=28, ha='right')
    annotate_savings(ax_co, tc_o, d_opt, '${:.0f}', 'cost')
    ax_co.set_ylabel('Total cost  ($)', color='#aaa', fontsize=8.5)
    ax_co.set_title(f'Cost — Optimistic  (40 bytes/param)\n{row_title}',
                    color='white', fontsize=7.5)
    ax_co.tick_params(colors='#888', labelsize=7.5)
    ax_co.tick_params(axis='x', rotation=28)
    for sp in ax_co.spines.values(): sp.set_visible(False)
    ax_co.grid(axis='y', alpha=0.08, color='#555')
    if row == 0:
        patches = [mpatches.Patch(color=c, label=f'{int(s[0]/1e6)}M {s[1]}')
                   for c, s in zip(STAGE_SHADES_O, STAGES_OPT)]
        ax_co.legend(handles=patches, fontsize=6.5, framealpha=0.25,
                     facecolor='#222240', labelcolor='white',
                     title='Opt. stages', title_fontsize=6.5, loc='upper left')

    # ── Time scatter: cost vs time for all direct-viable GPUs ───────────────
    # Conservative path viable GPUs
    all_h   = [d_cons[n][1] for n in d_cons]
    all_c   = [d_cons[n][0] for n in d_cons]
    all_col = [GPU[n][3] for n in d_cons]
    all_tf  = [GPU[n][1] for n in d_cons]
    if all_tf:
        sizes = [tf / max(all_tf) * 350 + 40 for tf in all_tf]
        ax_tc.scatter(all_h, all_c, c=all_col, s=sizes, alpha=0.88,
                      zorder=5, edgecolors='white', linewidths=0.4)
        max_h = max(all_h) if all_h else 1
        max_c = max(all_c) if all_c else 1
        for name, h, c in zip(d_cons.keys(), all_h, all_c):
            ha = 'left'; dx = max_h * 0.015
            if name in ('A100-80', 'H200'):
                ha = 'right'; dx = -dx
            ax_tc.annotate(name, (h, c), (h + dx, c + max_c * 0.01),
                           color='#ccc', fontsize=7, ha=ha)

    # Progressive star (conservative path)
    prog_c_toks = sum(s[2] for s in STAGES_CONS)
    ax_tc.scatter([th_c], [tc_c], c='white', s=150, zorder=6, marker='*',
                  label=f'Progressive (cons)\n${tc_c:.0f}  {th_c:.0f}h\n'
                        f'{prog_c_toks/1e9:.2f}B tokens')
    # Progressive star (optimistic path)
    prog_o_toks = sum(s[2] for s in STAGES_OPT)
    ax_tc.scatter([th_o], [tc_o], c='#aaffaa', s=150, zorder=6, marker='*',
                  label=f'Progressive (opt)\n${tc_o:.0f}  {th_o:.0f}h\n'
                        f'{prog_o_toks/1e9:.2f}B tokens')

    ax_tc.set_xlabel('Time  (hours)', color='#aaa', fontsize=8.5)
    ax_tc.set_ylabel('Cost  ($)', color='#aaa', fontsize=8.5)
    ax_tc.set_title('Cost vs Time  (direct cons. GPUs — bubble ∝ TFLOPS)',
                    color='white', fontsize=8)
    ax_tc.legend(fontsize=7, framealpha=0.3, facecolor='#222240',
                 labelcolor='white', loc='upper right')
    ax_tc.tick_params(colors='#888', labelsize=8)
    for sp in ax_tc.spines.values(): sp.set_visible(False)
    ax_tc.grid(alpha=0.08, color='#555')

# ── Global title and footnote ─────────────────────────────────────────────────
fig.suptitle(
    'Progressive Expansion vs Direct Training  —  1B Parameter Model\n'
    'Cons. ladder: 100M (5070) → 200M (4090) → 600M (H100-SXM) → 1B (H200)   '
    '·   Opt. ladder: 300M (5070) → 600M (4090) → 1B (A100-40)',
    color='white', fontsize=10, y=0.975)

fig.text(
    0.5, 0.01,
    'VRAM calibration: 51M model fills 12 GB at batch=160, seq=256 (activation-dominated)  ·  '
    f'75% Chinchilla = {chin_tokens/1e9:.0f}B tokens  ·  '
    f'Recovery = {int(RECOV/1e6)}M tokens/expansion (10% of 20M epoch)  ·  '
    'MFU uncertainty is the dominant timing variable  ·  '
    'Consumer GPU: vast.ai search price  ·  Datacenter: actual vast.ai on-demand listings May 2026',
    ha='center', color='#555577', fontsize=7.5)

out_path = os.path.join(OUT, 'training_cost_comparison.png')
plt.savefig(out_path, dpi=140, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print(f'Saved → {out_path}')
