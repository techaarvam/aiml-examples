"""
plot_100m_cost.py — Training cost and time comparison for a ~100M parameter model.

Progressive strategy: linear inner_dim expansion (+128 per step), matching actual
architecture — d256→d384→d512→d640→d768→d896.

N values from actual architecture (vecDims=256 frozen, N_core = 96*d^2 + adapters):
  d256:  32M total  (no upscale/downscale — first stage)
  d384:  40M total
  d512:  51M total
  d640:  65M total
  d768:  82M total
  d896: 103M total  ← target (~100M)

Recovery overhead: 10% of one epoch = 0.1 × 20M = 2M tokens per expansion.

VRAM: 12 GB per 100M params (empirical, batch=160, seq=256, no grad ckpt).
75% Chinchilla: D = 15 × N.
MFU: two scenarios shown — 0.35 (optimistic/compute-bound) and 0.15 (realistic
small-batch consumer GPU). 7h (MFU=0.35) is likely an underestimate; ~17h at
MFU=0.15. Actual time depends on achievable batch size within 12 GB VRAM.

Pricing: vast.ai May 2026 (consumer=search, datacenter=actual listings).
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

CHIN   = 15       # 75% of Chinchilla (D = 20N × 0.75)
FPT    = 6        # FLOPs per token per param (forward 2N + backward 4N)
RECOV  = 2e6      # 10% of one 20M-token epoch

# ── Architecture: linear +128 dim steps ───────────────────────────────────────
# N_embed = 50257×256×2 = 25.8M frozen
# N_core  ≈ 96×d² + 2×d×256 (adapters)  [no adapters at d256]
N_EMBED = 25.8e6
def n_total(d):
    core = 96 * d**2 + (2 * d * 256 if d > 256 else 0)
    return core + N_EMBED

DIMS    = [256, 384, 512, 640, 768, 896]
N_VALS  = {d: n_total(d) for d in DIMS}
N_TARG  = N_VALS[896]   # ~103M — target

print('Architecture sizes:')
for d, n in N_VALS.items():
    print(f'  d{d}: {n/1e6:.1f}M params  (N_core={96*d**2/1e6:.1f}M)')
print()

# ── GPU table ─────────────────────────────────────────────────────────────────
GPU_DATA = [
    # name          vram  TFLOPS  $/hr
    ('RTX 5070',    12,   100,   0.08),
    ('RTX 3090',    24,    71,   0.17),
    ('RTX 4090',    24,   165,   0.48),
    ('RTX 5090',    32,   400,   0.77),
    ('A100-40',     40,   312,   1.05),
    ('A100-80',     80,   312,   1.20),
    ('H100-SXM',    80,   989,   4.35),
    ('H200',       140,  1000,   4.00),
    ('B200',       179,  2250,   4.24),
]
GPU       = {r[0]: r[1:] for r in GPU_DATA}
GPU_ORDER = [r[0] for r in GPU_DATA]
COLORS    = {
    'RTX 5070': '#4488ff', 'RTX 3090': '#5599ee',
    'RTX 4090': '#66aadd', 'RTX 5090': '#77bbcc',
    'A100-40':  '#ff9944', 'A100-80':  '#ff7733',
    'H100-SXM': '#ff5522', 'H200':     '#dd3311',
    'B200':     '#991100',
}

def compute(name, n, tokens, mfu):
    tf, cph = GPU[name][1], GPU[name][2]
    tps     = tf * 1e12 * mfu / (FPT * n)
    hours   = tokens / tps / 3600
    return hours * cph, hours

# ── Progressive stages: d256 → d384 → ... → d896 (all on RTX 5070) ───────────
def make_prog_stages(mfu):
    stages = []
    chin_done = 0.0
    for i, d in enumerate(DIMS):
        n        = N_VALS[d]
        chin_now = CHIN * n
        new_toks = chin_now - chin_done
        recovery = RECOV if i > 0 else 0.0
        tokens   = new_toks + recovery
        c, h     = compute('RTX 5070', n, tokens, mfu)
        stages.append((f'd{d}', n, tokens, c, h))
        chin_done = chin_now
    return stages

# ── Direct: train N_TARG on each GPU ─────────────────────────────────────────
chin_tokens = CHIN * N_TARG   # tokens for direct training

def direct_all(mfu):
    costs, hours = {}, {}
    for name in GPU_ORDER:
        c, h = compute(name, N_TARG, chin_tokens, mfu)
        costs[name] = c
        hours[name] = h
    return costs, hours

# ── Summary printout ──────────────────────────────────────────────────────────
total_flops_no_ckpt = FPT * N_TARG * chin_tokens
total_flops_ckpt    = total_flops_no_ckpt * (8/6)   # grad ckpt adds ~33% (one extra fwd)

print(f'Target: d896  N = {N_TARG/1e6:.1f}M params')
print(f'75% Chinchilla: {chin_tokens/1e9:.2f}B tokens')
print(f'Total compute:  {total_flops_no_ckpt/1e15:.0f} PetaFLOPs  (no grad ckpt, 6N formula)')
print(f'                {total_flops_ckpt/1e15:.0f} PetaFLOPs  (with grad ckpt, ~8N)')
print()

for mfu_val, label in [(0.35, 'MFU=0.35 (optimistic)'), (0.15, 'MFU=0.15 (realistic small-batch)')]:
    stages = make_prog_stages(mfu_val)
    prog_cost  = sum(s[3] for s in stages)
    prog_hours = sum(s[4] for s in stages)
    prog_toks  = sum(s[2] for s in stages)
    direct_costs, direct_hours = direct_all(mfu_val)
    print(f'── {label} ──')
    print(f'  Direct 5070:  {direct_hours["RTX 5070"]:.1f}h  ${direct_costs["RTX 5070"]:.2f}')
    print(f'  Progressive:  {prog_hours:.1f}h  ${prog_cost:.2f}  ({prog_toks/1e9:.2f}B tokens total)')
    print(f'  Overhead:    +{(prog_toks-chin_tokens)/1e6:.0f}M tokens  '
          f'+{prog_hours - direct_hours["RTX 5070"]:.1f}h  '
          f'+${prog_cost - direct_costs["RTX 5070"]:.2f}')
    print()

# ── Plot — two rows, one per MFU scenario ────────────────────────────────────
fig = plt.figure(figsize=(18, 12), facecolor=DARK_BG)
gs  = GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.35,
               top=0.90, bottom=0.07, left=0.06, right=0.97)

MFU_SCENARIOS = [
    (0.35, 'MFU=0.35  (optimistic — compute-bound, large batch)'),
    (0.15, 'MFU=0.15  (realistic — small batch constrained by 12 GB VRAM)'),
]

bar_w = 0.52
STAGE_COLORS = [COLORS['RTX 5070']] * len(DIMS)
DIM_SHADES   = ['#2255cc', '#3366dd', '#4477ee', '#5588ff', '#66aaff', '#77bbff']

for row, (mfu_val, mfu_label) in enumerate(MFU_SCENARIOS):
    stages = make_prog_stages(mfu_val)
    prog_cost_total  = sum(s[3] for s in stages)
    prog_hours_total = sum(s[4] for s in stages)
    prog_toks_total  = sum(s[2] for s in stages)
    direct_costs, direct_hours = direct_all(mfu_val)

    ax_cost    = fig.add_subplot(gs[row, 0])
    ax_time    = fig.add_subplot(gs[row, 1])
    ax_scatter = fig.add_subplot(gs[row, 2])
    for ax in (ax_cost, ax_time, ax_scatter):
        ax.set_facecolor(DARK_BG)

    row_title = (f'{mfu_label}\n'
                 f'Total compute: {total_flops_no_ckpt/1e15:.0f} PF (no ckpt)  '
                 f'/ {total_flops_ckpt/1e15:.0f} PF (grad ckpt)  '
                 f'for {chin_tokens/1e9:.2f}B tokens')

    # ── Cost panel ─────────────────────────────────────────────────────────────
    bottom = 0.0
    for (lbl, n, toks, cost, h), shade in zip(stages, DIM_SHADES):
        ax_cost.bar('Progressive\nd256→d896', cost, bottom=bottom,
                    color=shade, alpha=0.90, width=bar_w, label=lbl)
        if cost > prog_cost_total * 0.06:
            ax_cost.text(0, bottom + cost / 2, f'${cost:.2f}',
                         ha='center', va='center', color='white', fontsize=7.5)
        bottom += cost

    ax_cost.text(0, prog_cost_total * 1.06, f'${prog_cost_total:.2f}',
                 ha='center', color='white', fontsize=9, fontweight='bold')

    for name in GPU_ORDER:
        c = direct_costs[name]
        ax_cost.bar(name, c, color=COLORS[name], alpha=0.85, width=bar_w)
        ax_cost.text(name, c * 1.06, f'${c:.2f}',
                     ha='center', color='#ccc', fontsize=7)

    ax_cost.set_ylabel('Total cost  ($)', color='#aaa', fontsize=8.5)
    ax_cost.set_title(f'Cost  —  {chin_tokens/1e9:.1f}B tokens\n{row_title}',
                      color='white', fontsize=8)
    ax_cost.tick_params(colors='#888', labelsize=7.5)
    ax_cost.tick_params(axis='x', rotation=32)
    for sp in ax_cost.spines.values(): sp.set_visible(False)
    ax_cost.grid(axis='y', alpha=0.08, color='#555')
    if row == 0:
        patches = [mpatches.Patch(color=c, label=s[0])
                   for c, s in zip(DIM_SHADES, stages)]
        ax_cost.legend(handles=patches, fontsize=7, framealpha=0.25,
                       facecolor='#222240', labelcolor='white',
                       title='Expansion stages', title_fontsize=7, loc='upper left')

    # ── Time panel ─────────────────────────────────────────────────────────────
    bottom = 0.0
    for (lbl, n, toks, cost, h), shade in zip(stages, DIM_SHADES):
        ax_time.bar('Progressive\nd256→d896', h, bottom=bottom,
                    color=shade, alpha=0.90, width=bar_w)
        if h > prog_hours_total * 0.06:
            ax_time.text(0, bottom + h / 2, f'{h:.1f}h',
                         ha='center', va='center', color='white', fontsize=7.5)
        bottom += h

    ax_time.text(0, prog_hours_total * 1.06, f'{prog_hours_total:.1f}h',
                 ha='center', color='white', fontsize=9, fontweight='bold')

    for name in GPU_ORDER:
        h = direct_hours[name]
        ax_time.bar(name, h, color=COLORS[name], alpha=0.85, width=bar_w)
        ax_time.text(name, h * 1.06, f'{h:.1f}h',
                     ha='center', color='#ccc', fontsize=7)

    ax_time.set_ylabel('Wall-clock time  (hours)', color='#aaa', fontsize=8.5)
    ax_time.set_title('Time  —  same token budget', color='white', fontsize=8)
    ax_time.tick_params(colors='#888', labelsize=7.5)
    ax_time.tick_params(axis='x', rotation=32)
    for sp in ax_time.spines.values(): sp.set_visible(False)
    ax_time.grid(axis='y', alpha=0.08, color='#555')

    # ── Scatter: cost vs time ──────────────────────────────────────────────────
    all_h    = [direct_hours[n]  for n in GPU_ORDER]
    all_c    = [direct_costs[n]  for n in GPU_ORDER]
    all_col  = [COLORS[n]        for n in GPU_ORDER]
    all_tf   = [GPU[n][1]        for n in GPU_ORDER]
    sizes    = [tf / max(all_tf) * 350 + 40 for tf in all_tf]

    ax_scatter.scatter(all_h, all_c, c=all_col, s=sizes, alpha=0.88,
                       zorder=5, edgecolors='white', linewidths=0.4)
    for name, h, c in zip(GPU_ORDER, all_h, all_c):
        ha = 'left'
        dx = max(all_h) * 0.015
        if name in ('RTX 3090', 'A100-80'):
            ha = 'right'; dx = -dx
        ax_scatter.annotate(name, (h, c), (h + dx, c + max(all_c)*0.01),
                            color='#ccc', fontsize=7, ha=ha)

    # Progressive point
    ax_scatter.scatter([prog_hours_total], [prog_cost_total], c='white', s=130,
                       zorder=6, marker='*',
                       label=f'Progressive d256→d896\n'
                             f'${prog_cost_total:.2f}  {prog_hours_total:.1f}h\n'
                             f'{prog_toks_total/1e9:.2f}B tokens '
                             f'(+{(prog_toks_total-chin_tokens)/1e6:.0f}M recovery)')
    ax_scatter.set_xlabel('Time  (hours)', color='#aaa', fontsize=8.5)
    ax_scatter.set_ylabel('Cost  ($)', color='#aaa', fontsize=8.5)
    ax_scatter.set_title('Cost vs Time  (bubble ∝ TFLOPS)', color='white', fontsize=8)
    ax_scatter.legend(fontsize=7.5, framealpha=0.3, facecolor='#222240',
                      labelcolor='white', loc='upper right')
    ax_scatter.tick_params(colors='#888', labelsize=8)
    for sp in ax_scatter.spines.values(): sp.set_visible(False)
    ax_scatter.grid(alpha=0.08, color='#555')

# ── Global title ──────────────────────────────────────────────────────────────
fig.suptitle(
    '~100M Parameter Model  (d896, 103M)  —  Progressive: d256→d384→d512→d640→d768→d896  (+128 dims/step)\n'
    f'All stages fit on RTX 5070 (12 GB)  ·  Recovery = 2M tokens/expansion (10% of 20M epoch)  ·  '
    f'75% Chinchilla = {chin_tokens/1e9:.1f}B tokens  ·  '
    f'{total_flops_no_ckpt/1e15:.0f} PF (no ckpt)  /  {total_flops_ckpt/1e15:.0f} PF (grad ckpt)',
    color='white', fontsize=9.5, y=0.975)

fig.text(0.5, 0.01,
         'vast.ai pricing May 2026  ·  consumer=search price  ·  datacenter=actual on-demand listings  ·  '
         'MFU uncertainty is the dominant timing variable — actual step/s on your hardware will calibrate this',
         ha='center', color='#555577', fontsize=7.5)

out_path = os.path.join(OUT, 'training_cost_100m.png')
plt.savefig(out_path, dpi=140, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print(f'Saved → {out_path}')
