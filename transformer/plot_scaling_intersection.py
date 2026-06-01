"""
plot_scaling_intersection.py — Kaplan vs Chinchilla optimal token budget
as a function of N_core (non-embedding trainable params).

Kaplan calibration:  GPT-3 175B non-embedding, 300B tokens
  D = k_K * N_core^0.37,  k_K ≈ 2.07e7

Chinchilla (standard): D = 20 * (N_embed + N_core)
Chinchilla (core only): D = 20 * N_core  [removes frozen embedding term]

N_embed = 25.8M (frozen, constant across all expansions)
N_core  = 96 * inner_dims^2  (8 layers × 12 params/layer/dim^2)

Run: python plot_scaling_intersection.py
Out: output/scaling_intersection.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, 'output')
os.makedirs(OUT, exist_ok=True)
DARK_BG = '#1a1a2e'

# ── Constants ──────────────────────────────────────────────────────────────
N_EMBED = 25.8e6   # frozen embedding + output projection (50257 × 256 × 2)
k_K     = 2.07e7   # Kaplan constant, calibrated: 175B non-embed → 300B tokens
EXP_K   = 0.37     # Kaplan exponent (= 0.27/0.73 from compute-optimal allocation)
CHIN_C  = 20       # Chinchilla coefficient

# ── Curves over N_core 1M → 100B ──────────────────────────────────────────
N_core = np.logspace(6, 11, 1000)   # 1M to 100B

D_kaplan        = k_K * N_core ** EXP_K
D_chin_total    = CHIN_C * (N_EMBED + N_core)
D_chin_core     = CHIN_C * N_core

# ── Intersections (numerical) ──────────────────────────────────────────────
def first_crossing(y1, y2, x):
    diff = y1 - y2
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    if len(sign_changes) == 0:
        return None, None
    i = sign_changes[0]
    # linear interpolation
    frac = -diff[i] / (diff[i+1] - diff[i])
    x_cross = x[i] + frac * (x[i+1] - x[i])
    y_cross = np.interp(x_cross, x, y1)
    return x_cross, y_cross

# Kaplan vs Chinchilla-total
N_cross_total, D_cross_total = first_crossing(D_kaplan, D_chin_total, N_core)
# Kaplan vs Chinchilla-core
N_cross_core, D_cross_core   = first_crossing(D_kaplan, D_chin_core, N_core)

# ── User model stages ──────────────────────────────────────────────────────
user_models = [
    ('d256\nbaseline', 256,  6.3e6),
    ('d384',           384, 14.4e6),
    ('d512\ncurrent',  512, 25.5e6),
    ('d640\nplanned',  640, 39.3e6),
    ('d768\nfinal',    768, 56.6e6),
]

# ── Plot ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 8), facecolor=DARK_BG)
ax.set_facecolor(DARK_BG)

ax.loglog(N_core, D_kaplan,
          color='#ff6688', lw=2.0,
          label=f'Kaplan:  D = {k_K:.2e} × N_core^{EXP_K}  (non-embed N, sub-linear)')
ax.loglog(N_core, D_chin_total,
          color='#44aaff', lw=2.0,
          label=f'Chinchilla:  D = {CHIN_C} × (N_embed + N_core)  (total N, linear)')
ax.loglog(N_core, D_chin_core,
          color='#44aaff', lw=1.2, ls='--', alpha=0.55,
          label=f'Chinchilla core-only:  D = {CHIN_C} × N_core  (reference)')

# Shade region where Kaplan > Chinchilla-total (user's current territory)
if N_cross_total:
    mask = N_core <= N_cross_total
    ax.fill_between(N_core[mask], D_kaplan[mask], D_chin_total[mask],
                    alpha=0.10, color='#ff6688', label='Kaplan > Chinchilla (user scale)')
    mask2 = N_core >= N_cross_total
    ax.fill_between(N_core[mask2], D_chin_total[mask2], D_kaplan[mask2],
                    alpha=0.10, color='#44aaff', label='Chinchilla > Kaplan (large scale)')

# Intersection markers
for N_x, D_x, label_x, col in [
    (N_cross_total, D_cross_total, 'Kaplan = Chinchilla-total', '#ffffff'),
    (N_cross_core,  D_cross_core,  'Kaplan = Chinchilla-core',  '#aaaaaa'),
]:
    if N_x is not None:
        ax.scatter([N_x], [D_x], color=col, s=80, zorder=6)
        ax.annotate(f'{label_x}\nN_core = {N_x/1e9:.1f}B\nD = {D_x/1e9:.0f}B tokens',
                    xy=(N_x, D_x), xytext=(N_x * 0.05, D_x * 4),
                    color=col, fontsize=7.5,
                    arrowprops=dict(arrowstyle='->', color=col, lw=0.9))

# User model vertical markers
ylim_bot = 1e8   # will be set by data; use a reference for label placement
for label, d_inner, n_core in user_models:
    D_k = k_K * n_core ** EXP_K
    D_c = CHIN_C * (N_EMBED + n_core)
    ax.axvline(n_core, color='#333355', lw=0.8, ls=':')
    ax.scatter([n_core], [D_k], color='#ff6688', s=45, zorder=5)
    ax.scatter([n_core], [D_c], color='#44aaff', s=45, zorder=5)
    ax.text(n_core, 1.1e8, label, color='#888899', fontsize=6.5,
            ha='center', va='bottom',
            transform=ax.get_xaxis_transform() if False else ax.transData)

# Axes formatting
ax.set_xlim(1e6, 1e11)
ax.set_ylim(1e8, 1e13)
ax.set_xlabel('N_core — non-embedding trainable params  (= 96 × inner_dims²)',
              color='#aaa', fontsize=9)
ax.set_ylabel('D_optimal — tokens', color='#aaa', fontsize=9)
ax.set_title(
    'Kaplan vs Chinchilla: optimal token budget as a function of N_core\n'
    f'(N_embed = 25.8M frozen constant;  k_K calibrated on GPT-3)',
    color='white', fontsize=10)
ax.tick_params(colors='#888', labelsize=7)
for sp in ['bottom', 'left', 'top', 'right']:
    ax.spines[sp].set_color('#444' if sp in ('bottom','left') else 'none')
ax.grid(True, which='both', alpha=0.15, color='#555')
ax.legend(fontsize=7.5, framealpha=0.25, labelcolor='white',
          facecolor='#222240', loc='upper left')

plt.tight_layout()
out_path = os.path.join(OUT, 'scaling_intersection.png')
plt.savefig(out_path, dpi=140, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print(f"Saved → {out_path}")
if N_cross_total:
    print(f"Kaplan = Chinchilla-total  at N_core = {N_cross_total/1e9:.2f}B  ({D_cross_total/1e9:.0f}B tokens)")
if N_cross_core:
    print(f"Kaplan = Chinchilla-core   at N_core = {N_cross_core/1e9:.2f}B  ({D_cross_core/1e9:.0f}B tokens)")
