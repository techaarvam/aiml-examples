"""
plot_loss_trend.py — BTM Run 4 full loss curve
X-axis: cumulative tokens (M), linearly scaled to training time.
  Each 10% heartbeat = 2M tokens (max_tokens=20M / epoch, always).
  w64 epoch-end points = 20M each (no intra-epoch heartbeats recorded there).

Run:
  python plot_loss_trend.py
Output:
  output/loss_trend_full.png
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, 'output')
os.makedirs(OUT, exist_ok=True)
DARK_BG = '#1a1a2e'

TOKENS_PER_EPOCH  = 20   # M
TOKENS_PER_HB     = 2    # M  (10% of 20M)

# ── Data: 10% heartbeat losses (or epoch-end where heartbeats not recorded) ──
# w64: epoch-end only (M1 machine, no intra-epoch heartbeats)
w64 = [6.4417, 6.2867, 6.1760, 6.1506, 6.1463, 6.1165, 6.0920]   # ep 1-7

# w128 branch: 1 epoch-end (M1)
w128 = [5.9181]

# post-merge ep11 (10 heartbeats @ 10%..100%)
ep11 = [5.9943, 5.9308, 5.8991, 5.8788, 5.8638, 5.8522, 5.8426, 5.8344, 5.8273, 5.8211]
ep12 = [5.8077, 5.7975, 5.7909, 5.7851, 5.7804, 5.7763, 5.7727, 5.7692, 5.7661, 5.7632]
ep13 = [5.7935, 5.7396, 5.7166, 5.7029, 5.6933, 5.6858, 5.6799, 5.6749, 5.6706, 5.6670]

# d384 warm-start (ep14); 2.6% mention excluded
ep14_warm = [6.1643, 5.9641, 5.8832, 5.8373, 5.8071, 5.7852, 5.7685, 5.7550, 5.7438, 5.7343]

# d384 cold-start control (ep1, random init, same arch + data as ep14)
ep14_cold = [8.5458, 8.3711, 8.2823, 8.2225, 8.1774, 8.1407, 8.1100, 8.0834, 8.0599, 8.0388]

# d512 expansion (ep15); batch=1280 on server — still 20M tokens/epoch
ep15 = [6.4423, 6.1993, 6.0597, 5.9673, 5.9022, 5.8532, 5.8146, 5.7828, 5.7565, 5.7338]

# d512 SDPA ep16 canonical (old unfused ep16 skipped); batch=512
ep16 = [5.6178, 5.4536, 5.3651, 5.3070, 5.2644, 5.2313, 5.2046, 5.1825, 5.1636, 5.1473]

# ep17-20: warm optimizer; killed cold-optimizer ep17 and start=5.36 mention excluded
ep17 = [5.1698, 5.1327, 5.1084, 5.0909, 5.0768, 5.0650, 5.0550, 5.0463, 5.0385, 5.0315]
ep18 = [5.1333, 5.0999, 5.0788, 5.0547, 5.0385, 5.0269, 5.0159, 5.0074, 5.0008, 4.9953]
ep19 = [5.1215, 5.0921, 5.0734, 5.0594, 5.0483, 5.0392, 5.0313, 5.0244, 5.0183, 5.0129]
ep20 = [5.0933, 5.0460, 5.0170]   # partial (3 heartbeats so far)

# ── Assign cumulative token positions (M) ─────────────────────────────────
# w64: 7 epoch-end points, each = 20M tokens
w64_x   = [TOKENS_PER_EPOCH * (i+1) for i in range(len(w64))]          # 20,40,...,140
cursor  = w64_x[-1]

# w128: 1 epoch-end = 20M
w128_x  = [cursor + TOKENS_PER_EPOCH]
cursor  = w128_x[-1]

def hb_x(pts, cursor):
    """Return (x_list, new_cursor) for a heartbeat series starting after cursor."""
    xs = [cursor + TOKENS_PER_HB * (i+1) for i in range(len(pts))]
    return xs, xs[-1]

ep11_x,      cursor = hb_x(ep11, cursor)
ep12_x,      cursor = hb_x(ep12, cursor)
ep13_x,      cursor = hb_x(ep13, cursor)
ep14_warm_x, cursor = hb_x(ep14_warm, cursor)
ep14_cold_x          = ep14_warm_x                # aligned: same epoch, same data
ep15_x,      cursor = hb_x(ep15, cursor)
ep16_x,      cursor = hb_x(ep16, cursor)
ep17_x,      cursor = hb_x(ep17, cursor)
ep18_x,      cursor = hb_x(ep18, cursor)
ep19_x,      cursor = hb_x(ep19, cursor)
ep20_x,      cursor = hb_x(ep20, cursor)

# ── Phase boundaries (in M tokens) for vertical lines ─────────────────────
boundaries = {
    'w128':        w128_x[0]        - TOKENS_PER_EPOCH / 2,
    'post-merge':  ep11_x[0]        - TOKENS_PER_HB / 2,
    'w256':        ep13_x[0]        - TOKENS_PER_HB / 2,
    'd384 expand': ep14_warm_x[0]   - TOKENS_PER_HB / 2,
    'd512 expand': ep15_x[0]        - TOKENS_PER_HB / 2,
    'SDPA':        ep16_x[0]        - TOKENS_PER_HB / 2,
}

# ── Plot ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 8), facecolor=DARK_BG)
ax.set_facecolor(DARK_BG)

# Phase boundary verticals + labels
for label, bx in boundaries.items():
    ax.axvline(bx, color='#444466', linewidth=0.8, linestyle='--', zorder=2)
    ax.text(bx + 0.5, 0.98, label, color='#666688',
            fontsize=6.5, va='top', rotation=90, alpha=0.85,
            transform=ax.get_xaxis_transform())

# Main warm-start series
def plot_series(ax, xs, ys, color, label, marker='o', lw=1.4, ms=3.5, ls='-', zorder=4):
    ax.plot(xs, ys, color=color, linewidth=lw, linestyle=ls,
            marker=marker, markersize=ms, label=label, zorder=zorder)

plot_series(ax, w64_x,        w64,       '#5588ff', 'w64 ep1–7 (epoch-end)')
plot_series(ax, w128_x,       w128,      '#88ccff', 'w128 ep1 (BTM branch)', marker='s', ms=5)
plot_series(ax, ep11_x+ep12_x, ep11+ep12,'#ffcc44', 'post-merge ep11–12')
plot_series(ax, ep13_x,       ep13,      '#ff9933', 'w256 ep13')
plot_series(ax, ep14_warm_x,  ep14_warm, '#ff5577', 'd384 ep14 (warm-start)')
plot_series(ax, ep15_x,       ep15,      '#cc44ff', 'd512 ep15 (dim expand)')
sdpa_x = ep16_x + ep17_x + ep18_x + ep19_x + ep20_x
sdpa_y = ep16   + ep17   + ep18   + ep19   + ep20
plot_series(ax, sdpa_x, sdpa_y, '#33ffcc', 'd512 SDPA ep16–20')

# Cold-start overlay (dashed, aligned with ep14)
plot_series(ax, ep14_cold_x, ep14_cold, '#ffaa00',
            'd384 ep1 cold-start (random init)',
            marker='^', ms=5, ls='--', lw=1.6, zorder=5)

# Annotate cold vs warm end-of-ep14
ax.annotate(f'{ep14_cold[-1]:.4f}',
            xy=(ep14_cold_x[-1], ep14_cold[-1]),
            xytext=(ep14_cold_x[-1] + 3, ep14_cold[-1] + 0.05),
            color='#ffaa00', fontsize=7.5, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#ffaa00', lw=0.8))
ax.annotate(f'{ep14_warm[-1]:.4f}',
            xy=(ep14_warm_x[-1], ep14_warm[-1]),
            xytext=(ep14_warm_x[-1] + 3, ep14_warm[-1] - 0.18),
            color='#ff5577', fontsize=7.5, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#ff5577', lw=0.8))

ax.set_xlabel('Tokens seen (M)  —  each 10% heartbeat = 2M tokens', color='#aaa', fontsize=9)
ax.set_ylabel('Loss', color='#aaa', fontsize=9)
ax.set_title('BTM Run 4 — Full Loss Curve (10% heartbeats, x-axis linear in tokens)',
             color='white', fontsize=11)
ax.tick_params(colors='#888', labelsize=7)
for sp in ['bottom', 'left']:
    ax.spines[sp].set_color('#444')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', color='#2a2a4a', linewidth=0.6, zorder=1)
ax.legend(loc='upper right', fontsize=7, framealpha=0.25,
          labelcolor='white', facecolor='#222240')

plt.tight_layout()
out_path = os.path.join(OUT, 'loss_trend_full.png')
plt.savefig(out_path, dpi=140, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print(f"Saved → {out_path}")
