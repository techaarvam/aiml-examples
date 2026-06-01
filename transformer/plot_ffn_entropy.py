"""
plot_ffn_entropy.py  —  spectral entropy capacity for FFN_up, Wo, Q, K, V
                         across all Run 4 checkpoints.

X-axis: global epoch number, linearly spaced (true time ordering).
  w64 ep7=7, w128 ep1=8, BTM merge≈10, w128c ep12=12, w256 ep13=13,
  d512 ep15-20=15-20.

Metric: norm_entropy = H / log(rank),  H = -Σ p_i log(p_i),  p_i = σ_i / Σσ
  norm_entropy ∈ [0,1]:  1 = all σ equal (full capacity),  0 = rank-1 (collapsed)

Panels:
  1. Heatmap  FFN_up  per layer × epoch
  2. Heatmap  Wo      per layer × epoch
  3. Line chart: mean norm_entropy for FFN_up, Wo, Q, K, V vs epoch

Output: output/ffn_entropy_capacity_all.png
"""

import os
import re
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, 'output')
os.makedirs(OUT, exist_ok=True)
DARK_BG = '#1a1a2e'
NUM_LAYERS = 8
NUM_HEADS  = 4

# ── Checkpoints (global_epoch = chronological epoch in Run 4) ──────────────
# w64 ep7  → w128 ep1 (+1)  → BTM merge ~ep10  → w128c ep12  → w256 ep13
# → d512 ep15 (dim expand) → ep16, ep18, ep19, ep20
CHECKPOINTS = [
    # label          phase    g_ep  loss    rel_path
    ('w64 m1',   'w64',    7,  6.0920, 'btm_r2_backups/model_step1_128_m1.pth'),
    ('w64 m2',   'w64',    7,  None,   'btm_r2_backups/model_step1_128_m2.pth'),
    ('w64 m3',   'w64',    7,  None,   'btm_r2_backups/model_step1_128_m3.pth'),
    ('w64 m4',   'w64',    7,  None,   'btm_r2_backups/model_step1_128_m4.pth'),
    ('w128 m1',  'w128',   8,  5.9181, 'btm_r2_backups/model_w128_m1.pth'),
    ('w128 m2',  'w128',   8,  None,   'btm_r2_backups/model_w128_m2.pth'),
    ('w128 m3',  'w128',   8,  None,   'btm_r2_backups/model_w128_m3.pth'),
    ('w128 m4',  'w128',   8,  None,   'btm_r2_backups/model_w128_m4.pth'),
    ('merged',   'merge',  10, None,   'btm_r2_backups/btm_w128_merged.pth'),
    ('w128c',    'w128c',  12, 5.7632, 'btm_r2_backups/btm_w128_cont.pth'),
    ('w256',     'w256',   13, 5.6670, 'btm_r2_backups/btm_w256_cont_ep13.pth'),
    ('d512u',    'd512u',  15, 5.7338, 'btm_r2_backups/d512_20260522/btm_d512_cont_20260522_203334/model.pth'),
    ('d512f',    'd512f',  15, 5.7338, 'btm_r2_backups/d512_20260522/btm_d512_cont_20260522_203334/model_fused.pth'),
    ('d512 ep16','d512',   16, 5.1473, 'runs/btm_d512_cont_local_20260527_070826/model.pth'),
    ('d512 ep18','d512',   18, 4.9953, 'runs/btm_d512_cont_local_20260527_203054/model.pth'),
    ('d512 ep19','d512',   19, 5.0129, 'runs/btm_d512_cont_local_20260528_223411/model.pth'),
    ('d512 ep19r','d512r', 19, None,   'runs/btm_d512_cont_local_20260528_223411/model_wo_repaired.pth'),
    ('d512 ep20','d512',   20, 4.9336, 'btm_r2_backups/d512_ep20_sdpa.pth'),
    ('d512 ep21','d512',   21, 4.9207, 'btm_r2_backups/d512_ep21_sdpa.pth'),
]

MANIFEST = os.path.join(HERE, 'analysis_checkpoints.json')

PHASE_COLORS = {
    'w64':   '#5588ff', 'w128':  '#88ccff', 'merge': '#aaaaff',
    'w128c': '#44aaff', 'w256':  '#ff9933',
    'd512u': '#ddaaff', 'd512f': '#cc88ff', 'd512r': '#888888',
    'd512': '#33ffcc', 'd512s': '#ffee55',
}


def load_manifest_checkpoints():
    if not os.path.exists(MANIFEST):
        return []
    with open(MANIFEST) as f:
        rows = json.load(f)
    out = []
    for row in rows:
        out.append((
            row['label'],
            row.get('phase', 'watch'),
            int(row['epoch']),
            row.get('loss'),
            row['rel_path'],
        ))
    return out


# ── Helpers ────────────────────────────────────────────────────────────────
def load_sd(path):
    ck = torch.load(path, map_location='cpu')
    sd = ck['model'] if (isinstance(ck, dict) and 'model' in ck) else ck
    out = {}
    for k, v in sd.items():
        k = k.replace('_orig_mod.', '')
        k = re.sub(r'^(mlp\.\d+)\.fc1\.(weight|bias)$', r'\1.0.\2', k)
        k = re.sub(r'^(mlp\.\d+)\.fc2\.(weight|bias)$', r'\1.2.\2', k)
        out[k] = v.float()
    return out


def norm_entropy(W_2d):
    S = np.linalg.svd(W_2d.numpy(), compute_uv=False)
    S = S[S > 0]
    if len(S) <= 1:
        return 0.0
    p = S / S.sum()
    return float(-np.sum(p * np.log(p + 1e-12))) / np.log(len(S))


def compute_capacities(sd):
    """Return per-matrix norm_entropy lists for FFN_up, FFN_dn, Wo, Q, K, V.
    Q/K/V are flat lists of length NUM_LAYERS*NUM_HEADS (heads within a layer are contiguous).
    Also returns q_layer/k_layer/v_layer: per-layer mean over heads (length NUM_LAYERS).
    """
    is_fused = any('qkv' in k for k in sd)
    ffn_up, ffn_dn, wo, q_s, k_s, v_s = [], [], [], [], [], []
    for L in range(NUM_LAYERS):
        ffn_up.append(norm_entropy(sd[f'mlp.{L}.0.weight']))
        ffn_dn.append(norm_entropy(sd[f'mlp.{L}.2.weight']))
        wo.append(norm_entropy(sd[f'Wo.{L}']))
        if is_fused:
            qkv   = sd[f'attentionHeads.{L}.qkv']     # [inner, 3*inner]
            inner = qkv.shape[0]
            hd    = inner // NUM_HEADS
            Q_all = qkv[:, :inner]
            K_all = qkv[:, inner:2*inner]
            V_all = qkv[:, 2*inner:]
            for h in range(NUM_HEADS):
                q_s.append(norm_entropy(Q_all[:, h*hd:(h+1)*hd]))
                k_s.append(norm_entropy(K_all[:, h*hd:(h+1)*hd]))
                v_s.append(norm_entropy(V_all[:, h*hd:(h+1)*hd]))
        else:
            for store, key in [(q_s, 'query'), (k_s, 'keys'), (v_s, 'value')]:
                W = sd[f'attentionHeads.{L}.{key}']  # [heads, in, head_dim]
                for h in range(W.shape[0]):
                    store.append(norm_entropy(W[h]))
    # Per-layer mean over heads (for heatmap display)
    q_layer = [float(np.mean(q_s[L*NUM_HEADS:(L+1)*NUM_HEADS])) for L in range(NUM_LAYERS)]
    k_layer = [float(np.mean(k_s[L*NUM_HEADS:(L+1)*NUM_HEADS])) for L in range(NUM_LAYERS)]
    v_layer = [float(np.mean(v_s[L*NUM_HEADS:(L+1)*NUM_HEADS])) for L in range(NUM_LAYERS)]
    return dict(ffn_up=ffn_up, ffn_dn=ffn_dn, wo=wo,
                q=q_s, k=k_s, v=v_s,
                q_layer=q_layer, k_layer=k_layer, v_layer=v_layer)


# ── Load ───────────────────────────────────────────────────────────────────
results = []   # (label, phase, g_epoch, loss, caps)
for label, phase, g_epoch, loss, rel_path in CHECKPOINTS + load_manifest_checkpoints():
    path = os.path.join(HERE, rel_path)
    if not os.path.exists(path):
        print(f"  SKIP: {rel_path}")
        continue
    print(f"  {label} …", flush=True)
    sd   = load_sd(path)
    caps = compute_capacities(sd)
    results.append((label, phase, g_epoch, loss, caps))
    print(f"    ffn_up={np.mean(caps['ffn_up']):.4f}  wo={np.mean(caps['wo']):.4f}  "
          f"q={np.mean(caps['q']):.4f}  k={np.mean(caps['k']):.4f}  v={np.mean(caps['v']):.4f}")


# ── Build epoch-averaged heatmap arrays ───────────────────────────────────
# Group per-layer arrays by (metric, epoch), then average same-epoch checkpoints
HEATMAP_METRICS = ('ffn_up', 'wo', 'q_layer', 'k_layer', 'v_layer')
ep_layers = defaultdict(lambda: defaultdict(list))  # metric → epoch → [8-elem lists]
for _, _, g_epoch, _, caps in results:
    for metric in HEATMAP_METRICS:
        ep_layers[metric][g_epoch].append(caps[metric])

unique_epochs = sorted({r[2] for r in results})

def avg_Z(metric):
    rows = []
    for ep in unique_epochs:
        rows.append(np.nanmean(ep_layers[metric][ep], axis=0))
    return np.array(rows).T   # (NUM_LAYERS, n_unique_epochs)

ffn_Z = avg_Z('ffn_up')
wo_Z  = avg_Z('wo')
q_Z   = avg_Z('q_layer')
k_Z   = avg_Z('k_layer')
v_Z   = avg_Z('v_layer')


def epoch_edges(centers):
    """Bin edges at midpoints; linear extrapolation at boundaries."""
    c = np.array(centers, dtype=float)
    if len(c) == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5])
    edges = np.empty(len(c) + 1)
    edges[1:-1] = (c[:-1] + c[1:]) / 2
    edges[0]    = c[0]  - (c[1]  - c[0])  / 2
    edges[-1]   = c[-1] + (c[-1] - c[-2]) / 2
    return edges

x_edges = epoch_edges(unique_epochs)
y_edges = np.arange(-0.5, NUM_LAYERS + 0.5)

# Phase boundaries: where the dominant phase changes between consecutive unique epochs
def dominant_phase(ep):
    for _, phase, g_epoch, *_ in results:
        if g_epoch == ep:
            return phase
    return ''

phase_xlines = []   # x-positions (edges) where phase changes
for j in range(1, len(unique_epochs)):
    if dominant_phase(unique_epochs[j]) != dominant_phase(unique_epochs[j - 1]):
        phase_xlines.append(x_edges[j])


# ── Plot ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 22), facecolor=DARK_BG)
gs  = fig.add_gridspec(6, 1, height_ratios=[1, 1, 1, 1, 1, 1.6], hspace=0.58)


def panel_vmin(Z):
    finite = Z[np.isfinite(Z)]
    return float(np.floor(np.nanmin(finite) * 100) / 100) if len(finite) else 0.0


def draw_heatmap(ax, Z, title, cmap='inferno'):
    ax.set_facecolor(DARK_BG)
    vmin = panel_vmin(Z)
    pc = ax.pcolormesh(x_edges, y_edges, Z, cmap=cmap, vmin=vmin, vmax=1.0, shading='flat')
    mid_val = (vmin + 1.0) / 2
    for i in range(NUM_LAYERS):
        for j in range(len(unique_epochs)):
            val = Z[i, j]
            if not np.isnan(val):
                xc = (x_edges[j] + x_edges[j + 1]) / 2
                ax.text(xc, i, f'{val:.3f}', ha='center', va='center',
                        color='white' if val < mid_val else '#111', fontsize=5.0)
    cb = plt.colorbar(pc, ax=ax, pad=0.01, fraction=0.020)
    cb.ax.tick_params(colors='#ccc', labelsize=7)
    cb.set_label(f'norm_entropy [{vmin:.2f}–1.00]', color='#ccc', fontsize=6.5)
    ax.set_yticks(range(NUM_LAYERS))
    ax.set_yticklabels([f'L{l}' for l in range(NUM_LAYERS)], color='#ccc', fontsize=7.5)
    ax.set_xticks(unique_epochs)
    ax.set_xticklabels([str(e) for e in unique_epochs], color='#ccc', fontsize=7.5)
    ax.set_xlabel('Global epoch', color='#aaa', fontsize=8)
    ax.set_ylabel('Layer', color='#ccc', fontsize=8)
    ax.set_xlim(x_edges[0], x_edges[-1])
    for xv in phase_xlines:
        ax.axvline(xv, color='#444466', lw=1.0, ls=':')
    ax.set_title(title, color='white', fontsize=9)


draw_heatmap(fig.add_subplot(gs[0]), ffn_Z,
             'FFN_up — norm_entropy per layer  (avg over same-epoch checkpoints)')
draw_heatmap(fig.add_subplot(gs[1]), wo_Z,
             'Wo (output proj) — norm_entropy per layer', cmap='plasma')
draw_heatmap(fig.add_subplot(gs[2]), q_Z,
             'Attn Q — norm_entropy per layer  (mean over 4 heads)', cmap='magma')
draw_heatmap(fig.add_subplot(gs[3]), k_Z,
             'Attn K — norm_entropy per layer  (mean over 4 heads)', cmap='magma')
draw_heatmap(fig.add_subplot(gs[4]), v_Z,
             'Attn V — norm_entropy per layer  (mean over 4 heads)', cmap='magma')


# ── Panel 6: mean capacity vs epoch (all matrix types) ────────────────────
ax3 = fig.add_subplot(gs[5])
ax3.set_facecolor(DARK_BG)

LINES = [
    ('ffn_up', 'FFN_up', '#cc44ff', '-',   'o', 5.0),
    ('wo',     'Wo',     '#44ffaa', '--',  's', 4.5),
    ('q',      'Attn Q', '#ff5566', '-.',  '^', 4.0),
    ('k',      'Attn K', '#ffaa33', ':',   'D', 4.0),
    ('v',      'Attn V', '#33aaff', '-',   'v', 4.0),
]

for metric, mlabel, color, ls, marker, ms in LINES:
    xs, ys = [], []
    for _, phase, g_epoch, _, caps in results:
        xs.append(g_epoch)
        ys.append(float(np.nanmean(caps[metric])))

    # Individual scatter (all checkpoints, clustered at same epoch)
    ax3.scatter(xs, ys, color=color, s=ms ** 2 * 0.7, zorder=5, alpha=0.45)

    # Mean-per-epoch line
    ep_avg = defaultdict(list)
    for x, y in zip(xs, ys):
        ep_avg[x].append(y)
    ep_x = sorted(ep_avg)
    ep_y = [float(np.mean(ep_avg[e])) for e in ep_x]
    ax3.plot(ep_x, ep_y, color=color, lw=1.6, ls=ls, marker=marker, ms=ms,
             label=mlabel, zorder=4)

ax3.axhline(1.0, color='#333', lw=0.7, ls='--', alpha=0.6)
ax3.set_xlim(x_edges[0], x_edges[-1])
ax3.set_xticks(unique_epochs)
ax3.set_xticklabels([str(e) for e in unique_epochs], color='#ccc', fontsize=7.5)
ax3.set_xlabel('Global epoch', color='#aaa', fontsize=8)
ax3.set_ylabel('Mean norm_entropy  [0→1]', color='#ccc', fontsize=8)
all_ys = [float(np.nanmean(caps[m])) for _, _, _, _, caps in results for m in ('ffn_up', 'wo', 'q', 'k', 'v')]
ylo = max(0.0, float(np.nanmin(all_ys)) - 0.03)
ax3.set_ylim(ylo, 1.02)
ax3.set_title('Mean normalised spectral entropy per matrix type  '
              '(1 = uniform σ = full capacity use)',
              color='white', fontsize=9)
ax3.tick_params(colors='#888', labelsize=7)
ax3.grid(axis='both', color='#2a2a4a', lw=0.5, zorder=1)
for sp in ['top', 'right']:
    ax3.spines[sp].set_visible(False)
for sp in ['bottom', 'left']:
    ax3.spines[sp].set_color('#444')

for xv in phase_xlines:
    ax3.axvline(xv, color='#333355', lw=0.9, ls=':', zorder=2)

# Phase labels at top of line chart
prev_x = x_edges[0]
prev_phase = dominant_phase(unique_epochs[0])
for j, ep in enumerate(unique_epochs):
    cur_phase = dominant_phase(ep)
    if cur_phase != prev_phase or ep == unique_epochs[-1]:
        mid = (prev_x + (x_edges[j] if cur_phase != prev_phase else x_edges[j + 1])) / 2
        ax3.text(mid, 1.008, prev_phase, ha='center', va='bottom',
                 color='#666688', fontsize=6.5, transform=ax3.transData)
        prev_x = x_edges[j] if cur_phase != prev_phase else x_edges[j + 1]
        prev_phase = cur_phase

ax3.legend(fontsize=8, framealpha=0.3, labelcolor='white',
           facecolor='#222240', loc='lower right', ncol=5)

plt.savefig(os.path.join(OUT, 'ffn_entropy_capacity_all.png'),
            dpi=140, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print(f"\nSaved → {os.path.join(OUT, 'ffn_entropy_capacity_all.png')}")

# ── Summary ────────────────────────────────────────────────────────────────
print(f"\n{'Label':<14} {'ep':>3}  {'FFN_up':>7}  {'Wo':>7}  {'Q':>7}  {'K':>7}  {'V':>7}  {'loss':>7}")
print('-' * 68)
for label, phase, g_epoch, loss, caps in results:
    loss_s = f'{loss:.4f}' if loss else '  —   '
    print(f"{label:<14} {g_epoch:>3}  "
          f"{np.mean(caps['ffn_up']):7.4f}  {np.mean(caps['wo']):7.4f}  "
          f"{np.mean(caps['q']):7.4f}  {np.mean(caps['k']):7.4f}  "
          f"{np.mean(caps['v']):7.4f}  {loss_s}")
