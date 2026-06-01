
# --------------------------------------------------
# Tech Aarvam — Checkpoint Matrix Analysis
# Condition number + effective rank (50% singular value mass)
# across all Run 4 checkpoints.
# --------------------------------------------------
import os, sys, re, json
import torch
import numpy as np
import pandas as pd

# ── Extract-single mode ────────────────────────────────────────────────────
# Called by trainer every N epochs via subprocess:
#   python analyze_checkpoints.py --extract-single model.pth --epoch N --output-csv path.csv
if '--extract-single' in sys.argv:
    import argparse
    _p = argparse.ArgumentParser()
    _p.add_argument('--extract-single', metavar='MODEL_PTH')
    _p.add_argument('--epoch', type=int, required=True)
    _p.add_argument('--output-csv', required=True)
    _a = _p.parse_args()

    def _load_sd_single(path):
        ck = torch.load(path, map_location='cpu')
        sd = ck['model'] if (isinstance(ck, dict) and 'model' in ck) else ck
        out = {}
        for k, v in sd.items():
            k = k.replace('_orig_mod.', '')
            k = re.sub(r'^(mlp\.\d+)\.fc1\.(weight|bias)$', r'\1.0.\2', k)
            k = re.sub(r'^(mlp\.\d+)\.fc2\.(weight|bias)$', r'\1.2.\2', k)
            out[k] = v.float()
        return out

    def _svd_norm_entropy(W_2d):
        S = np.linalg.svd(W_2d.numpy(), compute_uv=False)
        S = S[S > 0]
        if len(S) <= 1:
            return 0.0
        p = S / S.sum()
        return float(-np.sum(p * np.log(p + 1e-12))) / np.log(len(S))

    _sd    = _load_sd_single(_a.extract_single)
    _epoch = _a.epoch
    _n_layers = 8
    _n_heads  = 4
    _fused    = any('qkv' in k for k in _sd)

    row = {'epoch': _epoch}
    for L in range(_n_layers):
        row[f'ffn_up_L{L}']  = _svd_norm_entropy(_sd[f'mlp.{L}.0.weight'])
        row[f'ffn_dn_L{L}']  = _svd_norm_entropy(_sd[f'mlp.{L}.2.weight'])
        row[f'wo_L{L}']      = _svd_norm_entropy(_sd[f'Wo.{L}'])
        if _fused:
            _qkv  = _sd[f'attentionHeads.{L}.qkv']
            _idim = _qkv.shape[0]
            _hd   = _idim // _n_heads
            for _h in range(_n_heads):
                row[f'q_L{L}_h{_h}'] = _svd_norm_entropy(_qkv[:, _h*_hd:(_h+1)*_hd])
                row[f'k_L{L}_h{_h}'] = _svd_norm_entropy(_qkv[:, _idim+_h*_hd:_idim+(_h+1)*_hd])
                row[f'v_L{L}_h{_h}'] = _svd_norm_entropy(_qkv[:, 2*_idim+_h*_hd:2*_idim+(_h+1)*_hd])

    _out = _a.output_csv
    _write_header = not os.path.exists(_out)
    with open(_out, 'a') as _f:
        if _write_header:
            _f.write(','.join(row.keys()) + '\n')
        _f.write(','.join(str(v) for v in row.values()) + '\n')
    print(f"Entropy row appended → {_out}  (epoch={_epoch})", flush=True)
    sys.exit(0)
# ── End extract-single mode ────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, 'output')
os.makedirs(OUT, exist_ok=True)

CHECKPOINTS = [
    # (label,              phase,    epoch, loss,   rel_path)
    ('w64_ep7_m1',         'w64',    7,     6.0920, 'btm_r2_backups/model_step1_128_m1.pth'),
    ('w128_ep1_m1',        'w128',   1,     5.9181, 'btm_r2_backups/model_w128_m1.pth'),
    ('btm_merged',         'merge',  0,     None,   'btm_r2_backups/btm_w128_merged.pth'),
    ('w128_cont_ep12',     'w128c',  12,    5.7632, 'btm_r2_backups/btm_w128_cont.pth'),
    ('w256_ep13',          'w256',   13,    5.6670, 'btm_r2_backups/btm_w256_cont_ep13.pth'),
    ('d512_ep15_unfused',  'd512u',  15,    5.7338, 'btm_r2_backups/d512_20260522/btm_d512_cont_20260522_203334/model.pth'),
    ('d512_ep15_fused',    'd512f',  15,    5.7338, 'btm_r2_backups/d512_20260522/btm_d512_cont_20260522_203334/model_fused.pth'),
    ('d512_ep16_sdpa',     'd512',   16,    5.1473, 'runs/btm_d512_cont_local_20260527_070826/model.pth'),
    ('d512_ep18_sdpa',     'd512',   18,    4.9953, 'runs/btm_d512_cont_local_20260527_203054/model.pth'),
    ('d512_ep20_sdpa',     'd512',   20,    4.9336, 'btm_r2_backups/d512_ep20_sdpa.pth'),
    ('d512_ep21_sdpa',     'd512',   21,    4.9207, 'btm_r2_backups/d512_ep21_sdpa.pth'),
]

MANIFEST = os.path.join(HERE, 'analysis_checkpoints.json')

NUM_LAYERS = 8
NUM_HEADS  = 4


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


def load_sd(path):
    ck = torch.load(path, map_location='cpu')
    if isinstance(ck, dict) and 'model' in ck:
        sd = ck['model']
    else:
        sd = ck
    out = {}
    for k, v in sd.items():
        k = k.replace('_orig_mod.', '')
        k = re.sub(r'^(mlp\.\d+)\.fc1\.(weight|bias)$', r'\1.0.\2', k)
        k = re.sub(r'^(mlp\.\d+)\.fc2\.(weight|bias)$', r'\1.2.\2', k)
        out[k] = v.float()
    return out


def svd_metrics(W_2d):
    """Return (cond, eff_rank_50, spectral_entropy, norm_entropy, S).

    spectral_entropy: H = -Σ p_i log(p_i),  p_i = σ_i / Σσ  (nats)
    norm_entropy:     H / log(rank)  ∈ [0, 1]  — 1 = all σ equal, 0 = rank-1
    """
    W = W_2d.numpy()
    S = np.linalg.svd(W, compute_uv=False)
    S = S[S > 0]
    if len(S) == 0:
        return np.inf, 0, 0.0, 0.0, np.array([])
    cond   = float(S[0] / S[-1])
    total  = S.sum()
    cumsum = np.cumsum(S)
    eff50  = int(np.searchsorted(cumsum, 0.5 * total)) + 1
    p      = S / total
    H      = float(-np.sum(p * np.log(p + 1e-12)))
    H_norm = H / np.log(len(S)) if len(S) > 1 else 0.0
    return cond, eff50, H, H_norm, S


def make_rec(label, phase, epoch, loss, matrix, layer, head, M, cond, eff50, H, H_norm, S):
    return dict(label=label, phase=phase, epoch=epoch, loss=loss,
                matrix=matrix, layer=layer, head=head,
                shape=str(tuple(M.shape)),
                cond=cond, eff_rank_50=eff50,
                spectral_entropy=H, norm_entropy=H_norm,
                sv_max=float(S[0])    if len(S) else 0,
                sv_min=float(S[-1])   if len(S) else 0,
                sv_mean=float(S.mean()) if len(S) else 0)


def extract_records(label, phase, epoch, loss, sd):
    """Yield one dict per matrix."""
    is_fused   = any('qkv' in k for k in sd)
    has_scale  = 'upscale.weight' in sd

    records = []

    for L in range(NUM_LAYERS):
        if is_fused:
            qkv       = sd[f'attentionHeads.{L}.qkv']   # [inner, 3*inner]
            inner_dim = qkv.shape[0]
            head_dim  = inner_dim // NUM_HEADS
            Q = qkv[:, :inner_dim]
            K = qkv[:, inner_dim:2*inner_dim]
            V = qkv[:, 2*inner_dim:]
            for h in range(NUM_HEADS):
                for mat_name, M in [('Q', Q[:, h*head_dim:(h+1)*head_dim]),
                                    ('K', K[:, h*head_dim:(h+1)*head_dim]),
                                    ('V', V[:, h*head_dim:(h+1)*head_dim])]:
                    cond, eff50, H, H_norm, S = svd_metrics(M)
                    records.append(make_rec(label, phase, epoch, loss,
                                            mat_name, L, h, M, cond, eff50, H, H_norm, S))
        else:
            for mat_name, key in [('Q', 'query'), ('K', 'keys'), ('V', 'value')]:
                W_all = sd[f'attentionHeads.{L}.{key}']  # [heads, in, head_dim]
                for h in range(W_all.shape[0]):
                    M = W_all[h]
                    cond, eff50, H, H_norm, S = svd_metrics(M)
                    records.append(make_rec(label, phase, epoch, loss,
                                            mat_name, L, h, M, cond, eff50, H, H_norm, S))

        # Wo (output projection) — no head dimension
        M = sd[f'Wo.{L}']
        cond, eff50, H, H_norm, S = svd_metrics(M)
        records.append(make_rec(label, phase, epoch, loss,
                                'Wo', L, -1, M, cond, eff50, H, H_norm, S))

        # FFN
        for mat_name, key in [('FFN_up', 'mlp.{L}.0.weight'), ('FFN_down', 'mlp.{L}.2.weight')]:
            M = sd[key.format(L=L)]
            cond, eff50, H, H_norm, S = svd_metrics(M)
            records.append(make_rec(label, phase, epoch, loss,
                                    mat_name, L, -1, M, cond, eff50, H, H_norm, S))

    # upscale / downscale if present
    if has_scale:
        for mat_name, key in [('upscale', 'upscale.weight'), ('downscale', 'downscale.weight')]:
            M = sd[key]
            cond, eff50, H, H_norm, S = svd_metrics(M)
            records.append(make_rec(label, phase, epoch, loss,
                                    mat_name, -1, -1, M, cond, eff50, H, H_norm, S))
    return records


# ── Run analysis ──────────────────────────────────────────────────────────
all_records = []
loaded = []

for label, phase, epoch, loss, rel_path in CHECKPOINTS + load_manifest_checkpoints():
    path = os.path.join(HERE, rel_path)
    if not os.path.exists(path):
        print(f"  SKIP (not found): {rel_path}")
        continue
    print(f"  Loading {label} …", flush=True)
    sd = load_sd(path)
    recs = extract_records(label, phase, epoch, loss, sd)
    all_records.extend(recs)
    loaded.append((label, phase, epoch, loss))
    print(f"    {len(recs)} matrices analysed")

df = pd.DataFrame(all_records)
csv_path = os.path.join(HERE, 'checkpoint_matrix_analysis.csv')
df.to_csv(csv_path, index=False)
print(f"\nCSV saved → {csv_path}  ({len(df)} rows)")


# ── 3D Plots ──────────────────────────────────────────────────────────────
# Select every 5th checkpoint (by index in loaded list)
plot_indices = list(range(0, len(loaded), max(1, len(loaded)//5)))
if plot_indices[-1] != len(loaded)-1:
    plot_indices.append(len(loaded)-1)
plot_labels = [loaded[i][0] for i in plot_indices]
print(f"\n3D plots for: {plot_labels}")

DARK_BG = '#1a1a2e'

def make_3d_surface(metric, mat_filter, title_prefix, fname, clip_pct=95):
    """metric = 'cond' or 'eff_rank_50'. mat_filter = list of matrix names.
    clip_pct: percentile cap applied to Z before plotting (avoids scale collapse from outliers)."""
    n = len(plot_labels)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig = plt.figure(figsize=(6*ncols, 5*nrows), facecolor=DARK_BG)

    # Compute per-figure Z arrays first so we can set a common clip threshold
    all_Z = []
    grids = []
    for idx, lbl in enumerate(plot_labels):
        sub = df[(df['label'] == lbl) & (df['matrix'].isin(mat_filter)) & (df['head'] >= 0)]
        if sub.empty:
            sub = df[(df['label'] == lbl) & (df['matrix'].isin(mat_filter))]

        if sub.empty or 'head' not in sub.columns:
            grids.append(None)
            continue

        if sub['head'].max() >= 0:
            layers = sorted(sub['layer'].unique())
            heads  = sorted(sub['head'].unique())
            Z = np.zeros((len(heads), len(layers)))
            for i, h in enumerate(heads):
                for j, l in enumerate(layers):
                    vals = sub[(sub['head']==h) & (sub['layer']==l)][metric].values
                    Z[i, j] = np.median(vals) if len(vals) else 0
            X, Y = np.meshgrid(layers, heads)
        else:
            layers = sorted(sub['layer'].unique())
            Z = np.zeros((1, len(layers)))
            for j, l in enumerate(layers):
                vals = sub[sub['layer']==l][metric].values
                Z[0, j] = np.median(vals) if len(vals) else 0
            # plot_surface needs at least 2 rows. Duplicate layer-only matrices
            # into a thin strip so Wo/FFN surfaces render instead of blank axes.
            Z = np.vstack([Z, Z])
            X, Y = np.meshgrid(layers, [-0.15, 0.15])

        if metric == 'cond':
            pos_finite = np.isfinite(Z) & (Z > 0)
            pos_inf = np.isinf(Z) & (Z > 0)
            Z_log = np.full_like(Z, np.nan, dtype=float)
            if np.any(pos_finite):
                Z_log[pos_finite] = np.log10(Z[pos_finite])
                inf_level = float(np.nanmax(Z_log[pos_finite]) + 1.0)
            else:
                inf_level = 12.0
            Z_log[pos_inf] = inf_level
            Z = Z_log

        grids.append((X, Y, Z, lbl, sub))
        all_Z.append(Z)

    # Shared clip threshold across all subplots in this figure
    if all_Z:
        all_vals = np.concatenate([z.ravel() for z in all_Z])
        finite_vals = all_vals[np.isfinite(all_vals)]
        vmax = float(np.percentile(finite_vals, clip_pct)) if len(finite_vals) else 1.0
    else:
        vmax = 1.0

    for idx, lbl in enumerate(plot_labels):
        ax = fig.add_subplot(nrows, ncols, idx+1, projection='3d', facecolor=DARK_BG)
        if grids[idx] is None:
            ax.set_title(lbl, color='white', fontsize=8)
            continue

        X, Y, Z, lbl, sub = grids[idx]
        finite_z = Z[np.isfinite(Z)]
        zmin = 0 if metric != 'cond' else (max(0, float(np.nanmin(finite_z))) if len(finite_z) else 0)
        clipped = np.clip(Z, zmin, vmax)
        n_clipped = int(np.sum(Z > vmax))

        epoch_str = f"ep{loaded[plot_indices[idx]][2]}" if loaded[plot_indices[idx]][2] else 'merge'
        loss_val  = loaded[plot_indices[idx]][3]
        loss_str  = f" loss={loss_val:.4f}" if loss_val else ''
        clip_note = f" [clip>{vmax:.0f}:{n_clipped}]" if n_clipped else ''

        ax.set_facecolor(DARK_BG)
        ax.plot_surface(X, Y, clipped, cmap='plasma', edgecolor='none', alpha=0.9,
                        vmin=zmin, vmax=vmax)
        ax.set_title(f"{lbl}\n{epoch_str}{loss_str}{clip_note}", color='white', fontsize=7)
        ax.set_xlabel('Layer', color='#aaa', fontsize=7, labelpad=2)
        ax.set_ylabel('Head',  color='#aaa', fontsize=7, labelpad=2)
        zlabel = 'log10 Cond #' if metric == 'cond' else 'Eff rank 50%'
        ax.set_zlabel(zlabel, color='#aaa', fontsize=7, labelpad=2)
        ax.tick_params(colors='#888', labelsize=6)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.set_edgecolor('#333')

    clip_label = f'p{clip_pct} clip'
    fig.suptitle(f'{title_prefix} — {metric} ({clip_label})', color='white', fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, fname), dpi=130, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"  Saved {fname}")

# Attention matrices (Q, K, V)
make_3d_surface('cond',       ['Q','K','V'], 'Attention QKV',  '3d_attn_cond.png')
make_3d_surface('eff_rank_50',['Q','K','V'], 'Attention QKV',  '3d_attn_effrank.png')

# Wo (output projection)
make_3d_surface('cond',       ['Wo'],        'Output proj Wo', '3d_wo_cond.png')
make_3d_surface('eff_rank_50',['Wo'],        'Output proj Wo', '3d_wo_effrank.png')

# FFN
make_3d_surface('cond',       ['FFN_up','FFN_down'], 'FFN',   '3d_ffn_cond.png')
make_3d_surface('eff_rank_50',['FFN_up','FFN_down'], 'FFN',   '3d_ffn_effrank.png')

# ── Evolution heatmap: layer × checkpoint, Z = median cond (avg over heads/matrices) ──
for mat_group, mats, fname_prefix in [
    ('Attention QKV', ['Q','K','V'], 'evo_attn'),
    ('FFN',           ['FFN_up','FFN_down'], 'evo_ffn'),
    ('Wo',            ['Wo'], 'evo_wo'),
]:
    for metric in ['cond', 'eff_rank_50']:
        sub = df[df['matrix'].isin(mats) & (df['layer'] >= 0)]
        labels_ord = [l[0] for l in loaded]
        layers = sorted(sub['layer'].unique())
        Z = np.zeros((len(layers), len(labels_ord)))
        for j, lbl in enumerate(labels_ord):
            for i, lay in enumerate(layers):
                vals = sub[(sub['label']==lbl) & (sub['layer']==lay)][metric].values
                Z[i, j] = np.nanmedian(vals) if len(vals) else np.nan

        fig, ax = plt.subplots(figsize=(max(10, len(labels_ord)*1.4), 4), facecolor=DARK_BG)
        ax.set_facecolor(DARK_BG)
        im = ax.imshow(Z, aspect='auto', cmap='plasma', interpolation='nearest')
        ax.set_yticks(range(len(layers))); ax.set_yticklabels([f'L{l}' for l in layers], color='#ccc', fontsize=8)
        ax.set_xticks(range(len(labels_ord))); ax.set_xticklabels(labels_ord, rotation=35, ha='right', color='#ccc', fontsize=7)
        ax.set_ylabel('Layer', color='#ccc'); ax.set_xlabel('Checkpoint', color='#ccc')
        cb = plt.colorbar(im, ax=ax)
        cb.ax.tick_params(colors='#ccc', labelsize=7)
        metric_lbl = 'Condition number' if metric == 'cond' else 'Eff rank 50%'
        ax.set_title(f'{mat_group} — {metric_lbl} evolution across checkpoints', color='white', fontsize=10)
        plt.tight_layout()
        fname = f'{fname_prefix}_{metric}.png'
        plt.savefig(os.path.join(OUT, fname), dpi=130, bbox_inches='tight', facecolor=DARK_BG)
        plt.close()
        print(f"  Saved {fname}")

# ── FFN spectral entropy — generalized information carrying capacity ───────
# Per checkpoint: sum of norm_entropy across all FFN_up layers (8 layers).
# norm_entropy ∈ [0,1]: 1 = all singular values equal (max capacity usage).
# Total capacity score = Σ_{l=0}^{7} norm_entropy(FFN_up_l)  ∈ [0, 8].

labels_ord = [l[0] for l in loaded]
ffn_up_sub = df[(df['matrix'] == 'FFN_up') & (df['layer'] >= 0)]

# Per checkpoint per layer: norm_entropy of FFN_up
layers = sorted(ffn_up_sub['layer'].unique())
Z_ent  = np.zeros((len(layers), len(labels_ord)))
for j, lbl in enumerate(labels_ord):
    for i, lay in enumerate(layers):
        vals = ffn_up_sub[(ffn_up_sub['label'] == lbl) &
                          (ffn_up_sub['layer'] == lay)]['norm_entropy'].values
        Z_ent[i, j] = float(np.nanmedian(vals)) if len(vals) else np.nan

# Heatmap: layer × checkpoint
fig, axes = plt.subplots(2, 1, figsize=(max(10, len(labels_ord)*1.4), 7), facecolor=DARK_BG)

ax = axes[0]
ax.set_facecolor(DARK_BG)
im = ax.imshow(Z_ent, aspect='auto', cmap='inferno', interpolation='nearest',
               vmin=0, vmax=1)
ax.set_yticks(range(len(layers)))
ax.set_yticklabels([f'L{l}' for l in layers], color='#ccc', fontsize=8)
ax.set_xticks(range(len(labels_ord)))
ax.set_xticklabels(labels_ord, rotation=35, ha='right', color='#ccc', fontsize=7)
ax.set_ylabel('Layer', color='#ccc')
ax.set_xlabel('Checkpoint', color='#ccc')
cb = plt.colorbar(im, ax=ax)
cb.ax.tick_params(colors='#ccc', labelsize=7)
cb.set_label('Normalised entropy [0→1]', color='#ccc', fontsize=7)
ax.set_title('FFN_up normalised spectral entropy per layer  '
             '(1 = uniform σ = full capacity, 0 = rank-1)',
             color='white', fontsize=9)

# Total capacity score per checkpoint (sum over layers)
total_cap = Z_ent.sum(axis=0)   # shape: (n_checkpoints,)
ax2 = axes[1]
ax2.set_facecolor(DARK_BG)
ax2.bar(range(len(labels_ord)), total_cap, color='#cc44ff', alpha=0.8)
ax2.set_xticks(range(len(labels_ord)))
ax2.set_xticklabels(labels_ord, rotation=35, ha='right', color='#ccc', fontsize=7)
ax2.set_ylabel('Σ norm_entropy  (max = 8)', color='#ccc', fontsize=8)
ax2.set_ylim(0, len(layers))
ax2.axhline(len(layers), color='#555', lw=0.8, ls='--')
ax2.set_title('Total FFN capacity score  =  Σ norm_entropy(FFN_up_l)  across 8 layers',
              color='white', fontsize=9)
ax2.tick_params(colors='#888', labelsize=7)
for sp in ['top', 'right']:
    ax2.spines[sp].set_visible(False)
for sp in ['bottom', 'left']:
    ax2.spines[sp].set_color('#444')

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'ffn_entropy_capacity.png'), dpi=130,
            bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("  Saved ffn_entropy_capacity.png")

# Print summary table
print("\nFFN capacity score (Σ norm_entropy FFN_up, 8 layers, max=8):")
for j, lbl in enumerate(labels_ord):
    loss_val = loaded[j][3]
    loss_str = f"  loss={loss_val:.4f}" if loss_val else ''
    print(f"  {lbl:<20s}  {total_cap[j]:.4f}{loss_str}")

print("\nDone.")
