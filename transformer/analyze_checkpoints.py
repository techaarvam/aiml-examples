
# --------------------------------------------------
# Tech Aarvam — Checkpoint Matrix Analysis
# Condition number + effective rank (50% singular value mass)
# across all Run 4 checkpoints.
# --------------------------------------------------
import os, sys
import torch
import numpy as np
import pandas as pd
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
]

NUM_LAYERS = 8
NUM_HEADS  = 4


def load_sd(path):
    ck = torch.load(path, map_location='cpu')
    if isinstance(ck, dict) and 'model' in ck:
        sd = ck['model']
    else:
        sd = ck
    return {k.replace('_orig_mod.', ''): v.float() for k, v in sd.items()}


def svd_metrics(W_2d):
    """Return (condition_number, eff_rank_50pct, singular_values_array)."""
    W = W_2d.numpy()
    S = np.linalg.svd(W, compute_uv=False)
    S = S[S > 0]
    if len(S) == 0:
        return np.inf, 0, np.array([])
    cond   = float(S[0] / S[-1])
    total  = S.sum()
    cumsum = np.cumsum(S)
    eff50  = int(np.searchsorted(cumsum, 0.5 * total)) + 1
    return cond, eff50, S


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
                    cond, eff50, S = svd_metrics(M)
                    records.append(dict(label=label, phase=phase, epoch=epoch, loss=loss,
                                        matrix=mat_name, layer=L, head=h,
                                        shape=str(tuple(M.shape)),
                                        cond=cond, eff_rank_50=eff50,
                                        sv_max=float(S[0]) if len(S) else 0,
                                        sv_min=float(S[-1]) if len(S) else 0,
                                        sv_mean=float(S.mean()) if len(S) else 0))
        else:
            for mat_name, key in [('Q', 'query'), ('K', 'keys'), ('V', 'value')]:
                W_all = sd[f'attentionHeads.{L}.{key}']  # [heads, in, head_dim]
                for h in range(W_all.shape[0]):
                    M = W_all[h]
                    cond, eff50, S = svd_metrics(M)
                    records.append(dict(label=label, phase=phase, epoch=epoch, loss=loss,
                                        matrix=mat_name, layer=L, head=h,
                                        shape=str(tuple(M.shape)),
                                        cond=cond, eff_rank_50=eff50,
                                        sv_max=float(S[0]) if len(S) else 0,
                                        sv_min=float(S[-1]) if len(S) else 0,
                                        sv_mean=float(S.mean()) if len(S) else 0))

        # Wo (output projection) — no head dimension
        M = sd[f'Wo.{L}']
        cond, eff50, S = svd_metrics(M)
        records.append(dict(label=label, phase=phase, epoch=epoch, loss=loss,
                            matrix='Wo', layer=L, head=-1,
                            shape=str(tuple(M.shape)),
                            cond=cond, eff_rank_50=eff50,
                            sv_max=float(S[0]) if len(S) else 0,
                            sv_min=float(S[-1]) if len(S) else 0,
                            sv_mean=float(S.mean()) if len(S) else 0))

        # FFN
        for mat_name, key in [('FFN_up', 'mlp.{L}.0.weight'), ('FFN_down', 'mlp.{L}.2.weight')]:
            M = sd[key.format(L=L)]
            cond, eff50, S = svd_metrics(M)
            records.append(dict(label=label, phase=phase, epoch=epoch, loss=loss,
                                matrix=mat_name, layer=L, head=-1,
                                shape=str(tuple(M.shape)),
                                cond=cond, eff_rank_50=eff50,
                                sv_max=float(S[0]) if len(S) else 0,
                                sv_min=float(S[-1]) if len(S) else 0,
                                sv_mean=float(S.mean()) if len(S) else 0))

    # upscale / downscale if present
    if has_scale:
        for mat_name, key in [('upscale', 'upscale.weight'), ('downscale', 'downscale.weight')]:
            M = sd[key]
            cond, eff50, S = svd_metrics(M)
            records.append(dict(label=label, phase=phase, epoch=epoch, loss=loss,
                                matrix=mat_name, layer=-1, head=-1,
                                shape=str(tuple(M.shape)),
                                cond=cond, eff_rank_50=eff50,
                                sv_max=float(S[0]) if len(S) else 0,
                                sv_min=float(S[-1]) if len(S) else 0,
                                sv_mean=float(S.mean()) if len(S) else 0))
    return records


# ── Run analysis ──────────────────────────────────────────────────────────
all_records = []
loaded = []

for label, phase, epoch, loss, rel_path in CHECKPOINTS:
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

def make_3d_surface(metric, mat_filter, title_prefix, fname):
    """metric = 'cond' or 'eff_rank_50'. mat_filter = list of matrix names."""
    n = len(plot_labels)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig = plt.figure(figsize=(6*ncols, 5*nrows), facecolor=DARK_BG)

    for idx, lbl in enumerate(plot_labels):
        sub = df[(df['label'] == lbl) & (df['matrix'].isin(mat_filter)) & (df['head'] >= 0)]
        if sub.empty:
            # For non-head matrices (Wo, FFN)
            sub = df[(df['label'] == lbl) & (df['matrix'].isin(mat_filter))]

        ax = fig.add_subplot(nrows, ncols, idx+1, projection='3d', facecolor=DARK_BG)

        if sub.empty or 'head' not in sub.columns:
            ax.set_title(lbl, color='white', fontsize=8)
            continue

        if sub['head'].max() >= 0:
            # head × layer grid
            layers = sorted(sub['layer'].unique())
            heads  = sorted(sub['head'].unique())
            Z = np.zeros((len(heads), len(layers)))
            for i, h in enumerate(heads):
                for j, l in enumerate(layers):
                    vals = sub[(sub['head']==h) & (sub['layer']==l)][metric].values
                    Z[i, j] = np.median(vals) if len(vals) else 0
            X, Y = np.meshgrid(layers, heads)
        else:
            # layer only (Wo, FFN)
            layers = sorted(sub['layer'].unique())
            Z = np.zeros((1, len(layers)))
            for j, l in enumerate(layers):
                vals = sub[sub['layer']==l][metric].values
                Z[0, j] = np.median(vals) if len(vals) else 0
            X, Y = np.meshgrid(layers, [0])

        epoch_str = f"ep{loaded[plot_indices[idx]][2]}" if loaded[plot_indices[idx]][2] else 'merge'
        loss_val  = loaded[plot_indices[idx]][3]
        loss_str  = f" loss={loss_val:.4f}" if loss_val else ''
        ax.set_facecolor(DARK_BG)
        surf = ax.plot_surface(X, Y, Z, cmap='plasma', edgecolor='none', alpha=0.9)
        ax.set_title(f"{lbl}\n{epoch_str}{loss_str}", color='white', fontsize=7)
        ax.set_xlabel('Layer', color='#aaa', fontsize=7, labelpad=2)
        ax.set_ylabel('Head',  color='#aaa', fontsize=7, labelpad=2)
        zlabel = 'Cond #' if metric == 'cond' else 'Eff rank 50%'
        ax.set_zlabel(zlabel, color='#aaa', fontsize=7, labelpad=2)
        ax.tick_params(colors='#888', labelsize=6)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.set_edgecolor('#333')

    fig.suptitle(f'{title_prefix} — {metric}', color='white', fontsize=11, y=1.01)
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

print("\nDone.")
