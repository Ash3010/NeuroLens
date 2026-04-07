"""
advanced_viz.py
Three new visualizations aligned to the proposal:
  1. make_pca_tsne()     — PCA + t-SNE dimensionality reduction of activations
  2. make_isosurface()   — Activation threshold / isosurface map
  3. make_variance_map() — Per-channel variance + full stats (mean, var, sparsity)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


# ── Shared style helper ────────────────────────────────────────────────────
def _dark_fig(nrows=1, ncols=1, figsize=(12, 5)):
    plt.style.use('dark_background')
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.patch.set_facecolor('#0a0a0f')
    return fig, axes


def _style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor('#12121a')
    ax.spines[:].set_color('#2a2a3a')
    ax.tick_params(colors='#6b6b80', labelsize=7)
    if title:  ax.set_title(title,  color='#e8e8f0', fontsize=9,  pad=6, fontfamily='monospace')
    if xlabel: ax.set_xlabel(xlabel, color='#6b6b80', fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, color='#6b6b80', fontsize=8)


# ══════════════════════════════════════════════════════════════════════════
# 1.  PCA + t-SNE
# ══════════════════════════════════════════════════════════════════════════
def make_pca_tsne(fmap_tensor, layer_label='layer', fast_mode=True, max_channels=256):
    """
    Project each channel's activation map into a point in 2D space using
    PCA and t-SNE. Each dot = one channel filter.

    Fast mode:
      - subsample channels if there are too many
      - reduce to 30 dims using PCA before t-SNE
      - lower t-SNE iterations for speed
    """
    fmap = fmap_tensor[0].numpy()   # [C, H, W]
    C, H, W = fmap.shape
    X = fmap.reshape(C, H * W)

    # Optional channel subsampling for speed
    if fast_mode and C > max_channels:
        idx = np.linspace(0, C - 1, max_channels).astype(int)
        X = X[idx]
        fmap = fmap[idx]
        C = X.shape[0]

    # Standardize
    X_scaled = StandardScaler().fit_transform(X)

    # PCA for 2D plotting
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(X_scaled)
    pca_var = pca.explained_variance_ratio_ * 100

    # PCA pre-reduction before t-SNE
    if X_scaled.shape[1] > 30:
        X_tsne_in = PCA(n_components=30).fit_transform(X_scaled)
    else:
        X_tsne_in = X_scaled

    perplexity = min(30, max(5, C // 5))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init='pca',
        learning_rate='auto',
        random_state=42,
        max_iter=300 if fast_mode else 700,
        verbose=0
    )
    tsne_coords = tsne.fit_transform(X_tsne_in)

    mean_acts = fmap.mean(axis=(1, 2))
    norm_acts = (mean_acts - mean_acts.min()) / (mean_acts.max() - mean_acts.min() + 1e-8)

    fig, (ax1, ax2) = _dark_fig(1, 2, figsize=(14, 5))

    # PCA plot
    sc1 = ax1.scatter(
        pca_coords[:, 0], pca_coords[:, 1],
        c=norm_acts, cmap='viridis', s=40, alpha=0.85, edgecolors='none'
    )
    _style_ax(
        ax1,
        title=f'PCA — {layer_label} ({C} channels)',
        xlabel=f'PC1 ({pca_var[0]:.1f}% var)',
        ylabel=f'PC2 ({pca_var[1]:.1f}% var)'
    )
    cb1 = fig.colorbar(sc1, ax=ax1, fraction=0.03, pad=0.02)
    cb1.ax.tick_params(colors='#6b6b80', labelsize=7)
    cb1.set_label('Mean activation', color='#6b6b80', fontsize=7)

    # t-SNE plot
    sc2 = ax2.scatter(
        tsne_coords[:, 0], tsne_coords[:, 1],
        c=norm_acts, cmap='plasma', s=40, alpha=0.85, edgecolors='none'
    )
    _style_ax(
        ax2,
        title=f't-SNE — {layer_label} (fast={fast_mode}, perplexity={perplexity})',
        xlabel='t-SNE dim 1',
        ylabel='t-SNE dim 2'
    )
    cb2 = fig.colorbar(sc2, ax=ax2, fraction=0.03, pad=0.02)
    cb2.ax.tick_params(colors='#6b6b80', labelsize=7)
    cb2.set_label('Mean activation', color='#6b6b80', fontsize=7)

    fig.suptitle(
        f'Dimensionality Reduction of Channel Activations — {layer_label}',
        fontsize=11, color='#e8e8f0', fontfamily='monospace', y=1.01
    )
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════
# 2.  ISOSURFACE / THRESHOLD MAP
# ══════════════════════════════════════════════════════════════════════════
def make_isosurface(fmap_tensor, layer_label='layer', thresholds=None):
    """
    Treat the mean activation map as a scalar field and apply multiple
    threshold levels (isosurfaces in 2D = contour lines).
    Shows which spatial regions exceed each activation level.

    Args:
        fmap_tensor : torch.Tensor  shape [1, C, H, W]
        layer_label : str
        thresholds  : list of float (0–1), defaults to [0.3, 0.5, 0.7, 0.9]
    Returns:
        matplotlib Figure
    """
    if thresholds is None:
        thresholds = [0.3, 0.5, 0.7, 0.9]

    fmap  = fmap_tensor[0].numpy()    # [C, H, W]
    # Mean across channels → single scalar field [H, W]
    scalar_field = fmap.mean(axis=0)
    # Normalise to [0, 1]
    sf_norm = (scalar_field - scalar_field.min()) / \
              (scalar_field.max() - scalar_field.min() + 1e-8)

    n_thresh = len(thresholds)
    fig, axes = _dark_fig(1, n_thresh + 1, figsize=(4 * (n_thresh + 1), 4))
    if n_thresh == 0:
        axes = [axes]

    # Original scalar field
    ax0 = axes[0]
    im = ax0.imshow(sf_norm, cmap='hot', interpolation='bilinear')
    ax0.contour(sf_norm, levels=thresholds, colors=['#60a5fa', '#34d399', '#f472b6', '#fbbf24'],
                linewidths=0.8, alpha=0.9)
    _style_ax(ax0, title=f'Scalar field + all contours\n{layer_label}')
    fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04).ax.tick_params(colors='#6b6b80', labelsize=6)

    # One panel per threshold
    colors_thresh = ['#60a5fa', '#34d399', '#f472b6', '#fbbf24']
    for j, (ax, thr) in enumerate(zip(axes[1:], thresholds)):
        mask = sf_norm >= thr
        # Show the scalar field dimmed, highlight region above threshold
        ax.imshow(sf_norm, cmap='hot', alpha=0.35, interpolation='bilinear')
        ax.imshow(np.where(mask, sf_norm, np.nan),
                  cmap='hot', vmin=0, vmax=1, interpolation='bilinear')
        ax.contour(sf_norm, levels=[thr],
                   colors=[colors_thresh[j % len(colors_thresh)]],
                   linewidths=1.2)
        pct_active = mask.mean() * 100
        _style_ax(ax, title=f'Threshold ≥ {thr:.0%}\n{pct_active:.1f}% active')

    fig.suptitle(
        f'Isosurface / Threshold Visualization — {layer_label}',
        fontsize=11, color='#e8e8f0', fontfamily='monospace', y=1.02)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════
# 3.  VARIANCE MAP + FULL STATS
# ══════════════════════════════════════════════════════════════════════════
def make_variance_map(fmap_tensor, layer_label='layer'):
    """
    Proposal slide 6 says: aggregate mean, variance, and sparsity per channel.
    This adds the variance component that was missing.

    Shows:
      Row 1: Mean activation map, Variance map, Sparsity map (spatial)
      Row 2: Per-channel bar charts for mean, variance, sparsity

    Args:
        fmap_tensor : torch.Tensor  shape [1, C, H, W]
        layer_label : str
    Returns:
        matplotlib Figure
    """
    fmap = fmap_tensor[0].numpy()    # [C, H, W]
    C    = fmap.shape[0]
    n    = min(32, C)                # show up to 32 channels in bar charts

    # ── Spatial maps (averaged over channels) ─────────────────────────────
    mean_map     = fmap.mean(axis=0)
    var_map      = fmap.var(axis=0)
    sparsity_map = (fmap < 0.01).mean(axis=0)

    def _norm(m):
        return (m - m.min()) / (m.max() - m.min() + 1e-8)

    # ── Per-channel statistics ─────────────────────────────────────────────
    ch_means    = fmap.mean(axis=(1, 2))[:n]
    ch_vars     = fmap.var(axis=(1, 2))[:n]
    ch_sparsity = (fmap < 0.01).mean(axis=(1, 2))[:n] * 100

    # ── Figure layout: 2 rows × 3 cols ────────────────────────────────────
    fig = plt.figure(figsize=(16, 8))
    fig.patch.set_facecolor('#0a0a0f')

    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.3)
    ax_mean_map  = fig.add_subplot(gs[0, 0])
    ax_var_map   = fig.add_subplot(gs[0, 1])
    ax_spar_map  = fig.add_subplot(gs[0, 2])
    ax_mean_bar  = fig.add_subplot(gs[1, 0])
    ax_var_bar   = fig.add_subplot(gs[1, 1])
    ax_spar_bar  = fig.add_subplot(gs[1, 2])

    # ── Row 0: spatial heatmaps ───────────────────────────────────────────
    for ax, data, cmap, title in [
        (ax_mean_map,  _norm(mean_map),     'viridis', 'Mean activation map'),
        (ax_var_map,   _norm(var_map),      'plasma',  'Variance map'),
        (ax_spar_map,  sparsity_map,        'RdYlGn_r','Sparsity map (% near-zero)'),
    ]:
        im = ax.imshow(data, cmap=cmap, interpolation='bilinear')
        _style_ax(ax, title=title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(colors='#6b6b80', labelsize=6)

    # ── Row 1: per-channel bar charts ─────────────────────────────────────
    xs = np.arange(n)

    # Mean
    c1 = plt.cm.viridis(ch_means / (ch_means.max() + 1e-8))
    ax_mean_bar.bar(xs, ch_means, color=c1, width=0.8)
    _style_ax(ax_mean_bar,
              title=f'Per-channel mean  (first {n} ch)',
              xlabel='Channel', ylabel='Mean activation')

    # Variance
    c2 = plt.cm.plasma(ch_vars / (ch_vars.max() + 1e-8))
    ax_var_bar.bar(xs, ch_vars, color=c2, width=0.8)
    _style_ax(ax_var_bar,
              title=f'Per-channel variance  (first {n} ch)',
              xlabel='Channel', ylabel='Variance')

    # Sparsity
    c3 = plt.cm.RdYlGn_r(ch_sparsity / 100)
    ax_spar_bar.bar(xs, ch_sparsity, color=c3, width=0.8)
    ax_spar_bar.axhline(50, color='#6b6b80', linewidth=0.6, linestyle='--', alpha=0.6)
    _style_ax(ax_spar_bar,
              title=f'Per-channel sparsity  (first {n} ch)',
              xlabel='Channel', ylabel='Dead neurons (%)')

    for ax in [ax_mean_bar, ax_var_bar, ax_spar_bar]:
        ax.set_facecolor('#12121a')
        ax.spines[:].set_color('#2a2a3a')

    fig.suptitle(
        f'Channel Statistics: Mean · Variance · Sparsity — {layer_label}',
        fontsize=11, color='#e8e8f0', fontfamily='monospace')
    return fig
