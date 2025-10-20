#!/usr/bin/env python3
"""
Compare 8 datasets (e.g., SCP/SDP for k=1..4) in one set of figures.

Inputs (examples):
  python tools/plot_k_multi.py \
    --scp-metrics results_scp_8_k1/element_metrics.csv results_scp_8_k2/element_metrics.csv results_scp_8_k3/element_metrics.csv results_scp_8_k4/element_metrics.csv \
    --sdp-metrics results_sdp_8_k1/element_metrics.csv results_sdp_8_k2/element_metrics.csv results_sdp_8_k3/element_metrics.csv results_sdp_8_k4/element_metrics.csv \
    --k-labels 1 2 3 4 \
    --out-dir results_compare_8

Outputs in out-dir:
  - pareto_energy_multi.png         (8 curves: color by k, SCP solid / SDP dashed)
  - lorenz_energy_density_multi.png (same styling, with equality line)
  - utilization_boxplot_multi.png   (grouped boxplots per k: [SDP, SCP])
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import csv


def _load_element_metrics(path: Path) -> dict:
    data = {'active': [], 'U': [], 'V': [], 'sigma': []}
    with Path(path).open('r', encoding='utf-8') as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                data['active'].append(int(row.get('active', '0')))
                data['U'].append(float(row.get('U_J', 'nan')))
                data['V'].append(float(row.get('V_m3', 'nan')))
                data['sigma'].append(float(row.get('sigma_Pa', 'nan')))
            except Exception:
                continue
    for k in data:
        data[k] = np.asarray(data[k], dtype=float)
    return data


def _pareto_curve(U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    U = np.asarray(U, dtype=float)
    U = U[np.isfinite(U) & (U > 0)]
    if U.size == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    U_sorted = np.sort(U)[::-1]
    cum = np.cumsum(U_sorted)
    y = cum / cum[-1]
    x = (np.arange(1, U_sorted.size + 1)) / U_sorted.size
    return x, y


def _lorenz_curve(U: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    mask = np.isfinite(U) & np.isfinite(V) & (V > 0) & (U >= 0)
    U = U[mask]
    V = V[mask]
    if U.size == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    order = np.argsort(U / V)
    U = U[order]
    V = V[order]
    x = np.cumsum(V) / np.sum(V)
    y = np.cumsum(U) / np.sum(U)
    x = np.concatenate([[0.0], x])
    y = np.concatenate([[0.0], y])
    return x, y


def main():
    ap = argparse.ArgumentParser(description='Overlay plots for SCP/SDP across multiple k values')
    ap.add_argument('--scp-metrics', nargs='+', required=True, help='Paths to SCP element_metrics.csv (order matches k-labels)')
    ap.add_argument('--sdp-metrics', nargs='+', required=True, help='Paths to SDP element_metrics.csv (order matches k-labels)')
    ap.add_argument('--k-labels', nargs='+', required=False, help='Labels for k values (e.g., 1 2 3 4)')
    ap.add_argument('--out-dir', required=True, help='Output directory')
    args = ap.parse_args()

    scp_paths = [Path(p) for p in args.scp_metrics]
    sdp_paths = [Path(p) for p in args.sdp_metrics]
    if len(scp_paths) != len(sdp_paths):
        raise ValueError('scp-metrics and sdp-metrics must have the same length')
    n = len(scp_paths)
    k_labels = args.k_labels if args.k_labels and len(args.k_labels) == n else [str(i+1) for i in range(n)]

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Matplotlib setup
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Color palette (colorblind friendly-ish)
    base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # Load all datasets
    scp_data = [_load_element_metrics(p) for p in scp_paths]
    sdp_data = [_load_element_metrics(p) for p in sdp_paths]

    def active_mask(d):
        a = d['active']
        return np.isfinite(a) & (a > 0.5)

    # Pareto: 8 lines
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for i in range(n):
        color = base_colors[i % len(base_colors)]
        Um_scp = scp_data[i]['U'][active_mask(scp_data[i])]
        Um_sdp = sdp_data[i]['U'][active_mask(sdp_data[i])]
        x_scp, y_scp = _pareto_curve(Um_scp)
        x_sdp, y_sdp = _pareto_curve(Um_sdp)
        ax.plot(x_scp, y_scp, color=color, linewidth=1.1, alpha=0.95, label=f'k={k_labels[i]} SCP')
        ax.plot(x_sdp, y_sdp, color=color, linewidth=1.1, alpha=0.95, linestyle='--', label=f'k={k_labels[i]} SDP')
    ax.set_xlabel('Top members fraction')
    ax.set_ylabel('Cumulative energy share')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, frameon=True)
    fig.tight_layout()
    fig.savefig(outdir / 'pareto_energy_multi.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Lorenz: volume-weighted density on ACTIVE members only, 8 lines
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=1.0, label='Equality')
    for i in range(n):
        color = base_colors[i % len(base_colors)]
        m_scp = active_mask(scp_data[i])
        m_sdp = active_mask(sdp_data[i])
        lx_scp, ly_scp = _lorenz_curve(scp_data[i]['U'][m_scp], scp_data[i]['V'][m_scp])
        lx_sdp, ly_sdp = _lorenz_curve(sdp_data[i]['U'][m_sdp], sdp_data[i]['V'][m_sdp])
        ax.plot(lx_scp, ly_scp, color=color, linewidth=2.0, label=f'k={k_labels[i]} SCP')
        ax.plot(lx_sdp, ly_sdp, color=color, linewidth=2.0, linestyle='--', label=f'k={k_labels[i]} SDP')
    ax.set_xlabel('Cumulative volume fraction')
    ax.set_ylabel('Cumulative energy fraction')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, frameon=True)
    fig.tight_layout()
    fig.savefig(outdir / 'lorenz_energy_density_multi.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Utilization boxplots per k: sigma on active members, MPa
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    scale = 1e-6  # Pa -> MPa
    boxes = []
    labels = []
    positions = []
    width = 0.35
    for i in range(n):
        pos_center = i * 2.0
        # SDP (left), SCP (right)
        sig_sdp = sdp_data[i]['sigma'][active_mask(sdp_data[i])] * scale
        sig_scp = scp_data[i]['sigma'][active_mask(scp_data[i])] * scale
        bp = ax.boxplot([sig_sdp, sig_scp], positions=[pos_center - width/2, pos_center + width/2],
                        widths=0.3, showfliers=False, patch_artist=True)
        # colors
        for patch, clr in zip(bp['boxes'], ['#4c72b0', '#dd8452']):
            patch.set_facecolor(clr)
            patch.set_alpha(0.85)
        for med in bp['medians']:
            med.set_color('black')
            med.set_linewidth(1.2)
        boxes.append(bp)
        labels.append(f'k={k_labels[i]}')
        positions.append(pos_center)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel('|N|/A (MPa)')
    ax.grid(True, axis='y', alpha=0.3)
    # Legend proxy
    import matplotlib.patches as mpatches
    ax.legend([mpatches.Patch(color='#4c72b0'), mpatches.Patch(color='#dd8452')], ['SDP', 'SCP'], loc='best')
    fig.tight_layout()
    fig.savefig(outdir / 'utilization_boxplot_multi.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
