import json
import csv
import os
from typing import List, Dict, Any

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_step_details(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Normalize fields
    for rec in data:
        rec.setdefault('iteration', None)
        rec.setdefault('rho', None)
        rec.setdefault('alpha', None)
        rec.setdefault('cond_K_full', None)
        rec.setdefault('cond_K_half', None)
        # active sets
        removed = rec.get('active_set_removed', None)
        if isinstance(removed, list):
            rec['removed_count'] = len(removed)
        else:
            rec['removed_count'] = None
        # deltas
        for k in ('deltaC_A', 'deltaC_theta'):
            if k in rec and rec[k] is not None:
                try:
                    rec[k] = float(rec[k])
                except Exception:
                    rec[k] = None
            else:
                rec[k] = None
        # coerce numerics
        for k in ('rho', 'alpha', 'cond_K_full', 'cond_K_half', 'current_compliance', 'actual_compliance', 'predicted_compliance'):
            if k in rec and rec[k] is not None:
                try:
                    rec[k] = float(rec[k])
                except Exception:
                    pass
    return data


def plot_overview(records: List[Dict[str, Any]], out_path: str):
    # Prepare arrays
    iters = np.array([r.get('iteration', np.nan) for r in records], dtype=float)
    rho = np.array([r.get('rho', np.nan) for r in records], dtype=float)
    cond = np.array([r.get('cond_K_full', np.nan) for r in records], dtype=float)
    removed = np.array([r.get('removed_count', np.nan) for r in records], dtype=float)
    alpha = np.array([r.get('alpha', np.nan) for r in records], dtype=float)
    dA = np.array([r.get('deltaC_A', np.nan) for r in records], dtype=float)
    dTheta = np.array([r.get('deltaC_theta', np.nan) for r in records], dtype=float)

    # Figure with 3 panels
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), constrained_layout=True)

    # Panel 1: rho and cond
    ax1 = axes[0]
    sc = ax1.scatter(iters, rho, c=np.nan_to_num(alpha, nan=0.0), s=20 + 8*np.nan_to_num(removed, nan=0.0), cmap='viridis', alpha=0.9, edgecolors='k', linewidths=0.3)
    ax1.axhline(0.0, color='gray', linestyle='--', linewidth=1)
    ax1.set_title('Step Quality (rho) with alpha color and removed size')
    ax1.set_xlabel('Iteration (entry-wise)')
    ax1.set_ylabel('rho')
    cbar = fig.colorbar(sc, ax=ax1, orientation='vertical')
    cbar.set_label('alpha')

    ax1b = ax1.twinx()
    ax1b.plot(iters, cond, color='tab:red', alpha=0.6, linewidth=1.0, label='cond(K_full)')
    ax1b.set_ylabel('cond(K_full)')
    ax1b.grid(False)
    ax1.legend(loc='upper right')

    # Panel 2: deltaC decomposition (only where available)
    ax2 = axes[1]
    mask = ~np.isnan(dA) & ~np.isnan(dTheta)
    x_idx = np.arange(np.sum(mask))
    if np.any(mask):
        ax2.bar(x_idx, dA[mask], color='tab:blue', alpha=0.7, label='deltaC_A (A update @ fixed θ)')
        ax2.bar(x_idx, dTheta[mask], bottom=dA[mask], color='tab:orange', alpha=0.7, label='deltaC_theta (θ update @ fixed A)')
    ax2.axhline(0.0, color='gray', linestyle='--', linewidth=1)
    ax2.set_title('Compliance change decomposition for abnormal rho steps')
    ax2.set_xlabel('Abnormal-step index')
    ax2.set_ylabel('deltaC')
    ax2.legend()

    # Panel 3: Scatter cond vs rho, size by removed, color by alpha
    ax3 = axes[2]
    sc2 = ax3.scatter(cond, rho, c=np.nan_to_num(alpha, nan=0.0), s=20 + 8*np.nan_to_num(removed, nan=0.0), cmap='plasma', alpha=0.9, edgecolors='k', linewidths=0.3)
    ax3.axhline(0.0, color='gray', linestyle='--', linewidth=1)
    ax3.set_xlabel('cond(K_full)')
    ax3.set_ylabel('rho')
    cbar2 = fig.colorbar(sc2, ax=ax3, orientation='vertical')
    cbar2.set_label('alpha')
    ax3.set_title('cond(K) vs rho (size ~ removed count)')

    fig.suptitle('SCP Step Diagnostics Overview', fontsize=14, fontweight='bold')
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(root, 'results')
    json_path = os.path.join(results_dir, 'step_details.json')
    out_path = os.path.join(results_dir, 'step_diagnostics_overview.png')

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Not found: {json_path}")

    records = load_step_details(json_path)
    plot_overview(records, out_path)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()

