#!/usr/bin/env python3
"""
Plot Pareto (by members) and Lorenz (volume-weighted) curves from element_metrics.csv,
and utilization boxplots for two runs (e.g., SCP vs SDP).

Inputs:
  - --scp-metrics <path to results_scp_*/element_metrics.csv>
  - --sdp-metrics <path to results_sdp_*/element_metrics.csv>
  - --out-dir <directory to save plots>
  - --scp-label / --sdp-label (optional legends)
  - --scp-log / --sdp-log (optional logs to parse compliance for efficiency table)

Outputs:
  - pareto_energy.png
  - lorenz_energy_density.png (with Gini)
  - utilization_boxplot.png (|N|/A on active members)
  - summary.txt (top-10% energy share, Gini, optional k_eff)
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import csv
import re
import sys


def _load_element_metrics(path: Path) -> dict:
    data = {
        'active': [],
        'U': [],
        'V': [],
        'U_den': [],
        'sigma': [],
    }
    with Path(path).open('r', encoding='utf-8') as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                active = int(row.get('active', '0'))
                U = float(row.get('U_J', 'nan'))
                V = float(row.get('V_m3', 'nan'))
                U_den = float(row.get('U_per_vol_J_per_m3', 'nan'))
                sigma = float(row.get('sigma_Pa', 'nan'))
            except Exception:
                continue
            data['active'].append(active)
            data['U'].append(U)
            data['V'].append(V)
            data['U_den'].append(U_den)
            data['sigma'].append(sigma)
    for k in data:
        data[k] = np.asarray(data[k], dtype=float)
    return data


def _gini_from_lorenz(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Gini from Lorenz curve samples (x in [0,1], y cumulative in [0,1])."""
    # Ensure sorted by x
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]
    # Trapezoidal area under curve
    area = np.trapz(y, x)
    return float(1.0 - 2.0 * area)


def _parse_efficiency_from_logs(log_path: Path) -> float | None:
    """Parse final compliance from log if available.
    Returns compliance value or None.
    """
    try:
        text = Path(log_path).read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return None
    patterns = [
        r"Final compliance:\s*([0-9.eE+-]+)",
        r"C_new:\s*([0-9.eE+-]+)"
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
    return None


def _load_shell_metrics(path: Path) -> dict:
    out = {}
    try:
        with Path(path).open('r', encoding='utf-8') as f:
            rdr = csv.DictReader(f)
            rows = list(rdr)
            if rows:
                out = rows[-1]
    except Exception:
        pass
    return out


def plot_all(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    scp = _load_element_metrics(Path(args.scp_metrics))
    sdp = _load_element_metrics(Path(args.sdp_metrics))

    def pareto_curve(U):
        U = np.asarray(U, dtype=float)
        U = U[np.isfinite(U) & (U > 0)]
        if U.size == 0:
            return np.array([0.0, 1.0]), np.array([0.0, 1.0])
        U_sorted = np.sort(U)[::-1]
        cum = np.cumsum(U_sorted)
        y = cum / cum[-1]
        x = (np.arange(1, U_sorted.size + 1)) / U_sorted.size
        return x, y

    def lorenz_curve(U, V, use_density=True):
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        mask = np.isfinite(U) & np.isfinite(V) & (V > 0) & (U >= 0)
        U = U[mask]
        V = V[mask]
        if U.size == 0:
            return np.array([0.0, 1.0]), np.array([0.0, 1.0])
        if use_density:
            order = np.argsort(U / V)
        else:
            order = np.argsort(U)  # fallback
        U = U[order]
        V = V[order]
        x = np.cumsum(V) / np.sum(V)
        y = np.cumsum(U) / np.sum(U)
        # prepend origin
        x = np.concatenate([[0.0], x])
        y = np.concatenate([[0.0], y])
        return x, y

    # Consider only active elements for utilization plots
    def active_mask(d):
        a = d['active']
        return np.isfinite(a) & (a > 0.5)

    # Pareto
    x_scp, y_scp = pareto_curve(scp['U'][active_mask(scp)])
    x_sdp, y_sdp = pareto_curve(sdp['U'][active_mask(sdp)])

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(x_sdp, y_sdp, label=args.sdp_label, color='steelblue', linewidth=2)
    ax.plot(x_scp, y_scp, label=args.scp_label, color='darkorange', linewidth=2)
    ax.set_xlabel('Top members fraction')
    ax.set_ylabel('Cumulative energy share')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / 'pareto_energy.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Lorenz + Gini (volume-weighted)
    lx_scp, ly_scp = lorenz_curve(scp['U'], scp['V'], use_density=True)
    lx_sdp, ly_sdp = lorenz_curve(sdp['U'], sdp['V'], use_density=True)
    gini_scp = _gini_from_lorenz(lx_scp, ly_scp)
    gini_sdp = _gini_from_lorenz(lx_sdp, ly_sdp)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=1, label='Equality')
    ax.plot(lx_sdp, ly_sdp, label=f"{args.sdp_label} (G={gini_sdp:.3f})", color='steelblue', linewidth=2)
    ax.plot(lx_scp, ly_scp, label=f"{args.scp_label} (G={gini_scp:.3f})", color='darkorange', linewidth=2)
    ax.set_xlabel('Cumulative volume fraction')
    ax.set_ylabel('Cumulative energy fraction')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / 'lorenz_energy_density.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Utilization boxplot (|N|/A on active)
    sig_scp = scp['sigma'][active_mask(scp)]
    sig_sdp = sdp['sigma'][active_mask(sdp)]
    n_scp = int(np.sum(active_mask(scp)))
    n_sdp = int(np.sum(active_mask(sdp)))
    # Convert to MPa for nice axis if values are large
    scale = 1e-6
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.boxplot([sig_sdp * scale, sig_scp * scale],
               labels=[f"{args.sdp_label} (n={n_sdp})", f"{args.scp_label} (n={n_scp})"],
               showfliers=False)
    ax.set_ylabel('|N|/A (MPa)')
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / 'utilization_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Optional efficiency table if shell_metrics + logs are provided
    summary_lines = []
    summary_lines.append(f"Active members: {args.sdp_label}={n_sdp}, {args.scp_label}={n_scp}")
    if args.scp_shell_metrics and args.sdp_shell_metrics:
        ms_scp = _load_shell_metrics(Path(args.scp_shell_metrics))
        ms_sdp = _load_shell_metrics(Path(args.sdp_shell_metrics))
        c_scp = _parse_efficiency_from_logs(Path(args.scp_log)) if args.scp_log else None
        c_sdp = _parse_efficiency_from_logs(Path(args.sdp_log)) if args.sdp_log else None
        try:
            f_scp = float(ms_scp.get('load_l2_N', 'nan'))
            f_sdp = float(ms_sdp.get('load_l2_N', 'nan'))
        except Exception:
            f_scp = f_sdp = float('nan')
        if np.isfinite(f_scp) and c_scp and np.isfinite(c_scp):
            k_scp = (f_scp ** 2) / c_scp
            summary_lines.append(f"{args.scp_label}: c={c_scp:.4e}, ||f||₂={f_scp:.2f}, k_eff={k_scp:.4e}")
        if np.isfinite(f_sdp) and c_sdp and np.isfinite(c_sdp):
            k_sdp = (f_sdp ** 2) / c_sdp
            summary_lines.append(f"{args.sdp_label}: c={c_sdp:.4e}, ||f||₂={f_sdp:.2f}, k_eff={k_sdp:.4e}")

    # Add summary: top-10% energy share & Gini
    def top_share(U):
        U = np.asarray(U, dtype=float)
        U = U[np.isfinite(U) & (U > 0)]
        if U.size == 0:
            return float('nan')
        k = max(1, int(0.1 * U.size))
        U_sorted = np.sort(U)[::-1]
        return float(np.sum(U_sorted[:k]) / np.sum(U_sorted))

    share_scp = top_share(scp['U'][active_mask(scp)])
    share_sdp = top_share(sdp['U'][active_mask(sdp)])
    summary_lines.append(f"Top-10% members energy share: {args.sdp_label}={share_sdp:.3f}, {args.scp_label}={share_scp:.3f}")
    summary_lines.append(f"Lorenz Gini: {args.sdp_label}={gini_sdp:.3f}, {args.scp_label}={gini_scp:.3f}")

    (outdir / 'summary.txt').write_text("\n".join(summary_lines), encoding='utf-8')
    print("Saved:", outdir / 'pareto_energy.png')
    print("Saved:", outdir / 'lorenz_energy_density.png')
    print("Saved:", outdir / 'utilization_boxplot.png')
    print("Saved:", outdir / 'summary.txt')


def main():
    ap = argparse.ArgumentParser(description='Plot Pareto/Lorenz/utilization from element_metrics.csv')
    ap.add_argument('--scp-metrics', required=True)
    ap.add_argument('--sdp-metrics', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--scp-label', default='SCP')
    ap.add_argument('--sdp-label', default='SDP')
    ap.add_argument('--scp-shell-metrics')
    ap.add_argument('--sdp-shell-metrics')
    ap.add_argument('--scp-log')
    ap.add_argument('--sdp-log')
    args = ap.parse_args()
    plot_all(args)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
