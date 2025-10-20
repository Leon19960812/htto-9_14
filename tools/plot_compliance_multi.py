#!/usr/bin/env python3
"""
Overlay compliance evolution for multiple SCP runs (different k).

Inputs (examples):
  python tools/plot_compliance_multi.py \
    --logs results_scp_8_k1/optimization_log.csv results_scp_8_k2/optimization_log.csv \
           results_scp_8_k3/optimization_log.csv results_scp_8_k4/optimization_log.csv \
    --labels k=1 k=2 k=3 k=4 \
    --out-dir results_compare_8/all \
    [--normalize] [--use-log-text results_scp_8_k1/log_scp_8_k1.txt ...]

Notes
  - Primary source is optimization_log.csv written per run (with --output-dir).
  - We reconstruct the series as: [initial] + [actual_compliance at accepted steps].
  - If a CSV is missing, you may provide a corresponding textual log file via
    --use-log-text in the same order; the parser will try to read
    'Initial compliance:' and 'Compliance: A -> B' or 'Compliance: A → B'.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import csv
import re
import numpy as np


def _to_bool(x: str) -> bool:
    s = str(x).strip().lower()
    return s in ("1", "true", "yes", "y", "t")


def _load_series_from_csv(path: Path) -> list[float] | None:
    try:
        rows = list(csv.DictReader(Path(path).open('r', encoding='utf-8')))
    except Exception:
        return None
    if not rows:
        return None
    # Many users append multiple runs into the same CSV. Detect iteration resets
    # and keep only the last contiguous segment.
    segments: list[list[float]] = []
    current: list[float] = []
    prev_iter = None
    have_initial = False

    def _flush_segment():
        nonlocal current
        if current:
            segments.append(current)
        current = []

    for row in rows:
        it = None
        try:
            it = int(float(row.get('iteration', 'nan')))
        except Exception:
            it = None
        # start a new segment if iteration counter decreases (new run appended)
        if prev_iter is not None and it is not None and it < prev_iter:
            _flush_segment()
            have_initial = False
        if it is not None:
            prev_iter = it

        if _to_bool(row.get('accept_step', 'false')):
            if not have_initial:
                # initial is current_compliance_before of first accepted row
                try:
                    c0 = float(row.get('current_compliance_before', 'nan'))
                    if np.isfinite(c0):
                        current.append(c0)
                        have_initial = True
                except Exception:
                    pass
            # after-accept value
            val = None
            try:
                v = float(row.get('actual_compliance', 'nan'))
                if np.isfinite(v):
                    val = v
            except Exception:
                val = None
            if val is None:
                try:
                    before = float(row.get('current_compliance_before', 'nan'))
                    imp = float(row.get('improvement_percent', 'nan'))
                    if np.isfinite(before) and np.isfinite(imp):
                        val = before * (1.0 - imp / 100.0)
                except Exception:
                    val = None
            if val is not None and np.isfinite(val):
                current.append(val)

    _flush_segment()
    series = segments[-1] if segments else None
    return series if series else None


def _load_series_from_textlog(path: Path) -> list[float] | None:
    try:
        text = Path(path).read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return None
    series = []
    # Initial compliance line
    m0 = re.search(r"Initial\s+compliance:\s*([0-9.eE+-]+)", text)
    if m0:
        try:
            series.append(float(m0.group(1)))
        except Exception:
            pass
    # Accepted steps lines: both unicode arrow and ascii arrow
    for m in re.finditer(r"Compliance:\s*([0-9.eE+-]+)\s*(?:→|->)\s*([0-9.eE+-]+)", text):
        try:
            series.append(float(m.group(2)))
        except Exception:
            continue
    return series if series else None


def main():
    ap = argparse.ArgumentParser(description='Overlay SCP compliance histories for multiple k values')
    ap.add_argument('--logs', nargs='+', required=True, help='Paths to optimization_log.csv (one per k)')
    ap.add_argument('--labels', nargs='+', required=False, help='Labels for each series (e.g., k=1 k=2 ...)')
    ap.add_argument('--out-dir', required=True, help='Directory to save plot(s)')
    ap.add_argument('--normalize', action='store_true', help='Normalize each series by its initial value')
    ap.add_argument('--use-log-text', nargs='*', help='Optional textual logs (fallback when CSV missing), same order')
    args = ap.parse_args()

    csv_paths = [Path(p) for p in args.logs]
    labels = args.labels if args.labels and len(args.labels) == len(csv_paths) else [f'k={i+1}' for i in range(len(csv_paths))]
    text_paths = [Path(p) for p in (args.use_log_text or [])]
    while len(text_paths) < len(csv_paths):
        text_paths.append(Path(''))

    series_list = []
    for i, p in enumerate(csv_paths):
        s = _load_series_from_csv(p)
        if s is None and text_paths[i] and text_paths[i].exists():
            s = _load_series_from_textlog(text_paths[i])
        if s is None:
            raise FileNotFoundError(f'Cannot load series from {p} and no valid text log provided')
        # Optional normalization
        if args.normalize and s and s[0] != 0 and np.isfinite(s[0]):
            s = [v / s[0] for v in s]
        series_list.append(s)

    # Plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for i, s in enumerate(series_list):
        x = np.arange(len(s))
        color = colors[i % len(colors)]
        ax.plot(x, s, marker='o', markersize=4, linewidth=2.0, color=color, label=labels[i])
    ax.set_xlabel('Accepted iteration')
    ax.set_ylabel('Compliance' + (' (normalized)' if args.normalize else ''))
    # Optional log scale if spread is large and not normalized
    if not args.normalize:
        vals = np.concatenate([np.asarray(s, dtype=float) for s in series_list])
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size and (float(np.max(vals)) / max(float(np.min(vals)), 1e-30) > 50.0):
            ax.set_yscale('log')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(ncol=2)
    fig.tight_layout()

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / ('compliance_multi_norm.png' if args.normalize else 'compliance_multi.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
