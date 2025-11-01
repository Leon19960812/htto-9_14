"""
Generate a draft set of SCP figures for the paper.

Outputs:
- results/paper/scp/ground_structure.png
- results/paper/scp/optimized.png
- results/paper/scp/cleaned.png
- results/paper/scp/load_distribution.png
- results/paper/scp/area_histogram.png
- results/paper/scp/trust_region_evolution.png
"""

import os
import sys

# Prefer headless backend for CI/headless environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np

from scp_optimizer import SequentialConvexTrussOptimizer
from visualization import TrussVisualization


def main():
    out_dir = os.path.join("results", "paper", "scp")
    os.makedirs(out_dir, exist_ok=True)

    # Build and solve SCP (keep iterations modest for draft)
    opt = SequentialConvexTrussOptimizer(
        radius=2.0,
        n_sectors=12,
        inner_ratio=0.6,
        depth=10.0,
        volume_fraction=0.2,
        enable_middle_layer=False,
    )
    opt.optimization_params.max_iterations = 4
    areas, comp = opt.solve_scp_optimization()

    viz = TrussVisualization()

    # Ground structure
    viz.plot_single_figure(opt, figure_type="ground_structure",
                           save_path=os.path.join(out_dir, "ground_structure.png"), figsize=(8, 6))

    # Optimized structure
    viz.plot_single_figure(opt, figure_type="optimized",
                           save_path=os.path.join(out_dir, "optimized.png"), figsize=(8, 6))

    # Cleaned structure (areas below removal threshold suppressed)
    # Work around visualization's reliance on final_angles by providing coords directly
    try:
        theta_final = getattr(opt, 'final_angles', None)
        if theta_final is not None:
            theta_use = theta_final
        else:
            theta_use = getattr(opt, 'current_angles', None)
        coords_opt = opt._update_node_coordinates(theta_use)
        cleaned = opt.current_areas.copy()
        cleaned[cleaned < opt.removal_threshold] = 0
        fig, ax = plt.subplots(figsize=(8, 6))
        viz._plot_structure(opt, ax, cleaned, title="", linewidth_mode='variable', node_coords=coords_opt)
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, "cleaned.png"), dpi=300, bbox_inches='tight'); plt.close(fig)
    except Exception as e:
        print(f"Warning: cleaned structure plot failed: {e}")

    # Load distribution
    viz.plot_single_figure(opt, figure_type="load_distribution",
                           save_path=os.path.join(out_dir, "load_distribution.png"), figsize=(8, 6))

    # Area histogram
    viz.plot_single_figure(opt, figure_type="area_histogram",
                           save_path=os.path.join(out_dir, "area_histogram.png"), figsize=(8, 6))

    # Trust region evolution (use non-interactive save)
    try:
        fig_path = os.path.join(out_dir, "trust_region_evolution.png")
        viz.plot_trust_region_evolution_only(opt, save_path=fig_path, show_plot=False)
    except Exception as e:
        print(f"Warning: trust region evolution plot failed: {e}")

    print(f"Figures saved to: {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
