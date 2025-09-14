import argparse
import json
import sys
import os
from pathlib import Path

from .scp_optimizer import SequentialConvexTrussOptimizer


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run SCP truss optimization (PolarGeometry)")
    p.add_argument('--radius', type=float, default=2.0)
    p.add_argument('--n-sectors', type=int, default=12)
    p.add_argument('--inner-ratio', type=float, default=0.7)
    p.add_argument('--depth', type=float, default=50.0)
    p.add_argument('--volume-fraction', type=float, default=0.1)
    p.add_argument('--rings', type=str, default=None, help='Path to rings JSON file (list of ring dicts)')
    p.add_argument('--enable-aasi', action='store_true', help='Enable AASI (phase C)')
    p.add_argument('--max-iterations', type=int, default=10)
    p.add_argument('--save-figs', type=str, default=None, help='Directory to save figures after optimization')
    return p.parse_args(argv)


def load_rings(path: str, radius: float, n_sectors: int, inner_ratio: float):
    if not path:
        # default two-ring template
        return [
            {"radius": radius, "n_nodes": n_sectors + 1, "type": "outer"},
            {"radius": radius * inner_ratio, "n_nodes": n_sectors + 1, "type": "inner"},
        ]
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if not isinstance(data, list):
            raise ValueError('rings file must contain a JSON array')
        return data


def main(argv=None):
    args = parse_args(argv)
    rings = load_rings(args.rings, args.radius, args.n_sectors, args.inner_ratio)

    opt = SequentialConvexTrussOptimizer(
        radius=args.radius,
        n_sectors=args.n_sectors,
        inner_ratio=args.inner_ratio,
        depth=args.depth,
        volume_fraction=args.volume_fraction,
        enable_aasi=args.enable_aasi,
        polar_rings=rings,
    )
    opt.optimization_params.max_iterations = int(args.max_iterations)
    opt.solve_scp_optimization()

    print("\nOptimization finished.")
    if getattr(opt, 'final_compliance', None) is not None:
        print(f"Final compliance: {opt.final_compliance:.6e}")

    # Optional figure saving
    if args.save_figs:
        try:
            # Use non-interactive backend
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import numpy as np
            from .visualization import TrussVisualization

            out_dir = args.save_figs
            os.makedirs(out_dir, exist_ok=True)
            viz = TrussVisualization()

            # Ground structure
            viz.plot_single_figure(opt, figure_type="ground_structure",
                                   save_path=os.path.join(out_dir, "ground_structure.png"), figsize=(8, 6))

            # Optimized structure
            viz.plot_single_figure(opt, figure_type="optimized",
                                   save_path=os.path.join(out_dir, "optimized.png"), figsize=(8, 6))

            # Cleaned structure (avoid relying on final_angles directly)
            try:
                theta_use = getattr(opt, 'final_angles', None) or getattr(opt, 'current_angles', None)
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

            # Trust region evolution
            try:
                viz.plot_trust_region_evolution_only(opt, save_path=os.path.join(out_dir, "trust_region_evolution.png"), show_plot=False)
            except Exception as e:
                print(f"Warning: trust region plot failed: {e}")

            print(f"Figures saved to: {out_dir}")
        except Exception as e:
            print(f"Failed to save figures: {e}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
