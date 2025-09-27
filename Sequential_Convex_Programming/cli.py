import argparse
import json
import sys
import os
import math
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
    # Explicit ring control (matches optimizer/initializer capabilities)
    p.add_argument('--enable-middle-layer', action='store_true', help='Enable middle ring (third layer)')
    p.add_argument('--middle-layer-ratio', type=float, default=0.85, help='Radius ratio for middle ring (if enabled)')
    p.add_argument('--enable-aasi', action='store_true', help='Enable AASI (phase C)')
    p.add_argument('--max-iterations', type=int, default=10)
    p.add_argument('--save-figs', type=str, default=None, help='Directory to save figures after optimization')
    p.add_argument('--single-subproblem', action='store_true', help='Run one SDP subproblem only and exit')
    p.add_argument('--save-sdp', type=str, default=None, help='Optional path to save single SDP result JSON')
    p.add_argument('--sdp-fixed-geometry', action='store_true', help='Use fixed-geometry SDP (A-only), freeze theta')
    p.add_argument('--sdp-reg-eps', type=float, default=0.0, help='Optional small PSD regularization added to K in fixed-geometry SDP')
    p.add_argument('--sdp-lmi-eps', type=float, default=0.0, help='Optional small slack eps for LMI: [K f; f^T t] >> eps*I')
    p.add_argument('--sdp-verbose', action='store_true', help='Enable solver verbose for single fixed-geometry SDP')
    p.add_argument('--simple-loads', action='store_true', help='Use simple hydrostatic loads at load nodes (radial in), bypass shell FEA')
    p.add_argument('--enforce-symmetry', action='store_true', help='Enforce mirror symmetry constraints on theta variables')
    p.add_argument('--save-shell-iter', action='store_true', help='Save shell displacement figure for each accepted iteration')
    p.add_argument('--shell-debug-log', type=str, default=None, help='Optional path to log shell support mapping/loading diagnostics')
    return p.parse_args(argv)


def load_rings(path: str, radius: float, n_sectors: int, inner_ratio: float,
               enable_middle_layer: bool = False, middle_layer_ratio: float = 0.85):
    if not path:
        # default two-ring template
        rings = [
            {"radius": radius, "n_nodes": n_sectors + 1, "type": "outer"},
            {"radius": radius * inner_ratio, "n_nodes": n_sectors + 1, "type": "inner"},
        ]
        if enable_middle_layer:
            rings.insert(1, {"radius": radius * middle_layer_ratio, "n_nodes": n_sectors + 1, "type": "middle"})
        return rings
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if not isinstance(data, list):
            raise ValueError('rings file must contain a JSON array')
        return data


def main(argv=None):
    args = parse_args(argv)
    rings = load_rings(args.rings, args.radius, args.n_sectors, args.inner_ratio,
                       args.enable_middle_layer, args.middle_layer_ratio)

    shell_fig_dir = Path(args.save_figs) / "shell_displacement_iter" if args.save_figs else None

    opt = SequentialConvexTrussOptimizer(
        radius=args.radius,
        n_sectors=args.n_sectors,
        inner_ratio=args.inner_ratio,
        depth=args.depth,
        volume_fraction=args.volume_fraction,
        enable_middle_layer=args.enable_middle_layer,
        middle_layer_ratio=args.middle_layer_ratio,
        enable_aasi=args.enable_aasi,
        polar_rings=rings,
        simple_loads=bool(args.simple_loads),
        enforce_symmetry=bool(args.enforce_symmetry),
        shell_fig_dir=shell_fig_dir if args.save_shell_iter else None,
        save_shell_iter=bool(args.save_shell_iter),
    )

    if args.shell_debug_log and getattr(opt, 'load_calc', None) is not None:
        enable_log = getattr(opt.load_calc, 'enable_debug_logging', None)
        if callable(enable_log):
            try:
                enable_log(args.shell_debug_log)
                print(f"Shell debug logging enabled: {args.shell_debug_log}")
            except Exception as exc:
                print(f"Warning: failed to enable shell debug logging: {exc}")
    opt.optimization_params.max_iterations = int(args.max_iterations)
    # Optionally save pre-optimization ground structure (baseline)
    pre_out = None
    if args.save_figs:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import numpy as np
            from .visualization import TrussVisualization
            out_dir = args.save_figs
            os.makedirs(out_dir, exist_ok=True)
            viz = TrussVisualization()
            # Uniform areas for baseline drawing
            areas0 = np.full(opt.n_elements, max(opt.A_min, 1e-4), dtype=float)
            fig, ax = plt.subplots(figsize=(8, 6))
            viz._plot_structure(opt, ax, areas0, title="Ground Structure (pre-optimization)", linewidth_mode='uniform', node_coords=np.array(opt.nodes), min_area_to_draw=0.0)
            plt.tight_layout(); plt.savefig(os.path.join(out_dir, "ground_structure.png"), dpi=300, bbox_inches='tight'); plt.close(fig)
            pre_out = out_dir
        except Exception as e:
            print(f"Warning: failed saving pre-optimization ground structure: {e}")

    if args.single_subproblem:
        print("Running a single SDP subproblem (diagnostic mode)...")
        if args.sdp_fixed_geometry:
            from .sdp_single_fixed import run_single_fixed_sdp
            out = run_single_fixed_sdp(
                opt,
                save_path=args.save_sdp,
                reg_eps=float(args.sdp_reg_eps),
                lmi_eps=float(args.sdp_lmi_eps),
                verbose=bool(args.sdp_verbose),
            )
        else:
            out = opt.run_single_subproblem(save_path=args.save_sdp)
        # Brief console summary
        print(f"C_baseline: {out['C_k']:.6e}")
        print(f"C_new:      {out['C_new']:.6e}")
        if out['t_pred'] is not None:
            print(f"t_pred:     {out['t_pred']:.6e}")
        print(f"theta_len:  {len(out['theta_k'])} -> {len(out['theta_new'])}")
        print(f"A_len:      {len(out['A_k'])} -> {len(out['A_new'])}")
        # Area stats for visualization/removal understanding
        print("\nArea stats (m^2):")
        print(f"  A_min={out['A_min']:.3e}, removal_threshold={out['removal_threshold']:.3e}, A_max={out['A_max']:.3e}")
        print(f"  min={out['min_area']:.3e}, p1={out['p1_area']:.3e}, p5={out['p5_area']:.3e}, median={out['median_area']:.3e}")
        print(f"  active(>thr): {out['n_active']}/{out['n_total']}  removed(<=thr): {out['n_total']-out['n_active']}")

        # Optional figure saving for single subproblem run
        if args.save_figs:
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                import numpy as np
                from .visualization import TrussVisualization
                out_dir = args.save_figs
                os.makedirs(out_dir, exist_ok=True)
                viz = TrussVisualization()

                # Ground structure (baseline geometry with uniform areas)
                areas0 = np.full(opt.n_elements, max(opt.A_min, 1e-4), dtype=float)
                fig, ax = plt.subplots(figsize=(8, 6))
                viz._plot_structure(opt, ax, areas0, title="Ground Structure (single subproblem)", linewidth_mode='uniform', node_coords=np.array(opt.nodes), min_area_to_draw=0.0)
                plt.tight_layout(); plt.savefig(os.path.join(out_dir, "ground_structure.png"), dpi=300, bbox_inches='tight'); plt.close(fig)

                # Final structure
                if args.sdp_fixed_geometry:
                    # A-only; theta stays at baseline
                    theta_k = np.asarray(out.get('theta_k', []), dtype=float)
                    A_new = np.asarray(out.get('A_new', []), dtype=float)
                    coords_opt = opt._update_node_coordinates(theta_k)
                    title = "Single SDP Final Structure (fixed geometry)"
                else:
                    theta_new = np.asarray(out['theta_new'], dtype=float)
                    A_new = np.asarray(out['A_new'], dtype=float)
                    coords_opt = opt._update_node_coordinates(theta_new)
                    title = "Single SDP Final Structure"
                fig, ax = plt.subplots(figsize=(10, 6))
                viz._plot_structure(opt, ax, A_new, title=title, linewidth_mode='variable', node_coords=coords_opt)
                plt.tight_layout(); plt.savefig(os.path.join(out_dir, "final_structure.png"), dpi=300, bbox_inches='tight'); plt.close(fig)

                print(f"Figures saved to: {out_dir}")
            except Exception as e:
                print(f"Warning: failed saving figures for single subproblem: {e}")
        return 0
    
    if args.save_figs:
        shell_iter_dir = shell_fig_dir
        if shell_iter_dir is not None and args.save_shell_iter:
            try:
                shell_iter_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            print(f"[cli] shell displacement directory set to {shell_iter_dir}")

    optimization_failed = False
    failure_exc = None
    try:
        opt.solve_scp_optimization()
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        optimization_failed = True
        failure_exc = exc
        print("\n⚠️ Optimization terminated with an error. Partial results will be used for exports.")
        print(f"   Reason: {exc}")

    history_dir = args.save_figs if args.save_figs else 'results'
    try:
        opt._export_iteration_state_logs(history_dir)
    except Exception as e:
        print(f"Warning: failed to export theta/area history: {e}")

    if optimization_failed:
        if getattr(opt, 'final_areas', None) is None:
            opt.final_areas = getattr(opt, 'current_areas', None)
        if getattr(opt, 'final_angles', None) is None:
            opt.final_angles = getattr(opt, 'current_angles', None)
        if getattr(opt, 'final_compliance', None) is None:
            opt.final_compliance = getattr(opt, 'current_compliance', None)

    if not optimization_failed:
        print("\nOptimization finished.")
        if getattr(opt, 'final_compliance', None) is not None:
            print(f"Final compliance: {opt.final_compliance:.6e}")
    else:
        current_c = getattr(opt, 'current_compliance', None)
        if current_c is not None and math.isfinite(current_c):
            print(f"Current compliance at failure: {current_c:.6e}")

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

            # Final structure (single figure showing final theta and A)
            theta_final = getattr(opt, 'final_angles', None)
            if theta_final is not None:
                theta_use = theta_final
            else:
                theta_use = getattr(opt, 'current_angles', None)
            coords_opt = opt._update_node_coordinates(theta_use)
            areas_final = getattr(opt, 'current_areas', None)
            if areas_final is None:
                areas_final = np.full(opt.n_elements, max(opt.A_min, 1e-4), dtype=float)
            # Print load nodes quick info for diagnosis
            ln = getattr(opt.geometry, 'load_nodes', []) or []
            if ln:
                import numpy as np
                p0 = coords_opt[ln[0]]
                p1 = coords_opt[ln[-1]]
                print(f"Load nodes: {len(ln)}; first id={ln[0]} at ({p0[0]:.3f},{p0[1]:.3f}), last id={ln[-1]} at ({p1[0]:.3f},{p1[1]:.3f})")
            # Emphasize line width dynamic range
            fig, ax = plt.subplots(figsize=(10, 6))
            viz._plot_structure(opt, ax, areas_final, title="SCP Final Structure", linewidth_mode='variable', node_coords=coords_opt)
            plt.tight_layout(); plt.savefig(os.path.join(out_dir, "final_structure.png"), dpi=300, bbox_inches='tight'); plt.close(fig)

            # Load distribution
            viz.plot_single_figure(opt, figure_type="load_distribution",
                                   save_path=os.path.join(out_dir, "load_distribution.png"), figsize=(8, 6))

            # Shell displacement visualization (if shell FEA is active)
            shell_fea = getattr(getattr(opt, 'load_calc', None), 'shell_fea', None)
            if shell_fea and hasattr(shell_fea, 'visualize_last_solution'):
                try:
                    shell_fig = os.path.join(out_dir, "shell_displacement.png")
                    shell_fea.visualize_last_solution(scale=None, save_path=shell_fig)
                except Exception as e:
                    print(f"Warning: shell displacement plot failed: {e}")

            # Area histogram
            viz.plot_single_figure(opt, figure_type="area_histogram",
                                   save_path=os.path.join(out_dir, "area_histogram.png"), figsize=(8, 6))

            # Compliance evolution
            try:
                viz.plot_compliance_evolution(opt, save_path=os.path.join(out_dir, "compliance_evolution.png"), show_plot=False)
            except Exception as e:
                print(f"Warning: compliance plot failed: {e}")

            # Trust region evolution
            try:
                viz.plot_trust_region_evolution_only(opt, save_path=os.path.join(out_dir, "trust_region_evolution.png"), show_plot=False)
            except Exception as e:
                print(f"Warning: trust region plot failed: {e}")


            print(f"Figures saved to: {out_dir}")
            if optimization_failed:
                print("Partial figures correspond to the last accepted iterate.")
        except Exception as e:
            print(f"Failed to save figures: {e}")
            if optimization_failed:
                print("Partial figures correspond to the last accepted iterate.")

    return 0


if __name__ == '__main__':
    sys.exit(main())
