import argparse
import json
import sys
import os
import math
import csv
from datetime import datetime
from pathlib import Path

import numpy as np

from .scp_optimizer import SequentialConvexTrussOptimizer


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run SCP truss optimization (PolarGeometry)")
    p.add_argument('--radius', type=float, default=2.0)
    p.add_argument('--n-sectors', type=int, default=12)
    p.add_argument('--inner-ratio', type=float, default=0.7)
    p.add_argument('--depth', type=float, default=50.0)
    p.add_argument('--volume-fraction', type=float, default=0.1)
    # Output directory for logs (e.g., optimization_log.csv)
    p.add_argument('--output-dir', type=str, default=None,
                   help='Directory to write optimization_log.csv and related logs')
    # Ground structure angular window (K_THETA_STEPS)
    p.add_argument('--k-theta-steps', '--k-theta-step', dest='k_theta_steps', type=int, default=2,
                   help='Angular neighbor window multiplier for ground structure generation (default: 2)')
    # Optional absolute volume cap overrides
    p.add_argument('--volume-cap-abs', type=float, default=None,
                   help='Override absolute volume cap V_max (m^3). If set, ignores --volume-fraction')
    p.add_argument('--volume-cap-from-sector', type=int, default=None,
                   help='Compute absolute V_max from a reference ground structure with this sector count')
    p.add_argument('--volume-cap-frac', type=float, default=None,
                   help='Fraction of reference ground structure volume used to build absolute cap (e.g., 0.2)')
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
    p.add_argument('--node-merge-threshold', type=float, default=None,
                   help='Override node merge distance threshold (meters); defaults to optimizer value (0.2 m)')
    p.add_argument('--export-shell-metrics', type=str, default=None,
                   help='Optional CSV path to append shell displacement/reaction metrics after the run')
    p.add_argument('--shell-metrics-label', type=str, default=None,
                   help='Optional label for shell metrics row (defaults to run context)')
    p.add_argument('--export-element-metrics', type=str, default=None,
                   help='Optional CSV path to export per-element force/energy/utilization metrics for the final state')
    # Shell displacement visualization controls
    p.add_argument('--shell-disp-scale', type=float, default=None,
                   help='Override shell displacement magnification factor (auto if omitted)')
    p.add_argument('--shell-disp-unit', type=str, default='m',
                   help="Displacement colorbar unit: 'm' or 'mm' (default: m)")
    p.add_argument('--shell-disp-cbar-min-mm', type=float, default=None,
                   help='Fix colorbar min (mm). Useful for cross-run comparison')
    p.add_argument('--shell-disp-cbar-max-mm', type=float, default=None,
                   help='Fix colorbar max (mm). Useful for cross-run comparison')
    p.add_argument('--shell-disp-cmap', type=str, default='viridis_r',
                   help="Colormap name for displacement (default 'viridis_r' for light=low, dark=high)")
    p.add_argument('--shell-disp-cbar-fraction', type=float, default=None,
                   help='Matplotlib colorbar fraction (default 0.04)')
    p.add_argument('--shell-disp-cbar-shrink', type=float, default=None,
                   help='Matplotlib colorbar shrink (default 0.80)')
    p.add_argument('--overlay-structure-on-shell', action='store_true',
                   help='Overlay final truss lines on shell displacement plot for a combined figure')
    p.add_argument('--shell-disp-show-title', action='store_true',
                   help='Show title text on shell displacement figure (hidden by default)')
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


def _maybe_override_volume_cap(opt, args) -> None:
    """Optionally override absolute volume cap (V_max) on the optimizer.

    Two modes:
    - --volume-cap-abs: use the provided absolute number directly.
    - --volume-cap-from-sector S + --volume-cap-frac F: compute ground structure
      volume for a reference geometry with S sectors and set V_max = F * V_gs.
    """
    cap_abs = args.volume_cap_abs
    if cap_abs is None and args.volume_cap_from_sector is not None and args.volume_cap_frac is not None:
        try:
            from .truss_system_initializer import TrussSystemInitializer
            ref_init = TrussSystemInitializer(
                radius=float(getattr(opt, 'radius', args.radius)),
                n_sectors=int(args.volume_cap_from_sector),
                inner_ratio=float(getattr(opt, 'inner_ratio', args.inner_ratio)),
                depth=float(getattr(opt, 'depth', args.depth)),
                volume_fraction=1.0,  # temporary
                enable_middle_layer=bool(args.enable_middle_layer),
                middle_layer_ratio=float(args.middle_layer_ratio),
            )
            # Ground structure volume as total_length * A_max
            total_length = float(np.sum(ref_init.element_lengths)) if hasattr(ref_init, 'element_lengths') else 0.0
            a_max = float(getattr(ref_init, 'A_max', getattr(opt, 'A_max', 1e-2)))
            cap_abs = float(total_length * a_max * float(args.volume_cap_frac))
            print(f"[volume-cap] reference sectors={args.volume_cap_from_sector}, total_length={total_length:.3f} m, A_max={a_max:.3e} m², frac={args.volume_cap_frac:.3f}")
            print(f"[volume-cap] absolute cap V_max = {cap_abs*1e6:.2f} cm³")
        except Exception as e:
            print(f"Warning: failed to compute reference volume cap: {e}")
            cap_abs = None
    if cap_abs is not None:
        try:
            opt.volume_constraint = float(cap_abs)
            # propagate to initializer if present
            if hasattr(opt, 'initializer') and getattr(opt.initializer, 'constraint_data', None) is not None:
                try:
                    opt.initializer.constraint_data.volume_constraint = float(cap_abs)
                except Exception:
                    pass
            print(f"[volume-cap] Using absolute V_max = {opt.volume_constraint*1e6:.2f} cm³ (overrides fraction)")
        except Exception as exc:
            print(f"Warning: failed to apply absolute volume cap: {exc}")


def _export_shell_metrics(opt, csv_path: str, label: str = None, node_coords=None) -> bool:
    """Recompute shell response for the provided geometry and append metrics to CSV."""
    load_calc = getattr(opt, 'load_calc', None)
    if load_calc is None:
        print('Shell metrics skipped: load calculator unavailable.')
    return False


def _export_element_metrics(opt, csv_path: str) -> bool:
    """Export per-element force/energy/utilization metrics for the final state.

    Columns: eid,node_i,node_j,L_m,A_m2,V_m3,active,N_abs_N,U_J,U_per_vol_J_per_m3,sigma_Pa
    """
    try:
        import numpy as np
        import csv
    except Exception:
        return False

    # Resolve final state
    theta = getattr(opt, 'final_angles', None)
    if theta is None:
        theta = getattr(opt, 'current_angles', None)
    A = getattr(opt, 'final_areas', None)
    if A is None:
        A = getattr(opt, 'current_areas', None)
    if theta is None or A is None:
        raise RuntimeError('Final angles/areas are unavailable')
    theta = np.asarray(theta, dtype=float)
    A = np.asarray(A, dtype=float)

    # Compute element forces and lengths using existing optimizer routine
    if not hasattr(opt, '_compute_member_forces_and_lengths'):
        raise RuntimeError('Optimizer cannot compute member forces')
    N, L = opt._compute_member_forces_and_lengths(theta, A)
    N = np.asarray(N, dtype=float)
    L = np.asarray(L, dtype=float)

    E = float(getattr(opt, 'E_steel', getattr(getattr(opt, 'material_data', None), 'E_steel', 210e9)))
    thr = float(getattr(opt, 'removal_threshold', 0.0) or 0.0)
    elems = list(getattr(getattr(opt, 'geometry', opt), 'elements', []) or [])
    if not elems:
        raise RuntimeError('No elements available')

    rows = []
    for eid, pair in enumerate(elems):
        n1, n2 = int(pair[0]), int(pair[1])
        a = float(A[eid]) if eid < len(A) else float('nan')
        l = float(L[eid]) if eid < len(L) else float('nan')
        v = a * l if np.isfinite(a) and np.isfinite(l) else float('nan')
        n_abs = abs(float(N[eid])) if eid < len(N) else float('nan')
        u_j = 0.0
        if np.isfinite(a) and a > 0.0 and np.isfinite(l) and l > 0.0 and np.isfinite(n_abs):
            u_j = 0.5 * (n_abs**2) * l / (E * a)
        u_den = (u_j / v) if v and np.isfinite(v) and v > 0 else 0.0
        sigma = (n_abs / a) if a and np.isfinite(a) and a > 0 else 0.0
        active = int(a > thr if np.isfinite(a) else 0)
        rows.append([eid, n1, n2, l, a, v, active, n_abs, u_j, u_den, sigma])

    # Write CSV
    out_path = Path(csv_path)
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    header = ['eid', 'node_i', 'node_j', 'L_m', 'A_m2', 'V_m3', 'active', 'N_abs_N', 'U_J', 'U_per_vol_J_per_m3', 'sigma_Pa']
    with out_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return True
    if getattr(load_calc, 'simple_mode', False) or not getattr(load_calc, 'enable_shell', False):
        print('Shell metrics skipped: shell FEA disabled (simple loads mode).')
        return False
    shell = getattr(load_calc, 'shell_fea', None)
    if shell is None:
        print('Shell metrics skipped: shell FEA instance missing.')
        return False

    import numpy as np

    def _percentile(x, p, default=float('nan')):
        try:
            if x is None:
                return default
            arr = np.asarray(x, dtype=float)
            if arr.size == 0:
                return default
            return float(np.percentile(arr, p))
        except Exception:
            return default

    def _cov(x, default=float('nan')):
        try:
            if x is None:
                return default
            arr = np.asarray(x, dtype=float)
            if arr.size == 0:
                return default
            m = float(np.mean(arr))
            if not np.isfinite(m) or abs(m) <= 0:
                return default
            s = float(np.std(arr))
            return s / m
        except Exception:
            return default

    def _gini(x, default=float('nan')):
        try:
            arr = np.asarray(x, dtype=float)
            if arr.size == 0:
                return default
            # Shift to non-negative
            mn = float(np.min(arr))
            if mn < 0:
                arr = arr - mn
            s = float(np.sum(arr))
            if s == 0:
                return 0.0
            # Gini via sorted cumulative differences
            arr_sorted = np.sort(arr)
            n = arr_sorted.size
            cum = float(np.sum((np.arange(1, n + 1)) * arr_sorted))
            return (2.0 * cum) / (n * s) - (n + 1.0) / n
        except Exception:
            return default

    if node_coords is not None:
        coords = np.asarray(node_coords, dtype=float)
    else:
        theta_final = getattr(opt, 'final_angles', None)
        if theta_final is not None:
            try:
                coords = np.asarray(opt._update_node_coordinates(np.asarray(theta_final, dtype=float)), dtype=float)
            except Exception:
                coords = np.asarray(opt.nodes, dtype=float)
        else:
            coords = np.asarray(opt.nodes, dtype=float)
    if coords.ndim != 2 or coords.size == 0:
        print('Shell metrics skipped: invalid or empty node coordinates.')
        return False

    load_nodes = list(getattr(getattr(opt, 'geometry', None), 'load_nodes', getattr(opt, 'load_nodes', [])) or [])
    if not load_nodes:
        print('Shell metrics skipped: no load nodes defined.')
        return False

    # Ensure we operate with consistent geometry array length
    if coords.shape[0] < max(load_nodes) + 1:
        print('Shell metrics skipped: node coordinate array shorter than load node index.')
        return False

    try:
        load_data = load_calc.compute_hydrostatic_loads(opt.geometry, opt.depth, opt.radius, coords)
    except Exception as exc:
        print(f'Shell metrics export failed while recomputing shell loads: {exc}')
        return False

    disp = getattr(shell, '_last_displacement', None)
    reactions = getattr(shell, '_last_reactions', None)
    if disp is None or reactions is None:
        print('Shell metrics skipped: shell displacement or reactions unavailable.')
        return False

    disp = np.asarray(disp, dtype=float)
    reac = np.asarray(reactions, dtype=float)
    disp_mag = np.linalg.norm(disp, axis=1) if disp.size else np.array([], dtype=float)
    reac_mag = np.linalg.norm(reac, axis=1) if reac.size else np.array([], dtype=float)

    load_vec = np.asarray(getattr(load_data, 'load_vector', []), dtype=float)

    def _safe_stat(arr, func, default=0.0):
        return float(func(arr)) if arr.size else float(default)

    # Energy diagnostics from last shell solve (optional)
    energy_strain = float('nan')
    energy_pressure_work = float('nan')
    energy_balance_residual = float('nan')
    pressure_sum_x = float('nan')
    pressure_sum_y = float('nan')
    try:
        if hasattr(shell, 'get_last_solution'):
            last = shell.get_last_solution()
        else:
            last = None
        if last is not None and last.get('u') is not None and last.get('A') is not None and last.get('rhs') is not None:
            u_vec = np.asarray(last['u'], dtype=float).reshape(-1)
            A = np.asarray(last['A'], dtype=float)
            rhs = np.asarray(last['rhs'], dtype=float)
            n_dof = u_vec.size
            K = A[:n_dof, :n_dof]
            f_p = rhs[:n_dof]
            # 0.5 * u^T K u
            energy_strain = float(0.5 * (u_vec @ (K @ u_vec)))
            energy_pressure_work = float(u_vec @ f_p)
            energy_balance_residual = float(energy_pressure_work - 2.0 * energy_strain)
            fp2 = f_p.reshape(-1, 2)
            pressure_sum_x = float(np.sum(fp2[:, 0]))
            pressure_sum_y = float(np.sum(fp2[:, 1]))
    except Exception:
        pass

    # Per-support reaction magnitude statistics
    reac_mag_per_support = np.sqrt(reac[:, 0] ** 2 + reac[:, 1] ** 2) if reac.ndim == 2 else np.array([])

    metrics = {
        'label': label or 'run',
        'timestamp': datetime.utcnow().isoformat(timespec='seconds'),
        'radius_m': float(getattr(opt, 'radius', float('nan'))),
        'depth_m': float(getattr(opt, 'depth', float('nan'))),
        'volume_fraction': float(getattr(opt, 'volume_fraction', float('nan'))),
        'load_node_count': len(load_nodes),
        'shell_node_count': int(disp.shape[0]),
        'support_count': int(reac.shape[0]) if reac.ndim == 2 else 0,
        'disp_max_m': _safe_stat(disp_mag, np.max),
        'disp_rms_m': _safe_stat(disp_mag, lambda x: np.sqrt(np.mean(x ** 2))),
        'disp_mean_m': _safe_stat(disp_mag, np.mean),
        'disp_median_m': _safe_stat(disp_mag, np.median),
        'disp_p95_m': _safe_stat(disp_mag, lambda x: np.percentile(x, 95)),
        'reaction_max_N': _safe_stat(reac_mag, np.max),
        'reaction_rms_N': _safe_stat(reac_mag, lambda x: np.sqrt(np.mean(x ** 2))),
        'reaction_mean_N': _safe_stat(reac_mag, np.mean),
        'reaction_std_N': _safe_stat(reac_mag, np.std),
        'reaction_min_N': _safe_stat(reac_mag, np.min),
        'reaction_median_N': _safe_stat(reac_mag, np.median),
        'reaction_p05_N': _safe_stat(reac_mag, lambda x: np.percentile(x, 5)),
        'reaction_p95_N': _safe_stat(reac_mag, lambda x: np.percentile(x, 95)),
        'reaction_cov': _cov(reac_mag_per_support),
        'reaction_gini': _gini(reac_mag_per_support),
        'reaction_sum_x_N': _safe_stat(reac[:, 0], np.sum, default=0.0) if reac.ndim == 2 else 0.0,
        'reaction_sum_y_N': _safe_stat(reac[:, 1], np.sum, default=0.0) if reac.ndim == 2 else 0.0,
        'load_l2_N': _safe_stat(load_vec, lambda x: np.linalg.norm(x)),
        'load_rms_N': _safe_stat(load_vec, lambda x: np.sqrt(np.mean(x ** 2))),
        'load_mean_abs_N': _safe_stat(np.abs(load_vec), np.mean),
        'load_p95_abs_N': _safe_stat(np.abs(load_vec), lambda x: np.percentile(x, 95)),
        'load_cov': _cov(np.abs(load_vec)),
        'base_pressure_Pa': float(getattr(load_data, 'base_pressure', 0.0)),
        'pressure_sum_x_N': pressure_sum_x,
        'pressure_sum_y_N': pressure_sum_y,
        'energy_shell_strain_J': energy_strain,
        'energy_pressure_work_J': energy_pressure_work,
        'energy_balance_residual_J': energy_balance_residual,
    }

    out_path = Path(csv_path)
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    fieldnames = [
        'label', 'timestamp', 'radius_m', 'depth_m', 'volume_fraction',
        'load_node_count', 'shell_node_count', 'support_count',
        'disp_max_m', 'disp_rms_m', 'disp_mean_m', 'disp_median_m', 'disp_p95_m',
        'reaction_max_N', 'reaction_rms_N', 'reaction_mean_N', 'reaction_std_N',
        'reaction_min_N', 'reaction_median_N', 'reaction_p05_N', 'reaction_p95_N', 'reaction_cov', 'reaction_gini',
        'reaction_sum_x_N', 'reaction_sum_y_N',
        'load_l2_N', 'load_rms_N', 'load_mean_abs_N', 'load_p95_abs_N', 'load_cov',
        'base_pressure_Pa', 'pressure_sum_x_N', 'pressure_sum_y_N',
        'energy_shell_strain_J', 'energy_pressure_work_J', 'energy_balance_residual_J'
    ]

    file_exists = out_path.exists()
    try:
        with out_path.open('a', newline='', encoding='utf-8') as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics)
        print(f"Shell metrics saved to: {out_path}")
        return True
    except Exception as exc:
        print(f'Shell metrics export failed while writing CSV: {exc}')
        return False


def main(argv=None):
    args = parse_args(argv)
    rings = load_rings(args.rings, args.radius, args.n_sectors, args.inner_ratio,
                       args.enable_middle_layer, args.middle_layer_ratio)

    if args.save_shell_iter:
        base_dir = Path(args.save_figs) if args.save_figs else Path('results')
        shell_fig_dir = base_dir / "shell_displacement_iter"
    else:
        shell_fig_dir = None

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
        k_theta_steps=int(args.k_theta_steps),
        output_dir=args.output_dir,
        simple_loads=bool(args.simple_loads),
        enforce_symmetry=bool(args.enforce_symmetry),
        shell_fig_dir=shell_fig_dir if args.save_shell_iter else None,
        save_shell_iter=bool(args.save_shell_iter),
    )
    if args.node_merge_threshold is not None:
        try:
            opt.node_merge_threshold = float(args.node_merge_threshold)
            print(f"Node merge threshold set to {opt.node_merge_threshold:.4f} m")
        except Exception as exc:
            print(f"Warning: failed to apply node merge threshold ({exc}); using default {opt.node_merge_threshold}")

    if args.shell_debug_log and getattr(opt, 'load_calc', None) is not None:
        enable_log = getattr(opt.load_calc, 'enable_debug_logging', None)
        if callable(enable_log):
            try:
                enable_log(args.shell_debug_log)
                print(f"Shell debug logging enabled: {args.shell_debug_log}")
            except Exception as exc:
                print(f"Warning: failed to enable shell debug logging: {exc}")
    # Apply absolute volume cap override if requested
    try:
        _maybe_override_volume_cap(opt, args)
    except Exception as exc:
        print(f"Warning: volume cap override failed: {exc}")
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
            viz._plot_structure(opt, ax, areas0, title="", linewidth_mode='uniform', node_coords=np.array(opt.nodes), min_area_to_draw=0.0)
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

        coords_opt = None
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
                viz._plot_structure(opt, ax, areas0, title="", linewidth_mode='uniform', node_coords=np.array(opt.nodes), min_area_to_draw=0.0)
                plt.tight_layout(); plt.savefig(os.path.join(out_dir, "ground_structure.png"), dpi=300, bbox_inches='tight'); plt.close(fig)

                # Final structure
                if args.sdp_fixed_geometry:
                    # A-only; theta stays at baseline
                    theta_k = np.asarray(out.get('theta_k', []), dtype=float)
                    A_new = np.asarray(out.get('A_new', []), dtype=float)
                    coords_opt = opt._update_node_coordinates(theta_k)
                    # Expose results on optimizer for downstream exporters
                    try:
                        opt.final_angles = theta_k
                        opt.final_areas = A_new
                    except Exception:
                        pass
                    title = "Single SDP Final Structure (fixed geometry)"
                else:
                    theta_new = np.asarray(out['theta_new'], dtype=float)
                    A_new = np.asarray(out['A_new'], dtype=float)
                    coords_opt = opt._update_node_coordinates(theta_new)
                    try:
                        opt.final_angles = theta_new
                        opt.final_areas = A_new
                    except Exception:
                        pass
                    title = "Single SDP Final Structure"
                fig, ax = plt.subplots(figsize=(10, 6))
                viz._plot_structure(opt, ax, A_new, title=title, linewidth_mode='variable', node_coords=coords_opt)
                plt.tight_layout(); plt.savefig(os.path.join(out_dir, "final_structure.png"), dpi=300, bbox_inches='tight'); plt.close(fig)

                # Shell displacement visualization (if shell FEA is active) with optional overlay
                shell_fea = getattr(getattr(opt, 'load_calc', None), 'shell_fea', None)
                if shell_fea and hasattr(shell_fea, 'visualize_last_solution'):
                    try:
                        # Build overlay drawer to reuse project visualization styling
                        overlay_drawer = None
                        if bool(args.overlay_structure_on_shell):
                            try:
                                viz2 = TrussVisualization()
                                areas_use = A_new
                                thr = float(getattr(opt, 'removal_threshold', 0.0) or 0.0)

                                def _drawer(ax):
                                    try:
                                        viz2._plot_structure(opt, ax, np.asarray(areas_use),
                                                         title="", linewidth_mode='variable', node_coords=np.asarray(coords_opt),
                                                         min_area_to_draw=thr, hide_isolated_nodes=True)
                                    except Exception:
                                        pass

                                overlay_drawer = _drawer
                            except Exception:
                                overlay_drawer = None

                        disp_unit = (args.shell_disp_unit or 'm').lower()
                        cbar_range = None
                        if disp_unit == 'mm' and (args.shell_disp_cbar_min_mm is not None or args.shell_disp_cbar_max_mm is not None):
                            vmin = 0.0 if args.shell_disp_cbar_min_mm is None else float(args.shell_disp_cbar_min_mm)
                            vmax = float(args.shell_disp_cbar_max_mm) if args.shell_disp_cbar_max_mm is not None else None
                            if vmax is not None:
                                cbar_range = (vmin, vmax)

                        shell_fig = os.path.join(out_dir, "shell_displacement.png")
                        shell_fea.visualize_last_solution(
                            scale=float(args.shell_disp_scale) if args.shell_disp_scale is not None else None,
                            save_path=shell_fig,
                            cbar_range=cbar_range,
                            disp_unit=disp_unit,
                            overlay_truss_drawer=overlay_drawer,
                            overlay_kwargs={'color': 'navy', 'linewidths': 0.8, 'alpha': 0.5},
                            show_title=bool(args.shell_disp_show_title),
                            cbar_fraction=(float(args.shell_disp_cbar_fraction) if args.shell_disp_cbar_fraction is not None else None),
                            cbar_shrink=(float(args.shell_disp_cbar_shrink) if args.shell_disp_cbar_shrink is not None else None)
                        )
                    except Exception as e:
                        print(f"Warning: shell displacement plot (single-subproblem) failed: {e}")

                # Area histogram (for paper comparisons with SCP)
                try:
                    areas_mm2 = A_new * 1e6
                    thr_mm2 = float(max(getattr(opt, 'removal_threshold', 0.0), 0.0) * 1e6)
                    valid = areas_mm2[areas_mm2 > thr_mm2]
                    fig, ax = plt.subplots(figsize=(8, 6))
                    if valid.size > 0:
                        num_bins = max(1, min(25, valid.size))
                        ax.hist(valid, bins=num_bins, alpha=0.7, color='skyblue', edgecolor='black')
                        ax.axvline(x=thr_mm2, color='red', linestyle='--', label='Removal Threshold')
                        a_max_mm2 = max(viz._resolve_a_max_mm2(opt), float(np.max(valid)))
                        ax.set_xlim(0, a_max_mm2)
                        ax.set_xticks(np.linspace(0, a_max_mm2, 11))
                        ax.set_xlabel('Cross-sectional Area (mm²)', fontsize=14)
                        ax.set_ylabel('Number of Members', fontsize=14)
                        ax.tick_params(axis='both', labelsize=14)
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                    else:
                        ax.text(0.5, 0.5, 'No valid areas to display', ha='center', va='center', transform=ax.transAxes)
                    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "area_histogram.png"), dpi=300, bbox_inches='tight'); plt.close(fig)
                except Exception as _eh:
                    print(f"Warning: failed to save area histogram: {_eh}")

                # Export numerical data for downstream analysis
                try:
                    if hasattr(opt, '_export_iteration_state_logs'):
                        opt._export_iteration_state_logs(out_dir)
                except Exception as exp:
                    print(f"Warning: failed to export iteration logs: {exp}")
                try:
                    final_areas = getattr(opt, 'final_areas', None)
                    if final_areas is None:
                        final_areas = getattr(opt, 'current_areas', None)
                    if final_areas is None:
                        final_areas = A_new
                    if final_areas is not None:
                        np.savetxt(
                            os.path.join(out_dir, "final_areas.csv"),
                            np.asarray(final_areas, dtype=float),
                            delimiter=',',
                            header='area_m2',
                            comments=''
                        )
                except Exception as exp:
                    print(f"Warning: failed to export final areas: {exp}")

                print(f"Figures saved to: {out_dir}")
            except Exception as e:
                print(f"Warning: failed saving figures for single subproblem: {e}")

        if args.export_shell_metrics:
            label = args.shell_metrics_label or ('single_subproblem_fixed' if args.sdp_fixed_geometry else 'single_subproblem')
            try:
                _export_shell_metrics(opt, args.export_shell_metrics, label=label, node_coords=coords_opt)
            except Exception as exc:
                print(f"Warning: shell metrics export failed: {exc}")
        if args.export_element_metrics:
            try:
                _export_element_metrics(opt, args.export_element_metrics)
                print(f"Element metrics saved to: {args.export_element_metrics}")
            except Exception as exc:
                print(f"Warning: element metrics export failed: {exc}")
        return 0
    
    if args.save_figs or args.save_shell_iter:
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
        # Report optimized active members above removal threshold
        areas_final = getattr(opt, 'final_areas', None)
        if areas_final is None:
            areas_final = getattr(opt, 'current_areas', None)
        if areas_final is not None:
            try:
                thr = float(getattr(opt, 'removal_threshold', 0.0))
            except Exception:
                thr = 0.0
            try:
                n_total = len(areas_final)
                n_active = sum(1 for a in areas_final if a is not None and a > thr)
                print(f"Active members (>thr): {n_active}/{n_total}  removed(<=thr): {n_total - n_active}")
            except Exception:
                pass
    else:
        current_c = getattr(opt, 'current_compliance', None)
        if current_c is not None and math.isfinite(current_c):
            print(f"Current compliance at failure: {current_c:.6e}")
        # Report current active members if available
        areas_now = getattr(opt, 'current_areas', None)
        if areas_now is not None:
            try:
                thr = float(getattr(opt, 'removal_threshold', 0.0))
            except Exception:
                thr = 0.0
            try:
                n_total = len(areas_now)
                n_active = sum(1 for a in areas_now if a is not None and a > thr)
                print(f"Active members (>thr): {n_active}/{n_total}  removed(<=thr): {n_total - n_active}")
            except Exception:
                pass

    if args.export_shell_metrics:
        label = args.shell_metrics_label
        if label is None:
            label = 'final' if not optimization_failed else 'failure_state'
        try:
            _export_shell_metrics(opt, args.export_shell_metrics, label=label)
        except Exception as exc:
            print(f"Warning: shell metrics export failed: {exc}")

    # Optional element metrics export (final state), independent of shell metrics
    if args.export_element_metrics:
        try:
            _export_element_metrics(opt, args.export_element_metrics)
            print(f"Element metrics saved to: {args.export_element_metrics}")
        except Exception as exc:
            print(f"Warning: element metrics export failed: {exc}")

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
            viz._plot_structure(opt, ax, areas_final, title="", linewidth_mode='variable', node_coords=coords_opt)
            plt.tight_layout(); plt.savefig(os.path.join(out_dir, "final_structure.png"), dpi=300, bbox_inches='tight'); plt.close(fig)

            # Load distribution
            viz.plot_single_figure(opt, figure_type="load_distribution",
                                   save_path=os.path.join(out_dir, "load_distribution.png"), figsize=(8, 6))

            # Shell displacement visualization (if shell FEA is active)
            shell_fea = getattr(getattr(opt, 'load_calc', None), 'shell_fea', None)
            if shell_fea and hasattr(shell_fea, 'visualize_last_solution'):
                try:
                    shell_fig = os.path.join(out_dir, "shell_displacement.png")
                    overlay_segments = None
                    overlay_drawer = None
                    if bool(args.overlay_structure_on_shell):
                        try:
                            from .visualization import TrussVisualization
                            viz2 = TrussVisualization()
                            # Prepare geometry and areas
                            theta_final = getattr(opt, 'final_angles', None)
                            if theta_final is not None:
                                coords_overlay = opt._update_node_coordinates(np.asarray(theta_final, dtype=float))
                            else:
                                theta_use = getattr(opt, 'current_angles', None)
                                coords_overlay = opt._update_node_coordinates(np.asarray(theta_use, dtype=float)) if theta_use is not None else np.asarray(opt.nodes, dtype=float)
                            areas_use = getattr(opt, 'final_areas', None)
                            if areas_use is None:
                                areas_use = getattr(opt, 'current_areas', None)
                            thr = float(getattr(opt, 'removal_threshold', 0.0) or 0.0)

                            def _drawer(ax):
                                try:
                                    viz2._plot_structure(opt, ax, np.asarray(areas_use) if areas_use is not None else np.full(opt.n_elements, max(opt.A_min, 1e-4)),
                                                         title="", linewidth_mode='variable', node_coords=np.asarray(coords_overlay), min_area_to_draw=thr, hide_isolated_nodes=True)
                                except Exception:
                                    pass

                            overlay_drawer = _drawer
                        except Exception:
                            overlay_drawer = None

                    # Colorbar range handling (in chosen unit)
                    disp_unit = (args.shell_disp_unit or 'm').lower()
                    cbar_range = None
                    if disp_unit == 'mm' and (args.shell_disp_cbar_min_mm is not None or args.shell_disp_cbar_max_mm is not None):
                        vmin = 0.0 if args.shell_disp_cbar_min_mm is None else float(args.shell_disp_cbar_min_mm)
                        vmax = float(args.shell_disp_cbar_max_mm) if args.shell_disp_cbar_max_mm is not None else None
                        if vmax is not None:
                            cbar_range = (vmin, vmax)

                    shell_fea.visualize_last_solution(
                        scale=float(args.shell_disp_scale) if args.shell_disp_scale is not None else None,
                        save_path=shell_fig,
                        cbar_range=cbar_range,
                        disp_unit=disp_unit,
                        overlay_segments=overlay_segments,
                        overlay_kwargs={'color': 'navy', 'linewidths': 0.8, 'alpha': 0.5},
                        overlay_truss_drawer=overlay_drawer,
                        show_title=bool(args.shell_disp_show_title),
                        cbar_fraction=(float(args.shell_disp_cbar_fraction) if args.shell_disp_cbar_fraction is not None else None),
                        cbar_shrink=(float(args.shell_disp_cbar_shrink) if args.shell_disp_cbar_shrink is not None else None),
                        cmap=str(args.shell_disp_cmap or 'viridis_r')
                    )
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

            # Export numerical data for post-processing
            try:
                if hasattr(opt, '_export_iteration_state_logs'):
                    opt._export_iteration_state_logs(out_dir)
            except Exception as exp:
                print(f"Warning: failed to export iteration logs: {exp}")
            try:
                final_areas = getattr(opt, 'final_areas', None)
                if final_areas is None:
                    final_areas = getattr(opt, 'current_areas', None)
                if final_areas is not None:
                    np.savetxt(
                        os.path.join(out_dir, "final_areas.csv"),
                        np.asarray(final_areas, dtype=float),
                        delimiter=',',
                        header='area_m2',
                        comments=''
                    )
            except Exception as exp:
                print(f"Warning: failed to export final SCP areas: {exp}")


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
