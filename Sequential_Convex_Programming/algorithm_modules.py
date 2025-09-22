"""
Algorithm modules for the SCP optimizer.

Clean, minimal, and readable implementations that match the optimizer’s
expected interfaces. Comments are in English; code is UTF-8 and ASCII-safe.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np


# ==========================
# Parameter dataclasses
# ==========================


@dataclass
class TrustRegionParams:
    """Trust region configuration."""

    initial_radius: float = np.pi / 180.0
    max_radius: float = np.deg2rad(2.0)
    min_radius: float = np.pi / 720.0

    shrink_factor: float = 0.5
    expand_factor: float = 1.5
    accept_threshold: float = 0.01
    expand_threshold: float = 0.75


@dataclass
class GeometryParams:
    """Geometric constraints configuration."""

    min_angle_spacing: float = np.pi / 180.0
    boundary_buffer: float = np.pi / 360.0
    neighbor_move_cap: float = np.deg2rad(2.0)
    neighbor_move_eps: float = 1e-3


@dataclass
class OptimizationParams:
    """Optimization configuration."""

    max_iterations: int = 30
    convergence_tol: float = 1e-3
    gradient_fd_step: float = 1e-5


# ==========================
# Trust region manager
# ==========================


class TrustRegionManager:
    """Maintains and updates trust region radius."""

    def __init__(self, params: TrustRegionParams):
        self.params = params
        self.current_radius = float(params.initial_radius)

    def update_radius(self, step_quality: float) -> bool:
        """Update radius based on step quality and return acceptance flag."""
        old = self.current_radius
        if step_quality > self.params.expand_threshold:
            self.current_radius = min(
                self.params.expand_factor * self.current_radius, self.params.max_radius
            )
            accept = True
            status = "EXPAND"
        elif step_quality > self.params.accept_threshold:
            accept = True
            status = "ACCEPT"
        else:
            self.current_radius = max(
                self.params.shrink_factor * self.current_radius, self.params.min_radius
            )
            accept = False
            status = "REJECT"

        print(f"Trust region: {old:.4f} -> {self.current_radius:.4f} ({status})")
        return accept


# ==========================
# Convergence checker
# ==========================


class ConvergenceChecker:
    """Checks convergence using relative changes."""

    def __init__(self, tolerance: float, trust_region_manager: TrustRegionManager):
        self.tolerance = float(tolerance)
        self.trust_region_manager = trust_region_manager

    def check_convergence(
        self,
        theta_old: np.ndarray,
        theta_new: np.ndarray,
        A_old: Optional[np.ndarray] = None,
        A_new: Optional[np.ndarray] = None,
    ) -> bool:
        dtheta = float(np.linalg.norm(theta_new - theta_old))
        rel_theta = dtheta / max(float(np.linalg.norm(theta_old)), np.pi / 180.0)

        rel_A = 0.0
        if A_old is not None and A_new is not None:
            dA = float(np.linalg.norm(A_new - A_old))
            rel_A = dA / max(float(np.linalg.norm(A_old)), 1.0)

        converged = (rel_theta < self.tolerance) and (rel_A < self.tolerance)
        if converged:
            print("Converged:")
            print(f"  dtheta_rel={rel_theta:.3e}, dA_rel={rel_A:.3e}")
            print(f"  trust_radius={self.trust_region_manager.current_radius:.3e}")
        return converged


# ==========================
# Gradient calculator (placeholder)
# ==========================


class GradientCalculator:
    """Returns placeholder gradients for compatibility."""

    def __init__(self, optimizer_ref):
        self.opt = optimizer_ref
        self.fd_step = OptimizationParams().gradient_fd_step

    def compute_gradients(self, theta: np.ndarray) -> Tuple[List, List]:
        # Minimal placeholder: return empty lists to satisfy type checks.
        return [], []


# ==========================
# System calculator
# ==========================


class SystemCalculator:
    """Provides assembled compliance evaluation using the optimizer’s modules."""

    def __init__(self, optimizer_ref):
        self.opt = optimizer_ref

    def compute_actual_compliance(self, theta: np.ndarray, A: np.ndarray) -> float:
        """Compute compliance = f_red^T * u_red using current geometry and areas.

        Coordinates are obtained via the optimizer adapter to allow a future
        switch to PolarGeometry without changing this call-site.
        """
        # Update coordinates from theta using optimizer adapter
        coords = self.opt._update_node_coordinates(theta)
        # Element geometry
        lengths, directions = self.opt.geometry_calc.compute_element_geometry(
            coords, self.opt.geometry.elements
        )
        # Global stiffness and load
        K = self.opt.stiffness_calc.assemble_global_stiffness(
            self.opt.geometry, A, lengths, directions
        )
        # Use explicit load_nodes; legacy outer_nodes is deprecated
        load_nodes = getattr(self.opt.geometry, 'load_nodes', [])
        f = self.opt.load_calc.compute_load_vector(coords, load_nodes, self.opt.depth)
        # Reduce by free DOFs
        Kff = K[np.ix_(self.opt.free_dofs, self.opt.free_dofs)]
        ff = f[self.opt.free_dofs]
        # Stable solve with mild regularization fallback
        try:
            u_red = np.linalg.solve(Kff, ff)
        except np.linalg.LinAlgError:
            lam = 1e-8 * float(np.maximum(1e-16, np.mean(np.diag(Kff)) if Kff.size > 0 else 1.0))
            for _ in range(5):
                try:
                    u_red = np.linalg.solve(Kff + lam * np.eye(Kff.shape[0]), ff)
                    break
                except np.linalg.LinAlgError:
                    lam *= 10.0
            else:
                u_red = np.linalg.pinv(Kff, rcond=1e-10) @ ff
        return float(ff @ u_red)


# ==========================
# Step quality evaluator
# ==========================


class StepQualityEvaluator:
    """Evaluates step quality rho = actual_reduction / predicted_reduction."""

    def __init__(self, optimizer_ref):
        self.opt = optimizer_ref

    def evaluate_step_quality(
        self,
        theta_old: np.ndarray,
        theta_new: np.ndarray,
        A_new: np.ndarray,
        predicted_from_model: Optional[float] = None,
    ) -> float:
        current = float(self.opt.current_compliance)
        actual = float(self.opt.system_calculator.compute_actual_compliance(theta_new, A_new))
        if predicted_from_model is None:
            # Minimal fallback: use actual as predicted to avoid division issues.
            predicted = actual
        else:
            predicted = float(predicted_from_model)

        actual_reduction = current - actual
        predicted_reduction = current - predicted

        # Guard: if模型预测本身就是“变差”（或无法给出下降），直接判定为劣质步长
        # 返回 -inf 让信赖域机制拒绝该步
        if predicted_reduction <= 0.0 or not np.isfinite(predicted_reduction):
            return float('-inf')

        if abs(predicted_reduction) < 1e-16:
            return 1.0 if actual_reduction >= 0.0 else -1.0
        return float(actual_reduction / predicted_reduction)


# ==========================
# Initialization manager
# ==========================


class InitializationManager:
    """Provides initial theta and areas consistent with constraints and volume."""

    def __init__(self, geometry_params: GeometryParams, material_data):
        self.geom = geometry_params
        self.mat = material_data

    def initialize_node_angles(self, outer_nodes: List[int]) -> np.ndarray:
        """Evenly distribute theta for outer nodes respecting boundary buffer."""
        n = len(outer_nodes)
        if n <= 1:
            return np.array([np.pi / 2.0], dtype=float)
        low = float(self.geom.boundary_buffer)
        high = float(np.pi - self.geom.boundary_buffer)
        theta = np.linspace(low, high, n, dtype=float)
        # Enforce minimal spacing
        for i in range(1, n):
            theta[i] = max(theta[i], theta[i - 1] + float(self.geom.min_angle_spacing))
        theta[-1] = min(theta[-1], high)
        return theta

    def initialize_areas(
        self, n_elements: int, element_lengths: np.ndarray, volume_constraint: float
    ) -> np.ndarray:
        """Uniform areas satisfying the global volume constraint within bounds."""
        Lsum = float(np.sum(element_lengths)) if element_lengths is not None else float(n_elements)
        if Lsum <= 0.0:
            A_uniform = max(self.mat.A_min, min(self.mat.A_max, 1.0))
        else:
            A_target = float(volume_constraint) / Lsum
            A_uniform = max(self.mat.A_min, min(self.mat.A_max, A_target))
        return np.full(int(n_elements), float(A_uniform), dtype=float)


# ==========================
# Subproblem solver (placeholder)
# ==========================


class SubproblemSolver:
    """Builds and solves the linearized subproblem (minimal placeholder).

    For baseline compilation and run-ability, we return a zero step that
    satisfies bounds and trust region. This keeps the optimizer loop functional
    while we integrate PolarGeometry and a real convex subproblem later.
    """

    def __init__(self, optimizer_ref):
        self.opt = optimizer_ref
        self.gradient_calc = GradientCalculator(optimizer_ref)
        # Enforce MOSEK-only policy for subproblems. No solver fallback.
        try:
            import cvxpy as cp  # type: ignore
            installed = set(cp.installed_solvers())
        except Exception as e:
            raise RuntimeError("cvxpy with MOSEK is required but cvxpy import failed") from e
        if "MOSEK" not in installed:
            raise RuntimeError("MOSEK solver is required. No ECOS/SCS fallback allowed.")

    def solve_linearized_subproblem(
        self, A_k: np.ndarray, theta_k: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray, float, Optional[float]]]:
        """Solve Schur-complement SDP subproblem via cvxpy + MOSEK.

        LMI: [[K_lin_ff(theta, A), f_lin_ff(theta)], [f_lin_ff(theta)^T, t]] >= 0
        where K_lin_ff(theta, A) = sum_i A_i Ki_ff(θ_k) + sum_j (theta_j-θ_kj) Kθj_ff
              f_lin_ff(theta)     = f_ff(θ_k) + sum_j (theta_j-θ_kj) fθj_ff
        Other constraints: theta box+spacing+TR+caps, A bounds, global volume.
        """
        import cvxpy as cp

        n = int(theta_k.size)
        m = int(A_k.size)

        if n == 0 and m == 0:
            return A_k.copy(), theta_k.copy(), float(self.opt.current_compliance), float(self.opt.current_compliance)

        gp: GeometryParams = getattr(self.opt, "geometry_params", GeometryParams())
        r_tr = float(self.opt.trust_region_manager.current_radius)
        A_min = float(getattr(self.opt, 'A_min', 0.0))
        A_max = float(getattr(self.opt, 'A_max', 1.0))
        lengths = np.asarray(getattr(self.opt, 'element_lengths', np.ones(m)), dtype=float)
        V_max = float(getattr(self.opt, 'volume_constraint', float(np.sum(lengths) * A_max)))

        # Precompute geometry at theta_k
        coords_k = self.opt._update_node_coordinates(theta_k)
        elen_k, edir_k = self.opt._compute_element_geometry(coords_k)
        E = float(getattr(self.opt.material_data, 'E_steel', 1.0))
        # Scale loads to improve conditioning when K uses 1/L kernels
        import numpy as _np
        f_scale = 1.0 / max(float(_np.sqrt(E)), 1.0)

        # Build per-element unit global stiffness matrices at theta_k (A=1)
        n_dof = int(self.opt.n_dof)
        Ki_list = []  # full size
        for i, (n1, n2) in enumerate(self.opt.geometry.elements):
            L = float(max(elen_k[i], 1e-12))
            c, s = float(edir_k[i][0]), float(edir_k[i][1])
            # Use unit kernel normalized by 1/L (pull E out)
            k_coeff = 1.0 / L
            k_local = k_coeff * np.array([
                [c*c, c*s, -c*c, -c*s],
                [c*s, s*s, -c*s, -s*s],
                [-c*c, -c*s, c*c, c*s],
                [-c*s, -s*s, c*s, s*s],
            ], dtype=float)
            K_full = np.zeros((n_dof, n_dof), dtype=float)
            dofs = [2*n1, 2*n1+1, 2*n2, 2*n2+1]
            for r in range(4):
                for cidx in range(4):
                    K_full[dofs[r], dofs[cidx]] += k_local[r, cidx]
            Ki_list.append(K_full)

        free = np.asarray(self.opt.free_dofs, dtype=int)
        n_free = int(free.size)
        # Project element matrices to free DOFs
        Ki_ff = [K[np.ix_(free, free)] for K in Ki_list]

        # Base K at (theta_k, A_k) on free DOFs
        Kff_k = np.zeros((n_free, n_free), dtype=float)
        for i in range(m):
            Kff_k += float(A_k[i]) * Ki_ff[i]

        # Debug diagnostics at baseline
        try:
            Kff_k_sym = 0.5 * (Kff_k + Kff_k.T)
            evals = np.linalg.eigvalsh(Kff_k_sym)
            lam_min = float(np.min(evals)) if evals.size else float('nan')
            lam_max = float(np.max(evals)) if evals.size else float('nan')
            print(f"[SCP-SDP] n_free={n_free}, m={m}, lambda_min(K_aff@k)={lam_min:.3e}, lambda_max={lam_max:.3e}")
        except Exception:
            pass

        # f at theta_k on free DOFs
        f_full_k = self.opt._compute_load_vector(coords_k)
        fff_k = f_full_k[free] * f_scale

        # Finite-difference gradients w.r.t theta (for K and f)
        fd_h = float(getattr(self.opt.optimization_params, 'gradient_fd_step', 1e-6))
        Ktheta_ff = []  # list of (n_free,n_free)
        ftheta_ff = []  # list of (n_free,)
        if n > 0:
            for j in range(n):
                e = np.zeros(n, dtype=float); e[j] = fd_h
                coords_j = self.opt._update_node_coordinates(theta_k + e)
                elen_j, edir_j = self.opt._compute_element_geometry(coords_j)
                # Rebuild Kff for A_k at perturbed theta
                Kff_j = np.zeros((n_free, n_free), dtype=float)
                for i, (n1, n2) in enumerate(self.opt.geometry.elements):
                    L = float(max(elen_j[i], 1e-12))
                    c, s = float(edir_j[i][0]), float(edir_j[i][1])
                    k_coeff = 1.0 / L
                    k_local = k_coeff * np.array([
                        [c*c, c*s, -c*c, -c*s],
                        [c*s, s*s, -c*s, -s*s],
                        [-c*c, -c*s, c*c, c*s],
                        [-c*s, -s*s, c*s, s*s],
                    ], dtype=float)
                    K_full = np.zeros((n_dof, n_dof), dtype=float)
                    dofs = [2*n1, 2*n1+1, 2*n2, 2*n2+1]
                    for r in range(4):
                        for cidx in range(4):
                            K_full[dofs[r], dofs[cidx]] += k_local[r, cidx] * float(A_k[i])
                    # project and accumulate
                    Kff_j += K_full[np.ix_(free, free)]
                Ktheta_ff.append((Kff_j - Kff_k) / fd_h)

                # f gradient
                f_full_j = self.opt._compute_load_vector(coords_j)
                ftheta_ff.append(((f_full_j[free] * f_scale) - fff_k) / fd_h)

                # Restore baseline geometry before next perturbation to avoid drift
                if j < n - 1:
                    self.opt._update_node_coordinates(theta_k)

            # Ensure geometry is reset to baseline after all perturbations
            self.opt._update_node_coordinates(theta_k)

        # Further diagnostics: J_f/J_o split and gamma estimate
        try:
            node_ids = getattr(self.opt, 'theta_node_ids', []) or []
            load_set = set(getattr(self.opt.geometry, 'load_nodes', []) or [])
            J_f = list(range(n))
            J_o = []
            if len(node_ids) == n and load_set:
                J_f = [j for j in range(n) if int(node_ids[j]) in load_set]
                J_o = [j for j in range(n) if int(node_ids[j]) not in load_set]
            caps = getattr(self.opt, 'theta_move_caps', None)
            gamma_est = 0.0
            Lsum_o = 0.0
            for j in J_o:
                Sj = 0.5 * (Ktheta_ff[j] + Ktheta_ff[j].T)
                Lj = float(np.linalg.norm(Sj, 'fro'))
                capj = float(caps[j]) if (caps is not None and len(caps) == n) else float(self.opt.geometry_params.neighbor_move_cap)
                Lsum_o += Lj
                gamma_est += capj * Lj
            # Also compute all-theta bound (J_f + J_o)
            gamma_all = 0.0
            Lsum_all = 0.0
            for j in range(n):
                Sj = 0.5 * (Ktheta_ff[j] + Ktheta_ff[j].T)
                Lj = float(np.linalg.norm(Sj, 'fro'))
                capj = float(caps[j]) if (caps is not None and len(caps) == n) else float(self.opt.geometry_params.neighbor_move_cap)
                Lsum_all += Lj
                gamma_all += capj * Lj
            print(f"[SCP-SDP] J_f={len(J_f)}, J_o={len(J_o)}, sum||Sym(Kθ_j)||_F(J_o)={Lsum_o:.3e}, gamma_est(J_o)={gamma_est:.3e}, sum||Sym(Kθ_j)||_F(all)={Lsum_all:.3e}, gamma_est(all)={gamma_all:.3e}, TR={r_tr:.3e}")
        except Exception:
            pass

        # Decision variables
        theta = cp.Variable(n) if n > 0 else None
        A = cp.Variable(m) if m > 0 else None
        t = cp.Variable()  # compliance upper bound

        constraints = []

        # Geometry constraint diagnostics (initial slack at baseline order)
        try:
            if n > 1:
                gp_loc = gp
                node_ids = getattr(self.opt, 'theta_node_ids', []) or []
                if len(node_ids) == n:
                    coords_theta = np.asarray(coords_k, dtype=float)[np.asarray(node_ids, dtype=int)]
                    radii = np.hypot(coords_theta[:, 0], coords_theta[:, 1])
                    Rmax = float(np.max(radii)) if radii.size else 1.0
                    r_tol = max(1e-12, 1e-6 * Rmax)
                    dtheta = np.diff(theta_k)
                    same_mask = np.abs(np.diff(radii)) <= r_tol
                    cross_mask = ~same_mask
                    min_same_slack = None
                    min_cross_slack = None
                    if np.any(same_mask):
                        min_same_slack = float(np.min(dtheta[same_mask] - float(gp_loc.min_angle_spacing)))
                    if np.any(cross_mask):
                        min_cross_slack = float(np.min(dtheta[cross_mask]))
                    slack_lb = float(theta_k[0] - float(gp_loc.boundary_buffer))
                    slack_ub = float((np.pi - float(gp_loc.boundary_buffer)) - theta_k[-1])
                    print(f"[SCP-GEO] n_vars={n}, same_pairs={int(np.sum(same_mask))}, cross_pairs={int(np.sum(cross_mask))}, "
                          f"min_same_slack={min_same_slack}, min_cross_slack={min_cross_slack}, "
                          f"buffer_slack=({slack_lb:.3e},{slack_ub:.3e})")
        except Exception:
            pass

        # Theta constraints
        if n > 0:
            constraints.append(theta[0] >= float(gp.boundary_buffer))
            constraints.append(theta[-1] <= float(np.pi - gp.boundary_buffer))
            # 全局角序单调：对“近似同环”（半径接近）的相邻对施加最小角间距；
            # 跨环相邻对仅要求非递减（不加 spacing），避免多环同角导致的不可行。
            try:
                node_ids = getattr(self.opt, 'theta_node_ids', []) or []
                if len(node_ids) == n:
                    # 使用基线坐标估计半径
                    coords_theta = np.asarray(coords_k, dtype=float)[np.asarray(node_ids, dtype=int)]
                    radii = np.hypot(coords_theta[:, 0], coords_theta[:, 1])
                    Rmax = float(np.max(radii)) if radii.size else 1.0
                    r_tol = max(1e-12, 1e-6 * Rmax)
                    for i in range(1, n):
                        if abs(float(radii[i] - radii[i - 1])) <= r_tol:
                            constraints.append(theta[i] >= theta[i - 1] + float(gp.min_angle_spacing))
                        else:
                            constraints.append(theta[i] >= theta[i - 1])
                else:
                    for i in range(1, n):
                        constraints.append(theta[i] >= theta[i - 1] + float(gp.min_angle_spacing))
            except Exception:
                for i in range(1, n):
                    constraints.append(theta[i] >= theta[i - 1] + float(gp.min_angle_spacing))
            constraints.append(cp.norm2(theta - theta_k) <= r_tr)
            caps = getattr(self.opt, "theta_move_caps", None)
            if caps is not None and len(caps) == n:
                for i in range(n):
                    cap_i = float(caps[i])
                    constraints.append(theta[i] <= theta_k[i] + cap_i)
                    constraints.append(theta[i] >= theta_k[i] - cap_i)

            if getattr(self.opt, 'symmetry_active', False):
                pairs = getattr(self.opt, 'symmetry_pairs', []) or []
                for (i_idx, j_idx) in pairs:
                    if i_idx < n and j_idx < n:
                        constraints.append(theta[i_idx] + theta[j_idx] == float(np.pi))
                fixed_idx = getattr(self.opt, 'symmetry_fixed_indices', []) or []
                for idx in fixed_idx:
                    if idx < n:
                        constraints.append(theta[idx] == float(np.pi / 2.0))

        # Area constraints
        area_sym_active = bool(getattr(self.opt, 'area_symmetry_active', False))
        if m > 0:
            constraints += [A >= A_min, A <= A_max]
            constraints.append(lengths @ A <= V_max)
            A_req = getattr(self.opt, 'A_req_buckling', None)
            if A_req is not None and len(A_req) == m:
                constraints.append(A >= np.asarray(A_req, dtype=float))
            if area_sym_active:
                member_pairs = getattr(self.opt, 'symmetry_member_pairs', []) or []
                for (i_idx, j_idx) in member_pairs:
                    if i_idx < m and j_idx < m:
                        constraints.append(A[i_idx] == A[j_idx])

        # Build K_lin_ff and f_lin_ff affine expressions
        # K_lin_ff(theta, A) = sum_i A_i Ki_ff + sum_j (theta_j-θ_kj) Ktheta_ff[j]
        predicted_t = None
        if n_free == 0:
            # Degenerate; fallback to projection objective
            obj_terms = []
            if n > 0:
                obj_terms.append(cp.sum_squares(theta - theta_k))
            if m > 0:
                obj_terms.append(1e-6 * cp.sum_squares(A - A_k))
            objective = cp.Minimize(sum(obj_terms) if obj_terms else 0)
            prob = cp.Problem(objective, constraints)
        else:
            # Base affine stiffness (A-part only)
            K_aff = 0
            if m > 0:
                K_aff = sum(A[i] * Ki_ff[i] for i in range(m))
            else:
                K_aff = np.zeros((n_free, n_free))

            # Full θ linearization: include all θ directions in the stiffness expansion
            K_lin_keep = K_aff
            for j in range(n):
                K_lin_keep = K_lin_keep + (theta[j] - float(theta_k[j])) * Ktheta_ff[j]

            # f_lin_ff(theta)
            f_lin = fff_k.copy()
            if n > 0 and ftheta_ff:
                for j in range(n):
                    f_lin = f_lin + (theta[j] - float(theta_k[j])) * ftheta_ff[j]

            # LMI via Schur complement: [K_lin, f_lin; f_lin^T, t] ⪰ 0
            M11 = 0.5 * (K_lin_keep + K_lin_keep.T)
            M12 = cp.reshape(f_lin, (n_free, 1))
            M21 = cp.reshape(f_lin, (1, n_free))
            M22 = cp.reshape(t, (1, 1))
            M = cp.bmat([[M11, M12], [M21, M22]])
            constraints.append(M >> 0)

            # Objective: minimize t (+ small regularization)
            reg = 1e-9
            obj_terms = [t]
            if n > 0:
                obj_terms.append(reg * cp.sum_squares(theta - theta_k))
            if m > 0:
                obj_terms.append(reg * cp.sum_squares(A - A_k))
            objective = cp.Minimize(sum(obj_terms))
            prob = cp.Problem(objective, constraints)

        # Solve
        def _disable_area_symmetry_once() -> None:
            self.opt.area_symmetry_active = False
            self.opt.symmetry_member_pairs = []
            self.opt.symmetry_member_fixed = []

        try:
            prob.solve(solver=cp.MOSEK, verbose=False)
        except Exception as e:
            if area_sym_active:
                print(f"⚠️ Area symmetry solve failed; temporarily disabling area equality: {e}")
                _disable_area_symmetry_once()
                return self.solve_linearized_subproblem(A_k, theta_k)
            raise RuntimeError(f"MOSEK solve failed for SDP subproblem: {e}") from e
        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            if area_sym_active:
                print(f"⚠️ Area symmetry led to solver status {prob.status}; disabling and retrying automatically.")
                _disable_area_symmetry_once()
                return self.solve_linearized_subproblem(A_k, theta_k)
            raise RuntimeError(f"Subproblem not solved to optimality with MOSEK: status={prob.status}")

        theta_new = np.asarray(theta.value, dtype=float) if n > 0 else theta_k.copy()
        A_new = np.asarray(A.value, dtype=float) if m > 0 else A_k.copy()
        try:
            predicted_t = float(t.value) if 't' in locals() and t.value is not None else None
        except Exception:
            predicted_t = None
        compliance_new = self.opt.system_calculator.compute_actual_compliance(theta_new, A_new)
        return A_new, theta_new, float(compliance_new), predicted_t

    # --------------------------
    # Helpers
    # --------------------------
    def _project_theta(self, theta: np.ndarray) -> np.ndarray:
        gp = getattr(self.opt, "geometry_params", GeometryParams())
        # Box constraints
        if theta.size > 0:
            theta[0] = max(float(theta[0]), float(gp.boundary_buffer))
            theta[-1] = min(float(theta[-1]), float(np.pi - gp.boundary_buffer))
        # Minimal spacing (monotone increasing by spacing)
        for i in range(1, theta.size):
            min_allowed = float(theta[i - 1] + gp.min_angle_spacing)
            if theta[i] < min_allowed:
                theta[i] = min_allowed
        return theta
