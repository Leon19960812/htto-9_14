"""
Single-step fixed-geometry SDP (A-only) using the current optimizer stack.

This isolates a pure SDP that optimizes areas A with geometry frozen at
the current baseline theta_k. It is intended for debugging/benchmarking
against the SCP linearized subproblem, and to mirror the logic of the
previous fixed-geometry SDP while staying compatible with PolarGeometry.

Usage: triggered from CLI with --single-subproblem --sdp-fixed-geometry.
"""

from __future__ import annotations

from typing import Tuple, Optional, Dict, Any
import numpy as np


def run_single_fixed_sdp(optimizer, save_path: Optional[str] = None, reg_eps: float = 0.0, lmi_eps: float = 0.0, verbose: bool = False) -> Dict[str, Any]:
    """Solve a fixed-geometry SDP (optimize A only, freeze theta).

    Returns a dict with inputs/outputs; if save_path is provided, dumps JSON
    there (aligned with CLI expectations).
    """
    import cvxpy as cp
    import os, json

    # 1) Baseline variables and geometry
    theta_k, A_k = optimizer._initialize_optimization_variables()
    coords_k = optimizer._update_node_coordinates(theta_k)
    elen_k, edir_k = optimizer._compute_element_geometry(coords_k)
    # Baseline compliance for parity with linearized subproblem output
    C_k = float(optimizer.system_calculator.compute_actual_compliance(theta_k, A_k))

    E = float(getattr(optimizer, 'E_steel', 210e9))
    n_dof = int(getattr(optimizer, 'n_dof', 0))
    m = int(len(optimizer.geometry.elements))

    # 2) Build unit global stiffness matrices (full DOFs), then reduce via selection matrix S
    print(f"[FixedSDP] Start. n_dof will be {n_dof}; elements m={m}")
    Ki_full = []            # normalized by 1/L (E factored out)
    Ki_full_phys = []       # physical stiffness with E/L
    for i, (n1, n2) in enumerate(optimizer.geometry.elements):
        L = float(max(elen_k[i], 1e-12))
        c, s = float(edir_k[i][0]), float(edir_k[i][1])
        # Normalized unit kernel (without E): 1/L
        k_coeff_norm = 1.0 / L
        k_local_norm = k_coeff_norm * np.array([
            [c * c, c * s, -c * c, -c * s],
            [c * s, s * s, -c * s, -s * s],
            [-c * c, -c * s, c * c, c * s],
            [-c * s, -s * s, c * s, s * s],
        ], dtype=float)
        # Physical kernel with E factor
        k_local_phys = E * k_local_norm
        K = np.zeros((n_dof, n_dof), dtype=float)
        Kp = np.zeros((n_dof, n_dof), dtype=float)
        dofs = [2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1]
        for r in range(4):
            for cidx in range(4):
                K[dofs[r], dofs[cidx]] += k_local_norm[r, cidx]
                Kp[dofs[r], dofs[cidx]] += k_local_phys[r, cidx]
        if K.shape != (n_dof, n_dof):
            print(f"[FixedSDP] Ki_full[{i}] unexpected shape {K.shape} (n_dof={n_dof})")
        Ki_full.append(K)
        Ki_full_phys.append(Kp)

    free = np.asarray(optimizer.free_dofs, dtype=int)
    n_free = int(free.size)
    print(f"[FixedSDP] free_dofs count n_free={n_free}")
    # Selection matrix S to project free DOFs: K_ff = S K S^T, f_ff = S f
    S = np.zeros((n_free, n_dof), dtype=float)
    for i, dof in enumerate(free):
        if 0 <= int(dof) < n_dof:
            S[i, int(dof)] = 1.0
    S_c = cp.Constant(S)

    # 3) Load vector at frozen geometry (reduced)
    # Use initializer load vector for parity with old fixed SDP
    f_full = np.asarray(getattr(optimizer, 'load_vector', optimizer._compute_load_vector(coords_k)), dtype=float)
    f_ff = f_full[free]

    # 4) Decision vars and constraints
    A = cp.Variable(m, pos=True)
    t = cp.Variable((1, 1))

    A_min = float(getattr(optimizer, 'A_min', 1e-8))
    A_max = float(getattr(optimizer, 'A_max', 1e-2))
    lengths = np.asarray(getattr(optimizer, 'element_lengths', np.ones(m)), dtype=float)
    V_max = float(getattr(optimizer, 'volume_constraint', float(np.sum(lengths) * A_max)))

    A_min_safe = max(A_min, 1e-8)
    base_constraints = [A >= A_min_safe, A <= A_max, lengths @ A <= V_max]

    # K(A) on free DOFs
    # Build full K(A) then reduce: K_ff = S K S^T
    K_terms_full = [A[i] * cp.Constant(K) for i, K in enumerate(Ki_full)]
    if K_terms_full:
        K_full = K_terms_full[0]
        for term in K_terms_full[1:]:
            K_full = K_full + term
    else:
        K_full = cp.Constant(np.zeros((n_dof, n_dof)))
    # Symmetrize for PSD robustness
    K_full = 0.5 * (K_full + K_full.T)
    K_red = S_c @ K_full @ S_c.T
    print(f"[FixedSDP] Shapes: K_full{K_full.shape} -> K_red{K_red.shape}; S{S.shape}")
    if reg_eps and reg_eps > 0.0:
        K_red = K_red + float(reg_eps) * cp.Constant(np.eye(n_free))

    # Schur complement LMI (use stacked blocks)
    # Scale load for numerical robustness: divide by sqrt(E) to match K scaled by 1/E
    f_full = np.asarray(getattr(optimizer, 'load_vector', np.zeros(n_dof)), dtype=float).reshape((n_dof, 1)) / max(np.sqrt(E), 1.0)
    print(f"[FixedSDP] f_full shape: {f_full.shape}")
    f_ff_c = S_c @ cp.Constant(f_full)
    print(f"[FixedSDP] f_ff shape (expr): {f_ff_c.shape}; t shape: {t.shape}")
    try:
        top = cp.hstack([K_red, f_ff_c])
        bottom = cp.hstack([f_ff_c.T, t])
        M = cp.vstack([top, bottom])
        print(f"[FixedSDP] Assembled Schur block shape: {M.shape}")
    except Exception as e:
        print(f"[FixedSDP] Block assemble failure: {e}")
        M = cp.bmat([[K_red, f_ff_c], [f_ff_c.T, t]])
    # do not build problem yet; build constraints in solve block

    # 5) Solve with MOSEK
    try:
        constraints = []
        if lmi_eps and lmi_eps > 0.0:
            # Enforce strict feasibility with a small slack: M - eps*I >> 0
            Iblk = np.eye(n_free + 1)
            lmi_con = (M - float(lmi_eps) * cp.Constant(Iblk) >> 0)
        else:
            lmi_con = (M >> 0)
        # Rebuild problem with chosen LMI constraint
        constraints = base_constraints + [lmi_con]
        prob = cp.Problem(cp.Minimize(t), constraints)
        prob.solve(solver=cp.MOSEK, verbose=bool(verbose), mosek_params={
            'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-8,
            'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-8,
            'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-8,
        })
    except Exception as e:
        raise RuntimeError(f"MOSEK failed for fixed SDP: {e}") from e
    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"Fixed SDP not optimal: status={prob.status}")

    A_new = np.asarray(A.value, dtype=float)
    # Compute compliance using reconstructed K at frozen geometry (parity with old SDP verification)
    K_recon = np.zeros((n_dof, n_dof), dtype=float)
    for i in range(m):
        K_recon += float(A_new[i]) * Ki_full_phys[i]
    Kff = K_recon[np.ix_(free, free)]
    # Regularize in eval only to avoid singular solve
    lam = 0.0
    try:
        u = np.linalg.solve(Kff, f_ff)
    except np.linalg.LinAlgError:
        lam = 1e-12 * (np.mean(np.diag(Kff)) if Kff.size else 1.0)
        u = np.linalg.solve(Kff + lam * np.eye(Kff.shape[0]), f_ff)
    compliance_new = float(f_ff @ u)

    out = {
        'theta_k': np.asarray(theta_k, dtype=float).tolist(),
        'A_k': np.asarray(A_k, dtype=float).tolist(),
        'A_new': A_new.tolist(),
        'C_k': float(C_k),
        'C_new': float(compliance_new),
        # Align keys with linearized subproblem for CLI consumers
        'theta_new': np.asarray(theta_k, dtype=float).tolist(),
        't_pred': None,
        'trust_radius': float(getattr(optimizer.trust_region_manager, 'current_radius', 0.0)),
        'A_min': float(A_min),
        'A_max': float(A_max),
        'removal_threshold': float(getattr(optimizer, 'removal_threshold', 0.0)),
        'n_total': int(m),
        'n_active': int(np.sum(A_new > float(getattr(optimizer, 'removal_threshold', 0.0)))),
        'min_area': float(np.min(A_new)) if A_new.size else None,
        'p1_area': float(np.percentile(A_new, 1)) if A_new.size else None,
        'p5_area': float(np.percentile(A_new, 5)) if A_new.size else None,
        'median_area': float(np.median(A_new)) if A_new.size else None,
    }

    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        except Exception:
            pass
        tmp = save_path + ".tmp"
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        os.replace(tmp, save_path)

    return out
