"""
Collinear member cleanup (node melt) for final structures.

Greedy post-processing that replaces two collinear members i–j and j–k by a
single member i–k when node j has valence=2 and is not in a protected whitelist.
Intended for visualization/export after optimization; does not mutate the
optimizer in-place unless explicitly used to do so.
"""
from __future__ import annotations

from typing import Iterable, List, Tuple, Dict, Set
from types import SimpleNamespace

import numpy as np


def _edge_key(i: int, j: int) -> Tuple[int, int]:
    return (i, j) if i <= j else (j, i)


def _build_adjacency(elements: Iterable[Tuple[int, int]], n_nodes: int) -> List[Set[int]]:
    adj: List[Set[int]] = [set() for _ in range(n_nodes)]
    for a, b in elements:
        if 0 <= a < n_nodes and 0 <= b < n_nodes and a != b:
            adj[a].add(b)
            adj[b].add(a)
    return adj


def melt_collinear(
    nodes_xy: np.ndarray,
    elements: List[Tuple[int, int]],
    areas: np.ndarray,
    whitelist_nodes: Iterable[int] = (),
    tol: float = 1e-9,
    min_len: float = 1e-9,
    mode: str = 'volume',  # 'volume' or 'stiffness'
    active_threshold: float | None = None,  # if set, build adjacency using only A > active_threshold
    a_max: float | None = None,  # if set, cap any created/merged area by A_max
) -> Tuple[List[Tuple[int, int]], np.ndarray, Dict[int, Tuple[int, int]]]:
    """Return cleaned (elements, areas) and a map of merged nodes.

    - nodes_xy: (n,2) coordinates used to test collinearity/lengths
    - elements: list of (i,j)
    - areas: per-element areas aligned with elements
    - whitelist_nodes: nodes that must not be removed (supports, loads, symmetry-pinned)
    - tol: collinearity tolerance relative to lengths (|cross| <= tol*|v1|*|v2|)
    - min_len: minimum segment length to consider (absolute)
    - mode: 'volume' -> A_new = (Aij*Lij + Ajk*Ljk)/Lik; 'stiffness' -> A_new = Lik / (Lij/Aij + Ljk/Ajk)
    """
    n_nodes = int(nodes_xy.shape[0])
    whitelist: Set[int] = set(int(x) for x in (whitelist_nodes or []))

    # Work on copies
    elems = [(_edge_key(int(i), int(j))) for (i, j) in elements]
    A = np.asarray(areas, dtype=float).copy()

    # Map edge -> index for fast lookup
    def rebuild_index() -> Dict[Tuple[int, int], int]:
        return {tuple(elems[k]): k for k in range(len(elems))}

    merged_nodes: Dict[int, Tuple[int, int]] = {}
    changed = True
    while changed:
        changed = False
        # Build adjacency: optionally only with "active" members (A > active_threshold)
        if active_threshold is not None:
            try:
                A_arr = np.asarray(A, dtype=float).reshape(-1)
                active_elems = [tuple(elems[k]) for k in range(len(elems)) if float(A_arr[k]) > float(active_threshold)]
            except Exception:
                active_elems = list(elems)
            adj = _build_adjacency(active_elems, n_nodes)
        else:
            adj = _build_adjacency(elems, n_nodes)
        eidx = rebuild_index()

        for j in range(n_nodes):
            if j in whitelist:
                continue
            nbrs = list(adj[j])
            if len(nbrs) != 2:
                continue
            i, k = int(nbrs[0]), int(nbrs[1])
            # geometry
            xi = nodes_xy[i]
            xj = nodes_xy[j]
            xk = nodes_xy[k]
            v1 = xi - xj
            v2 = xk - xj
            Lij = float(np.hypot(v1[0], v1[1]))
            Ljk = float(np.hypot(v2[0], v2[1]))
            if not (Lij > min_len and Ljk > min_len):
                continue
            # collinearity + opposite direction
            cross = float(abs(v1[0]*v2[1] - v1[1]*v2[0]))
            if cross > tol * Lij * Ljk:
                continue
            if float(v1[0]*v2[0] + v1[1]*v2[1]) >= 0.0:
                continue
            # (i,k) may already exist. If it exists, we'll merge area into it instead of skipping.
            ik = _edge_key(i, k)
            # retrieve element indices and areas
            ij = _edge_key(i, j)
            jk = _edge_key(j, k)
            if ij not in eidx or jk not in eidx:
                continue
            idx_ij = eidx[ij]
            idx_jk = eidx[jk]
            Aij = float(max(A[idx_ij], 0.0))
            Ajk = float(max(A[idx_jk], 0.0))
            Lik = float(np.hypot(*(xk - xi)))
            if Lik <= min_len:
                continue
            if mode == 'stiffness' and Aij > 0.0 and Ajk > 0.0:
                Anew = Lik / (Lij / Aij + Ljk / Ajk)
            else:
                Anew = (Aij * Lij + Ajk * Ljk) / Lik
            # Cap by A_max if provided
            if a_max is not None:
                try:
                    Anew = min(float(Anew), float(a_max))
                except Exception:
                    pass
            # apply changes
            if ik in eidx:
                # merge into existing i-k member (parallel addition of area)
                idx_ik = eidx[ik]
                merged_val = float(max(A[idx_ik], 0.0)) + float(max(Anew, 0.0))
                if a_max is not None:
                    try:
                        merged_val = min(merged_val, float(a_max))
                    except Exception:
                        pass
                A[idx_ik] = merged_val
                # remove old two members
                for idx in sorted([idx_ij, idx_jk], reverse=True):
                    elems.pop(idx)
                    A = np.delete(A, idx)
            else:
                # remove old, then add new i-k
                for idx in sorted([idx_ij, idx_jk], reverse=True):
                    elems.pop(idx)
                    A = np.delete(A, idx)
                elems.append(ik)
                A = np.append(A, Anew)
            merged_nodes[j] = (i, k)
            changed = True
            break  # restart after structural change

    return elems, A, merged_nodes


def build_optimizer_view(base, elements: List[Tuple[int, int]], final_areas: np.ndarray):
    """Create a lightweight view of optimizer with replaced elements/areas for plotting/export."""
    view = SimpleNamespace()
    view.__dict__.update(base.__dict__)
    view.elements = [list(map(int, e)) for e in elements]
    view.n_elements = len(view.elements)
    view.final_areas = np.asarray(final_areas, dtype=float)
    return view
