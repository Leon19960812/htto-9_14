"""
Node merging utilities (unified 1D theta).
This module provides distance-based node grouping and merging without any layer concept.
"""

from dataclasses import dataclass
from typing import List, Tuple, Set
import numpy as np


@dataclass
class MergePlan:
    node_merges: List[Tuple[int, List[int]]]
    element_removals: Set[int]
    target_coords: dict
    target_angles: dict


@dataclass
class MergeResult:
    merged_pairs: List[Tuple[int, int]]
    removed_members: List[int]
    theta_updated: np.ndarray
    A_updated: np.ndarray
    geometry_updated: 'GeometryData'
    structure_modified: bool


class NodeMerger:
    def __init__(self, geometry, radius, constraint_calc):
        self.geometry = geometry
        self.radius = float(radius)
        self.constraint_calc = constraint_calc

    def group_nodes_by_radius(self, theta: np.ndarray, radius: float = None) -> List[List[int]]:
        if radius is None:
            radius = 0.5
        # Determine optimized node ids from load_nodes (no layer semantics)
        if hasattr(self.geometry, 'load_nodes') and self.geometry.load_nodes:
            node_ids = list(self.geometry.load_nodes[:len(theta)])
        else:
            node_ids = list(range(min(len(theta), getattr(self.geometry, 'n_nodes', len(theta)))))

        groups: List[List[int]] = []
        used: Set[int] = set()

        # Use current radius and theta to build positions along the ring for the selected nodes
        coords = {}
        for i, nid in enumerate(node_ids):
            ang = float(theta[i]) if i < len(theta) else 0.0
            coords[nid] = (self.radius * np.cos(ang), self.radius * np.sin(ang))

        for i, nid_i in enumerate(node_ids):
            if nid_i in used:
                continue
            xi, yi = coords[nid_i]
            group = [nid_i]
            for j, nid_j in enumerate(node_ids):
                if i == j or nid_j in used:
                    continue
                xj, yj = coords[nid_j]
                if np.hypot(xi - xj, yi - yj) < radius:
                    group.append(nid_j)
            if len(group) > 1:
                groups.append(group)
                used.update(group)
        return groups

    def merge_node_groups(self, theta: np.ndarray, A: np.ndarray, merge_groups: List[List[int]]) -> MergeResult:
        if not merge_groups:
            return self._create_no_change_result(theta, A)
        plan = self._create_merge_plan(merge_groups, theta, A)
        result = self._execute_merge_plan(plan, theta, A)
        self._validate_merge_result(result)
        return result

    def _create_merge_plan(self, merge_groups, theta, A):
        plan = MergePlan(node_merges=[], element_removals=set(), target_coords={}, target_angles={})
        # Determine optimized node ids list
        if hasattr(self.geometry, 'load_nodes') and self.geometry.load_nodes:
            node_ids = list(self.geometry.load_nodes[:len(theta)])
        else:
            node_ids = list(range(min(len(theta), getattr(self.geometry, 'n_nodes', len(theta)))))

        for group in merge_groups:
            # choose representative as the minimal id for determinism
            target_node = min(group)
            merged_nodes = [n for n in group if n != target_node]
            w_angle = self._compute_weighted_angle(group, node_ids, theta, A)
            plan.node_merges.append((target_node, merged_nodes))
            plan.target_angles[target_node] = w_angle
            plan.target_coords[target_node] = [self.radius * np.cos(w_angle), self.radius * np.sin(w_angle)]
            for m in merged_nodes:
                for eid, (n1, n2) in enumerate(self.geometry.elements):
                    if n1 == m or n2 == m:
                        plan.element_removals.add(eid)
        return plan

    def _execute_merge_plan(self, plan: MergePlan, theta: np.ndarray, A: np.ndarray) -> MergeResult:
        import copy
        geometry = copy.deepcopy(self.geometry)

        # 1) update target node coordinates
        for nid, xy in plan.target_coords.items():
            if 0 <= nid < len(geometry.nodes):
                geometry.nodes[nid] = xy

        # 2) build mapping old->new (remove merged nodes)
        remove_set = set([m for _, ms in plan.node_merges for m in ms])
        old_to_new = {}
        new_nodes = []
        for old_id, xy in enumerate(self.geometry.nodes):
            if old_id in remove_set:
                continue
            old_to_new[old_id] = len(new_nodes)
            new_nodes.append(xy)
        geometry.nodes = new_nodes
        geometry.n_nodes = len(new_nodes)
        geometry.n_dof = 2 * geometry.n_nodes

        # 3) rebuild elements and areas (merge parallels by summing areas)
        edge_map = {}
        new_elements = []
        new_A = []
        for eid, (n1, n2) in enumerate(self.geometry.elements):
            if n1 in remove_set or n2 in remove_set:
                continue
            nn1 = old_to_new.get(n1, None)
            nn2 = old_to_new.get(n2, None)
            if nn1 is None or nn2 is None or nn1 == nn2:
                continue
            key = (nn1, nn2) if nn1 < nn2 else (nn2, nn1)
            if key in edge_map:
                idx = edge_map[key]
                new_A[idx] += A[eid] if eid < len(A) else 0.0
            else:
                edge_map[key] = len(new_elements)
                new_elements.append([key[0], key[1]])
                new_A.append(A[eid] if eid < len(A) else 0.0)
        geometry.elements = new_elements
        geometry.n_elements = len(new_elements)

        # 4) rebuild DOFs via constraint calculator
        fixed_dofs, free_dofs = self.constraint_calc.setup_boundary_conditions(geometry)
        geometry.fixed_dofs = fixed_dofs
        geometry.free_dofs = free_dofs

        # 5) remap theta (1D) for optimized nodes list
        if hasattr(self.geometry, 'load_nodes') and self.geometry.load_nodes:
            old_ids = list(self.geometry.load_nodes[:len(theta)])
            # Remap geometry.load_nodes through old_to_new
            new_load_nodes = []
            for i in getattr(geometry, 'load_nodes', []):
                if i in old_to_new:
                    new_load_nodes.append(old_to_new[i])
            geometry.load_nodes = new_load_nodes
            new_ids = [i for i in new_load_nodes]
        else:
            old_ids = list(range(min(len(theta), self.geometry.n_nodes)))
            new_ids = list(range(min(len(theta), geometry.n_nodes)))
        theta_updated = self._remap_layer_theta(theta, old_ids, new_ids, plan, old_to_new)

        return MergeResult(
            merged_pairs=[(t, m) for t, ms in plan.node_merges for m in ms],
            removed_members=sorted(list(plan.element_removals)),
            theta_updated=theta_updated,
            A_updated=np.array(new_A, dtype=float),
            geometry_updated=geometry,
            structure_modified=len(plan.node_merges) > 0,
        )

    def _remap_layer_theta(self, layer_theta: np.ndarray, old_ids: List[int], new_ids: List[int], plan: MergePlan, old_to_new: dict) -> np.ndarray:
        if not old_ids or not new_ids:
            return np.array([], dtype=float)

        # representative mapping new_id -> weighted angle
        rep_angles = {}
        for target, merged in plan.node_merges:
            group = [target] + merged
            # compute weighted angle from old theta for nodes that were in optimized set
            total = 0.0
            accum = 0.0
            for nid in group:
                if nid in old_ids:
                    idx = old_ids.index(nid)
                    w = 1.0
                    total += w
                    accum += w * (layer_theta[idx] if idx < len(layer_theta) else 0.0)
            if total > 0:
                rep_angles[old_to_new[target]] = accum / total

        theta_new = []
        for nid in new_ids:
            if nid in rep_angles:
                theta_new.append(rep_angles[nid])
            else:
                # find corresponding old id if exists
                candidates = [oid for oid, nn in old_to_new.items() if nn == nid]
                angle_val = 0.0
                for oid in candidates:
                    if oid in old_ids:
                        idx = old_ids.index(oid)
                        if idx < len(layer_theta):
                            angle_val = float(layer_theta[idx])
                            break
                theta_new.append(angle_val)
        return np.array(theta_new, dtype=float)

    def _compute_weighted_angle(self, group: List[int], optimized_ids: List[int], theta: np.ndarray, A: np.ndarray) -> float:
        total = 0.0
        accum = 0.0
        for nid in group:
            # weight by sum of incident areas
            w = 0.0
            for eid, (n1, n2) in enumerate(self.geometry.elements):
                if n1 == nid or n2 == nid:
                    w += (A[eid] if eid < len(A) else 0.0)
            if w <= 0:
                continue
            if nid in optimized_ids:
                idx = optimized_ids.index(nid)
                if idx < len(theta):
                    total += w
                    accum += w * float(theta[idx])
        return (accum / total) if total > 0 else 0.0

    def _create_no_change_result(self, theta: np.ndarray, A: np.ndarray) -> MergeResult:
        return MergeResult(
            merged_pairs=[],
            removed_members=[],
            theta_updated=theta.copy(),
            A_updated=A.copy(),
            geometry_updated=self.geometry,
            structure_modified=False,
        )
