
"""
Node merging utilities (unified 1D theta).
This module provides distance-based node grouping and merging without any layer concept.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

MERGE_THRESHOLD_DEFAULT = 0.1


@dataclass
class MergeResult:
    merged_pairs: List[Tuple[int, int]]
    removed_members: List[int]
    theta_updated: np.ndarray
    theta_ids_updated: List[int]
    A_updated: np.ndarray
    geometry_updated: 'GeometryData'
    structure_modified: bool


class NodeMerger:
    """Utility to detect and merge near-duplicate nodes based on Euclidean distance."""

    def __init__(self, geometry, constraint_calc, merge_threshold: float = MERGE_THRESHOLD_DEFAULT, a_max: Optional[float] = None):
        self.geometry = geometry
        self.constraint_calc = constraint_calc
        self.merge_threshold = float(merge_threshold)
        self.a_max = None if a_max is None else float(a_max)

    def find_merge_groups(
        self,
        theta_ids: List[int],
        merge_threshold: Optional[float] = None,
        active_areas: Optional[np.ndarray] = None,
        removal_threshold: Optional[float] = None,
    ) -> List[List[int]]:
        """Identify node groups whose connecting members are shorter than the threshold."""
        thr = float(merge_threshold) if merge_threshold is not None else self.merge_threshold
        active_areas = None if active_areas is None else np.asarray(active_areas, dtype=float)
        coords = np.asarray(getattr(self.geometry, 'nodes', []), dtype=float)
        elements = getattr(self.geometry, 'elements', []) or []
        if coords.size == 0 or len(elements) == 0:
            return []

        supports = set(self._get_support_nodes())
        theta_id_set = set(int(nid) for nid in theta_ids)

        def _edge_is_candidate(eid: int) -> bool:
            # Length-only criterion; areas are ignored per user guidance.
            return True

        parent: Dict[int, int] = {}

        def find(x: int) -> int:
            parent.setdefault(x, x)
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for eid, (n1, n2) in enumerate(elements):
            if not _edge_is_candidate(eid):
                continue
            if n1 == n2:
                continue
            p1 = coords[n1]
            p2 = coords[n2]
            length = float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))
            if length < thr:
                union(int(n1), int(n2))

        clusters: Dict[int, Set[int]] = {}
        for node in list(parent.keys()):
            root = find(node)
            clusters.setdefault(root, set()).add(node)

        groups: List[List[int]] = []
        for members in clusters.values():
            if len(members) <= 1:
                continue
            members_sorted = sorted(members)
            support_members = [nid for nid in members_sorted if nid in supports]
            if support_members:
                target = support_members[0]
            else:
                theta_members = [nid for nid in members_sorted if nid in theta_id_set]
                target = theta_members[0] if theta_members else members_sorted[0]
            rest = [nid for nid in members_sorted if nid != target]
            if rest:
                groups.append([target] + rest)
        return groups

    def merge_node_groups(
        self,
        theta: np.ndarray,
        theta_ids: List[int],
        areas: np.ndarray,
        merge_groups: List[List[int]],
    ) -> MergeResult:
        """Execute node merging given pre-computed merge groups."""
        if not merge_groups:
            return MergeResult(
                merged_pairs=[],
                removed_members=[],
                theta_updated=np.asarray(theta, dtype=float).copy(),
                theta_ids_updated=list(theta_ids),
                A_updated=np.asarray(areas, dtype=float).copy(),
                geometry_updated=self.geometry,
                structure_modified=False,
            )

        import copy

        geometry = copy.deepcopy(self.geometry)
        areas = np.asarray(areas, dtype=float) if areas is not None else np.zeros(len(self.geometry.elements), dtype=float)
        coords_old = np.asarray(self.geometry.nodes, dtype=float)
        support_nodes = set(self._get_support_nodes())

        target_coords: Dict[int, np.ndarray] = {}
        remove_nodes: Set[int] = set()
        merged_pairs: List[Tuple[int, int]] = []
        removed_members: List[int] = []
        representative_map: Dict[int, int] = {}

        for group in merge_groups:
            if len(group) < 2:
                continue
            target = int(group[0])
            members = [int(nid) for nid in group[1:]]
            representative_map[target] = target
            for member in members:
                representative_map[member] = target
            remove_nodes.update(members)
            merged_pairs.extend((target, member) for member in members)
            removed_members.extend(members)

            if target in support_nodes and 0 <= target < coords_old.shape[0]:
                target_coords[target] = coords_old[target]
            else:
                pts = []
                for nid in group:
                    if 0 <= nid < coords_old.shape[0]:
                        pts.append(coords_old[nid])
                if pts:
                    new_coord = np.mean(np.asarray(pts, dtype=float), axis=0)
                    target_coords[target] = new_coord

        remove_nodes = set(remove_nodes)

        # Ensure nodes not in representative_map default to themselves
        total_nodes = len(self.geometry.nodes)
        for nid in range(total_nodes):
            representative_map.setdefault(nid, nid)

        old_to_new: Dict[int, int] = {}
        new_nodes: List[List[float]] = []
        for old_id, xy in enumerate(self.geometry.nodes):
            rep = representative_map.get(old_id, old_id)
            if rep != old_id:
                continue
            new_xy = target_coords.get(old_id, np.asarray(xy, dtype=float))
            old_to_new[old_id] = len(new_nodes)
            new_nodes.append([float(new_xy[0]), float(new_xy[1])])

        # 为被移除节点建立到代表节点的新索引映射
        for old_id in range(total_nodes):
            if old_id in old_to_new:
                continue
            rep = representative_map.get(old_id, old_id)
            mapped_idx = old_to_new.get(rep)
            if mapped_idx is not None:
                old_to_new[old_id] = mapped_idx

        geometry.nodes = new_nodes
        geometry.n_nodes = len(new_nodes)
        geometry.n_dof = 2 * geometry.n_nodes

        edge_map: Dict[Tuple[int, int], int] = {}
        new_elements: List[List[int]] = []
        new_areas: List[float] = []
        for eid, (n1, n2) in enumerate(self.geometry.elements):
            nn1 = old_to_new.get(n1)
            nn2 = old_to_new.get(n2)
            if nn1 is None or nn2 is None or nn1 == nn2:
                continue
            key = (nn1, nn2) if nn1 < nn2 else (nn2, nn1)
            area_val = float(areas[eid]) if eid < len(areas) else 0.0
            if key in edge_map:
                idx = edge_map[key]
                new_areas[idx] += area_val
                if self.a_max is not None:
                    new_areas[idx] = min(new_areas[idx], self.a_max)
            else:
                edge_map[key] = len(new_elements)
                new_elements.append([key[0], key[1]])
                if self.a_max is not None:
                    new_areas.append(min(area_val, self.a_max))
                else:
                    new_areas.append(area_val)

        geometry.elements = new_elements
        geometry.n_elements = len(new_elements)

        self._remap_geometry_sets(geometry, old_to_new)

        fixed_dofs, free_dofs = self.constraint_calc.setup_boundary_conditions(geometry)
        geometry.fixed_dofs = fixed_dofs
        geometry.free_dofs = free_dofs

        theta_ids_new: List[int] = []
        for nid in theta_ids:
            mapped = old_to_new.get(int(nid))
            if mapped is None:
                continue
            if mapped not in theta_ids_new:
                theta_ids_new.append(mapped)

        coords_new = np.asarray(geometry.nodes, dtype=float)
        if theta_ids_new:
            theta_vals = np.arctan2(coords_new[theta_ids_new, 1], coords_new[theta_ids_new, 0])
            order = np.argsort(theta_vals)
            theta_updated = theta_vals[order]
            theta_ids_updated = [theta_ids_new[i] for i in order]
        else:
            theta_updated = np.asarray([], dtype=float)
            theta_ids_updated = []

        return MergeResult(
            merged_pairs=merged_pairs,
            removed_members=removed_members,
            theta_updated=theta_updated,
            theta_ids_updated=theta_ids_updated,
            A_updated=np.asarray(new_areas, dtype=float),
            geometry_updated=geometry,
            structure_modified=len(merged_pairs) > 0,
        )

    def _get_support_nodes(self) -> List[int]:
        import numpy as _np

        try:
            explicit = getattr(self.geometry, 'support_nodes', None)
            if explicit:
                return list(dict.fromkeys(int(n) for n in explicit))
            coords = _np.asarray(getattr(self.geometry, 'nodes', []), dtype=float)
            inner = getattr(self.geometry, 'inner_nodes', []) or []
            outer = getattr(self.geometry, 'outer_nodes', []) or []
            if len(inner) >= 2:
                cand = _np.asarray(inner, dtype=int)
            elif len(outer) >= 2:
                cand = _np.asarray(outer, dtype=int)
            else:
                ln = getattr(self.geometry, 'load_nodes', []) or []
                cand = _np.asarray(ln, dtype=int) if len(ln) >= 2 else _np.asarray([], dtype=int)
            if cand.size >= 2 and coords.size:
                ang = _np.arctan2(coords[cand, 1], coords[cand, 0])
                i_min = int(cand[int(_np.argmin(ang))])
                i_max = int(cand[int(_np.argmax(ang))])
                return [i_min, i_max] if i_min != i_max else [i_min]
        except Exception:
            pass
        n = int(getattr(self.geometry, 'n_nodes', 0))
        return [0, n - 1] if n >= 2 else []

    def _incident_area_weight(self, node_id: int, areas: np.ndarray) -> float:
        total = 0.0
        if areas is None:
            return 0.0
        for eid, (n1, n2) in enumerate(getattr(self.geometry, 'elements', []) or []):
            if n1 == node_id or n2 == node_id:
                if eid < len(areas):
                    total += float(areas[eid])
        return total

    def _remap_geometry_sets(self, geometry, mapping: Dict[int, int]) -> None:
        def _remap(indices: Optional[List[int]]) -> List[int]:
            if not indices:
                return []
            seen: Set[int] = set()
            result: List[int] = []
            for nid in indices:
                mapped = mapping.get(int(nid))
                if mapped is None or mapped in seen:
                    continue
                seen.add(mapped)
                result.append(mapped)
            return result

        for attr in ('load_nodes', 'inner_nodes', 'outer_nodes', 'middle_nodes', 'support_nodes'):
            if hasattr(geometry, attr):
                setattr(geometry, attr, _remap(getattr(geometry, attr, [])))

