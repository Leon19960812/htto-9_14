"""
Truss system initializer (clean, minimal, PolarGeometry-backed).

This module provides minimal, consistent data structures and helpers required
by the optimizer. Geometry comes from PolarGeometry; this file does not own
ground structure generation. Comments are in English; UTF-8 + LF.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


# ==========================
# Data structures
# ==========================


@dataclass
class GeometryData:
    """Geometry container used by assembly and optimizers.

    - nodes: list of [x, y]
    - elements: list of [n1, n2]
    - load_nodes: indices of nodes where loads are applied (runtime optimized set)
    Other fields are kept for compatibility with the existing codebase.
    """

    nodes: List[List[float]]
    elements: List[List[int]]
    outer_nodes: List[int]
    load_nodes: List[int]
    middle_nodes: List[int]
    inner_nodes: List[int]
    n_nodes: int
    n_elements: int
    n_dof: int


@dataclass
class MaterialData:
    E_steel: float
    rho_steel: float
    rho_water: float
    g: float
    A_min: float
    A_max: float
    removal_threshold: float


@dataclass
class LoadData:
    load_vector: np.ndarray
    base_pressure: float
    depth: float


@dataclass
class ConstraintData:
    volume_constraint: float
    volume_fraction: float
    fixed_dofs: List[int]
    free_dofs: List[int]


# ==========================
# Geometry helpers
# ==========================


class GeometryCalculator:
    """Geometry utilities used by the optimizer."""

    def compute_element_lengths(self, geometry: GeometryData) -> np.ndarray:
        nodes = np.asarray(geometry.nodes, dtype=float)
        lengths = []
        for n1, n2 in geometry.elements:
            dx = nodes[n1, 0] - nodes[n2, 0]
            dy = nodes[n1, 1] - nodes[n2, 1]
            lengths.append(float(np.hypot(dx, dy)))
        return np.asarray(lengths, dtype=float)

    def compute_element_geometry(self, coords: np.ndarray, elements: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]:
        """Return (lengths, directions) where directions are unit (c, s)."""
        coords = np.asarray(coords, dtype=float)
        lengths = []
        dirs = []
        for n1, n2 in elements:
            dx = coords[n2, 0] - coords[n1, 0]
            dy = coords[n2, 1] - coords[n1, 1]
            L = float(np.hypot(dx, dy))
            if L <= 1e-16:
                c, s = 1.0, 0.0
            else:
                c, s = float(dx / L), float(dy / L)
            lengths.append(L)
            dirs.append([c, s])
        return np.asarray(lengths, dtype=float), np.asarray(dirs, dtype=float)

    def update_node_coordinates(self, geometry: GeometryData, theta: np.ndarray, radius: float) -> np.ndarray:
        """Map 1D theta to Cartesian coordinates for optimized nodes.

        Only nodes in `geometry.load_nodes[:len(theta)]` are updated along a
        semicircle of the given radius; other nodes remain unchanged.
        Returns a new numpy array of shape (n_nodes, 2).
        """
        coords = np.asarray(geometry.nodes, dtype=float).copy()
        ids = list(geometry.load_nodes[: len(theta)]) if hasattr(geometry, "load_nodes") else list(range(len(theta)))
        for i, nid in enumerate(ids):
            ang = float(theta[i])
            coords[nid, 0] = radius * np.cos(ang)
            coords[nid, 1] = radius * np.sin(ang)
        return coords


# ==========================
# Assembly helpers
# ==========================


class StiffnessCalculator:
    """Assembly utilities for stiffness matrices."""

    def __init__(self, material_data: MaterialData):
        self.material_data = material_data

    def precompute_unit_stiffness_matrices(self, geometry: GeometryData, element_lengths: np.ndarray) -> List[np.ndarray]:
        """Return a list of per-element 4x4 unit stiffness kernels (for diagnostics)."""
        kernels: List[np.ndarray] = []
        nodes = np.asarray(geometry.nodes, dtype=float)
        for i, (n1, n2) in enumerate(geometry.elements):
            L = float(max(element_lengths[i], 1e-12))
            dx = nodes[n2, 0] - nodes[n1, 0]
            dy = nodes[n2, 1] - nodes[n1, 1]
            c = dx / L
            s = dy / L
            k = np.array([
                [c * c, c * s, -c * c, -c * s],
                [c * s, s * s, -c * s, -s * s],
                [-c * c, -c * s, c * c, c * s],
                [-c * s, -s * s, c * s, s * s],
            ], dtype=float)
            kernels.append(k)
        return kernels

    def assemble_global_stiffness(self, geometry: GeometryData, A: np.ndarray, element_lengths: np.ndarray, element_directions: np.ndarray) -> np.ndarray:
        K = np.zeros((geometry.n_dof, geometry.n_dof), dtype=float)
        E = float(self.material_data.E_steel)
        for i, (n1, n2) in enumerate(geometry.elements):
            Ai = float(max(float(A[i]), float(self.material_data.A_min)))
            L = float(max(element_lengths[i], 1e-12))
            c, s = float(element_directions[i, 0]), float(element_directions[i, 1])
            k_coeff = E * Ai / L
            k_local = k_coeff * np.array([
                [c * c, c * s, -c * c, -c * s],
                [c * s, s * s, -c * s, -s * s],
                [-c * c, -c * s, c * c, c * s],
                [-c * s, -s * s, c * s, s * s],
            ], dtype=float)
            dofs = [2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1]
            for r in range(4):
                for cidx in range(4):
                    K[dofs[r], dofs[cidx]] += k_local[r, cidx]
        return K


# ==========================
# Constraints helpers
# ==========================


class ConstraintCalculator:
    """Constraint utilities (volume and DOFs)."""

    def initialize_constraints(self, geometry: GeometryData, element_lengths: np.ndarray, volume_fraction: float, A_max: float) -> ConstraintData:
        total_length = float(np.sum(element_lengths))
        volume_constraint = float(total_length * float(A_max) * float(volume_fraction))
        return ConstraintData(
            volume_constraint=volume_constraint,
            volume_fraction=float(volume_fraction),
            fixed_dofs=[],
            free_dofs=[],
        )

    def setup_boundary_conditions(self, geometry: GeometryData) -> Tuple[List[int], List[int]]:
        # Prefer explicit support list prepared during initialization
        explicit = getattr(geometry, 'support_nodes', None)
        if explicit:
            fixed_nodes = list(dict.fromkeys(int(n) for n in explicit))
        else:
            # Backward-compatible fallback: derive from ring endpoints
            fixed_nodes: List[int] = []
            try:
                coords = np.asarray(geometry.nodes, dtype=float)

                def _endpoints(node_ids: List[int]) -> List[int]:
                    if not node_ids or len(node_ids) < 2:
                        return []
                    arr = np.asarray(node_ids, dtype=int)
                    ang = np.arctan2(coords[arr, 1], coords[arr, 0])
                    i_min = int(arr[int(np.argmin(ang))])
                    i_max = int(arr[int(np.argmax(ang))])
                    return [i_min, i_max] if i_min != i_max else [i_min]

                inner = getattr(geometry, 'inner_nodes', []) or []
                middle = getattr(geometry, 'middle_nodes', []) or []
                outer = getattr(geometry, 'outer_nodes', []) or []

                fixed_nodes.extend(_endpoints(inner))
                fixed_nodes.extend(_endpoints(middle))

                if not fixed_nodes:
                    cand = outer or (getattr(geometry, 'load_nodes', []) or [])
                    fixed_nodes.extend(_endpoints(cand))
            except Exception:
                fixed_nodes = []
            if not fixed_nodes and getattr(geometry, 'n_nodes', 0) >= 2:
                fixed_nodes = [0, geometry.n_nodes - 1]

        fixed_dofs: List[int] = []
        for nid in fixed_nodes:
            fixed_dofs.extend([2 * nid, 2 * nid + 1])
        free_dofs = [i for i in range(geometry.n_dof) if i not in fixed_dofs]
        return fixed_dofs, free_dofs


# ==========================
# Unified initializer
# ==========================


class TrussSystemInitializer:
    """Unified initializer using PolarGeometry and clean helpers."""

    def __init__(self, radius=2.0, n_sectors=12, inner_ratio=0.7, depth=50, volume_fraction=0.2, E_steel=210e9, enable_middle_layer=False, middle_layer_ratio=0.85, use_polar: bool = True, polar_config: dict = None, simple_loads: bool = False):
        # Store basic parameters
        self.radius = float(radius)
        self.n_sectors = int(n_sectors)
        self.inner_ratio = float(inner_ratio)
        self.depth = float(depth)
        self.volume_fraction = float(volume_fraction)
        self.enable_middle_layer = bool(enable_middle_layer)
        self.middle_layer_ratio = float(middle_layer_ratio)

        # Material
        self.material_data = MaterialData(
            E_steel=float(E_steel),
            rho_steel=7850.0,
            rho_water=1025.0,
            g=9.81,
            A_min=1e-5,
            A_max=1e-2,
            removal_threshold=1e-4,
        )

        # Calculators
        self.geometry_calc = GeometryCalculator()
        try:
            from .load_calculator_with_shell import LoadCalculatorWithShell
            shell_params = {
                "outer_radius": self.radius,
                "depth": self.depth,
                "thickness": 0.01,
                "n_circumferential": max(8, self.n_sectors + 1),
                "n_radial": 2,
                "E_shell": float(self.material_data.E_steel),
            }
            self.use_simple_loads = bool(simple_loads)
            self.load_calc = LoadCalculatorWithShell(
                self.material_data,
                enable_shell=(not self.use_simple_loads),
                shell_params=shell_params,
                simple_mode=self.use_simple_loads,
            )
        except Exception as e:
            raise ImportError(f"LoadCalculatorWithShell is required: {e}")
        self.stiffness_calc = StiffnessCalculator(self.material_data)
        self.constraint_calc = ConstraintCalculator()

        # Build geometry from PolarGeometry (no fallback)
        self._initialize_from_polar(polar_config or {})

    def _initialize_from_polar(self, polar_cfg: dict):
        from .polar_geometry import PolarConfig as _PolarConfig, PolarGeometry as _PolarGeometry

        rings = polar_cfg.get("rings") or [
            {"radius": self.radius, "n_nodes": self.n_sectors + 1, "type": "outer"},
            {"radius": self.radius * self.inner_ratio, "n_nodes": self.n_sectors + 1, "type": "inner"},
        ]
        pg = _PolarGeometry(_PolarConfig(rings=rings))
        # Expose PolarGeometry instance for downstream optimizer usage
        self.polar_geometry = pg

        nodes_xy = [[float(n.x), float(n.y)] for n in pg.nodes]
        elements = [[int(i), int(j)] for (i, j) in pg.connections]

        # Node groups by type (compatibility lists)
        outer_nodes = [n.id for n in pg.nodes if getattr(n, "node_type", "") == "outer"]
        middle_nodes = [n.id for n in pg.nodes if getattr(n, "node_type", "") == "middle"]
        inner_nodes = [n.id for n in pg.nodes if getattr(n, "node_type", "") == "inner"]

        # Load nodes: prefer config-provided; otherwise, use outer_nodes
        cfg_load = getattr(pg, "config", None)
        if cfg_load and getattr(cfg_load, "load_nodes", None):
            load_nodes = list(cfg_load.load_nodes)
        else:
            load_nodes = list(outer_nodes)

        # Determine support nodes: all ring endpoints except the ring carrying loads
        coords_arr = np.asarray(nodes_xy, dtype=float)
        ring_groups = {
            "outer": outer_nodes,
            "middle": middle_nodes,
            "inner": inner_nodes,
        }

        load_set = set(load_nodes)

        def _endpoints(ids: List[int]) -> List[int]:
            if not ids or len(ids) < 2:
                return []
            arr = np.asarray(ids, dtype=int)
            ang = np.arctan2(coords_arr[arr, 1], coords_arr[arr, 0])
            imin = int(arr[int(np.argmin(ang))])
            imax = int(arr[int(np.argmax(ang))])
            return [imin, imax] if imin != imax else [imin]

        support_nodes: List[int] = []
        for ring_name, ids in ring_groups.items():
            if not ids:
                continue
            ring_set = set(ids)
            overlap = len(ring_set & load_set)
            ratio = float(overlap) / float(len(ring_set)) if ring_set else 0.0
            if ratio > 0.5:  # treat this ring as the load ring
                continue
            support_nodes.extend(_endpoints(ids))

        # Remove potential duplicates and ensure supports are not load nodes
        support_nodes = [nid for nid in dict.fromkeys(sorted(support_nodes)) if nid not in load_set]
        pg.set_support_nodes(support_nodes)

        n_nodes = len(nodes_xy)
        self.geometry = GeometryData(
            nodes=nodes_xy,
            elements=elements,
            outer_nodes=list(outer_nodes),
            load_nodes=list(load_nodes),
            inner_nodes=list(inner_nodes),
            middle_nodes=list(middle_nodes),
            n_nodes=n_nodes,
            n_elements=len(elements),
            n_dof=2 * n_nodes,
        )
        # Persist support node list for downstream consumers
        self.geometry.support_nodes = list(support_nodes)

        # Derived/assembled data
        self.element_lengths = self.geometry_calc.compute_element_lengths(self.geometry)
        self.load_data = self.load_calc.compute_hydrostatic_loads(self.geometry, self.depth, self.radius, np.asarray(self.geometry.nodes, dtype=float))
        self.unit_stiffness_matrices = self.stiffness_calc.precompute_unit_stiffness_matrices(self.geometry, self.element_lengths)
        self.constraint_data = self.constraint_calc.initialize_constraints(self.geometry, self.element_lengths, self.volume_fraction, self.material_data.A_max)
        fixed_dofs, free_dofs = self.constraint_calc.setup_boundary_conditions(self.geometry)
        self.constraint_data.fixed_dofs = fixed_dofs
        self.constraint_data.free_dofs = free_dofs

        self._setup_legacy_attributes()
        self._print_initialization_info()

    def _setup_legacy_attributes(self):
        # Geometry snapshot
        self.nodes = self.geometry.nodes
        self.elements = self.geometry.elements
        self.outer_nodes = self.geometry.outer_nodes
        self.load_nodes = getattr(self.geometry, "load_nodes", self.geometry.outer_nodes)
        self.inner_nodes = self.geometry.inner_nodes
        self.middle_nodes = self.geometry.middle_nodes
        self.support_nodes = getattr(self.geometry, 'support_nodes', [])
        self.n_nodes = self.geometry.n_nodes
        self.n_elements = self.geometry.n_elements
        self.n_dof = self.geometry.n_dof

        # Material
        self.E_steel = self.material_data.E_steel
        self.rho_steel = self.material_data.rho_steel
        self.rho_water = self.material_data.rho_water
        self.g = self.material_data.g
        self.A_min = self.material_data.A_min
        self.A_max = self.material_data.A_max
        self.removal_threshold = self.material_data.removal_threshold

        # Loads
        self.load_vector = self.load_data.load_vector
        self.base_pressure = self.load_data.base_pressure

        # Constraints
        self.volume_constraint = self.constraint_data.volume_constraint
        self.fixed_dofs = self.constraint_data.fixed_dofs
        self.free_dofs = self.constraint_data.free_dofs

        # Derived radii
        self.inner_radius = self.radius * self.inner_ratio
        self.middle_radius = (self.radius * self.middle_layer_ratio) if self.enable_middle_layer else None

        # Optimizer state placeholders
        self.final_areas = None
        self.final_compliance = None
        self.verification_passed = None
        self.final_angles = None
        self.current_angles = None
        self.current_areas = None
        self.current_compliance = None

    def _print_initialization_info(self):
        print(f"Structure: R={self.radius} m, Sectors={self.n_sectors}, Inner ratio={self.inner_ratio}")
        print(f"Depth: {self.depth} m, Pressure: {self.base_pressure/1000:.1f} kPa")
        print(f"Volume fraction: {self.volume_fraction:.3f}")
        print(f"Nodes: {self.n_nodes}, Elements: {self.n_elements}, DOF: {self.n_dof}")
        ln = getattr(self.geometry, 'load_nodes', [])
        print(f"Load nodes: {len(ln)}")
        # Report actual support nodes derived from fixed DOFs (not load endpoints)
        try:
            sup_nodes = sorted(set(int(d // 2) for d in self.fixed_dofs))
            print(f"  Support nodes: {sup_nodes}")
            if getattr(self, 'support_nodes', None):
                print(f"  Declared supports: {self.support_nodes}")
        except Exception:
            pass
        print(f"  Fixed DOFs: {len(self.fixed_dofs)}  Free DOFs: {len(self.free_dofs)}")

    # Delegation to node merger utilities
    def group_nodes_by_radius(self, theta: np.ndarray, radius: float = None) -> List[List[int]]:
        from .node_merger import NodeMerger
        return NodeMerger(self.geometry, self.radius, self.constraint_calc).group_nodes_by_radius(theta, radius)

    def merge_node_groups(self, theta: np.ndarray, A: np.ndarray, merge_groups: List[List[int]]):
        from .node_merger import NodeMerger
        result = NodeMerger(self.geometry, self.radius, self.constraint_calc).merge_node_groups(theta, A, merge_groups)
        if result.structure_modified:
            self.geometry = result.geometry_updated
            self.nodes = self.geometry.nodes
            self.elements = self.geometry.elements
            self.load_nodes = getattr(self.geometry, "load_nodes", self.geometry.outer_nodes)
            self.inner_nodes = self.geometry.inner_nodes
            self.n_nodes = self.geometry.n_nodes
            self.n_elements = self.geometry.n_elements
            self.n_dof = self.geometry.n_dof
            self.fixed_dofs, self.free_dofs = self.constraint_calc.setup_boundary_conditions(self.geometry)
            self.element_lengths = self.geometry_calc.compute_element_lengths(self.geometry)
            self.unit_stiffness_matrices = self.stiffness_calc.precompute_unit_stiffness_matrices(self.geometry, self.element_lengths)
        return {
            "merged_pairs": result.merged_pairs,
            "removed_members": result.removed_members,
            "theta_updated": result.theta_updated,
            "A_updated": result.A_updated,
            "geometry_updated": result.geometry_updated,
            "structure_modified": result.structure_modified,
        }
