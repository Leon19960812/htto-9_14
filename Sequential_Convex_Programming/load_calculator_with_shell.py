"""
扩展的载荷计算器 - 集成壳体FEA
  
Date: 2025/9/3
"""
import numpy as np
import math
from typing import List, Optional, Dict, Tuple, Any
from pathlib import Path
from .shell_fea_2d import Shell2DFEA, ShellMaterialData

class LoadCalculatorWithShell:
    """
        载荷计算器
    """
    
    def __init__(self, material_data, enable_shell=True, shell_params=None,
                 simple_mode: bool = False, filter_params: Optional[Dict[str, Any]] = None):
        """
        初始化载荷计算器
        
        Parameters:
        -----------
        material_data : MaterialData
            桁架材料数据（来自你的原始代码）
        enable_shell : bool
            是否启用壳体计算
        shell_params : dict
            壳体参数配置
        """
        self.material_data = material_data
        self.enable_shell = enable_shell
        self.simple_mode = bool(simple_mode)
        
        # 壳体FEA模块
        self.shell_fea = None
        self.debug_file: Optional[Path] = None
        self._debug_prev_loads: Dict[int, Tuple[float, float]] = {}
        self.current_load_vector: Optional[np.ndarray] = None
        self._last_raw_load_vector: Optional[np.ndarray] = None
        self._load_history: List[np.ndarray] = []
        self.load_filter_config: Dict[str, Any] = {}
        self.enable_fir_filter: bool = False
        self.fir_window: int = 1
        self.fir_min_history: int = 1
        self._fir_base_weights: Optional[np.ndarray] = None
        # 在解析灵敏度开发阶段默认关闭 FIR 滤波，避免对载荷历史的依赖
        self._force_disable_filter: bool = True
        self._last_shell_load_jacobian: Optional[np.ndarray] = None

        if enable_shell and shell_params and not self.simple_mode:
            self._initialize_shell_fea(shell_params)

        # 载荷滤波配置：优先使用 shell_params['load_filter']，可被显式参数覆盖
        filter_cfg: Optional[Dict[str, Any]] = None
        if isinstance(shell_params, dict):
            # 允许外部显式重新启用 FIR；默认保持关闭
            disable_flag = shell_params.get('disable_fir_filter')
            if disable_flag is not None:
                self._force_disable_filter = bool(disable_flag)
            filter_cfg = shell_params.get('load_filter') or shell_params.get('fir_filter')
        if filter_params:
            merged = dict(filter_cfg or {})
            merged.update(filter_params)
            filter_cfg = merged
        self._init_filter(filter_cfg)
        
        print(f"LoadCalculator initialized:")
        mode_str = (
            'Shell' if (self.enable_shell and self.shell_fea and not self.simple_mode) else (
                'SimpleHydrostatic' if self.simple_mode else 'Disabled')
        )
        print(f"  Mode: {mode_str}")
    
    def _initialize_shell_fea(self, shell_params):
        """初始化壳体FEA模块"""
        try:
            # 提取参数
            thickness = shell_params.get('thickness', 0.1)
            outer_radius = shell_params.get('outer_radius', 5.0)
            depth = shell_params.get('depth', 50.0)
            thickness = shell_params.get('thickness', 0.1)
            n_circumferential = shell_params.get('n_circumferential', 20)
            n_radial = shell_params.get('n_radial', 2)  # 改为2层，更薄
            
            # 壳体材料（高刚度）
            shell_material = ShellMaterialData(
                E=shell_params.get('E_shell', 210e9),  # 默认210 GPa
                nu=shell_params.get('nu', 0.3),
                thickness=thickness,
                rho_water=self.material_data.rho_water,
                g=self.material_data.g
            )
            
            # 创建壳体FEA模块
            # Ensure that shell inner wall coincides with truss radius when possible
            self.shell_fea = Shell2DFEA(
                outer_radius=float(outer_radius),
                depth=depth,
                n_circumferential=n_circumferential,
                n_radial=n_radial,
                material_data=shell_material
            )
            
            print("  [OK] Shell FEA module initialized successfully")
            
        except Exception as e:
            print(f"  [ERROR] Shell FEA initialization failed: {e}")
            self.shell_fea = None
            self.enable_shell = False

    def _init_filter(self, config: Optional[Dict[str, Any]]) -> None:
        """Initialize or reset FIR smoothing configuration."""
        # 解析梯度阶段强制关闭 FIR，除非调用方明确要求开启
        if self._force_disable_filter and config and 'disable_fir_filter' in config:
            self._force_disable_filter = bool(config.get('disable_fir_filter'))

        if self._force_disable_filter:
            self.load_filter_config = {}
            self._load_history = []
            self._last_raw_load_vector = None
            self.enable_fir_filter = False
            self.fir_window = 1
            self.fir_min_history = 1
            self._fir_base_weights = None
            return

        self.load_filter_config = dict(config or {})
        self.load_filter_config.pop('disable_fir_filter', None)
        self._load_history = []
        self._last_raw_load_vector = None
        # 只有同时启用壳体、非 simple_mode 且配置标记为 enabled 时才激活滤波
        enabled_flag = bool(self.load_filter_config.get('enabled', False))
        self.enable_fir_filter = bool(enabled_flag and self.enable_shell and not self.simple_mode)
        if not self.enable_fir_filter:
            self.fir_window = 1
            self.fir_min_history = 1
            self._fir_base_weights = None
            return

        self.fir_window = max(2, int(self.load_filter_config.get('window', 5)))
        self.fir_min_history = max(1, int(self.load_filter_config.get('min_history', 2)))
        decay = float(self.load_filter_config.get('decay', 0.6))
        raw_weights = self.load_filter_config.get('weights')

        if raw_weights is not None:
            base = np.asarray(raw_weights, dtype=float).ravel()
            base = base[np.isfinite(base)]
        else:
            base = decay ** np.arange(self.fir_window, dtype=float)

        if base.size == 0:
            base = np.ones(self.fir_window, dtype=float)

        if base.size < self.fir_window:
            if raw_weights is None:
                extra = decay ** np.arange(base.size, self.fir_window, dtype=float)
                base = np.concatenate([base, extra])
            else:
                base = np.pad(base, (0, self.fir_window - base.size), mode='edge')

        base = base[:self.fir_window]
        s = float(np.sum(base))
        if not np.isfinite(s) or s <= 0.0:
            base = np.ones(self.fir_window, dtype=float)
            s = float(self.fir_window)

        self._fir_base_weights = base / s
        self.fir_min_history = min(self.fir_min_history, self.fir_window)
        print(
            f"  FIR smoothing enabled: window={self.fir_window}, "
            f"weights={np.round(self._fir_base_weights, 4)}"
        )

    def configure_filter(self, config: Optional[Dict[str, Any]]) -> None:
        """Public helper to reconfigure FIR smoothing after construction."""
        self._init_filter(config)

    def _get_fir_weights(self, history_len: int) -> np.ndarray:
        if self._fir_base_weights is None or self._fir_base_weights.size == 0:
            weights = np.ones(history_len, dtype=float)
            return weights / float(history_len)
        base = self._fir_base_weights[:history_len]
        if base.size < history_len:
            pad_val = base[-1] if base.size else 1.0
            base = np.pad(base, (0, history_len - base.size), constant_values=pad_val)
        s = float(np.sum(base))
        if not np.isfinite(s) or s <= 0.0:
            base = np.ones(history_len, dtype=float)
            s = float(history_len)
        return base / s

    def _apply_load_filter(self, load_vector: np.ndarray) -> np.ndarray:
        if not self.enable_fir_filter:
            return load_vector

        if self._load_history and self._load_history[-1].shape != load_vector.shape:
            self._load_history = []
        self._load_history.append(load_vector.copy())
        if len(self._load_history) > self.fir_window:
            self._load_history.pop(0)

        history_len = len(self._load_history)
        if history_len < self.fir_min_history:
            return self._load_history[-1]

        weights = self._get_fir_weights(history_len)
        filtered = np.zeros_like(load_vector, dtype=float)
        for weight, vec in zip(weights, reversed(self._load_history)):
            filtered += weight * vec
        return filtered

    def compute_hydrostatic_loads(self, geometry, depth, radius, node_coords):
        """
        计算静水压力载荷
        
        Parameters:
        -----------
        geometry : GeometryData
            几何数据
        depth : float
            水深
        radius : float
            半径
        node_coords : np.ndarray
            更新后的节点坐标
        """
        # Simple mode fallback
        if self.simple_mode:
            return self._compute_simple_loads(geometry, depth, radius, node_coords)

        if not self.enable_shell or self.shell_fea is None:
            raise RuntimeError("Shell FEA is required but not initialized")

        if node_coords is None:
            raise ValueError("node_coords is required for dynamic shell FEA calculation")
            
        print("Using shell-based load calculation...")
        return self._compute_shell_based_loads(geometry, depth, radius, node_coords)
    
    def _compute_shell_based_loads(self, geometry, depth, radius, node_coords):
        """
        基于壳体的载荷计算
        """
        print("Computing shell-based loads...")
        
        # 获取外层节点当前坐标
        outer_node_coords = np.array([node_coords[i] for i in geometry.load_nodes])
        # Project support positions to shell inner wall (attach to inner boundary)
        try:
            inner_radius = float(self.shell_fea.outer_radius - self.shell_fea.material.thickness)
            if inner_radius > 0 and outer_node_coords.size:
                pos_inner = []
                for (px, py) in outer_node_coords:
                    r = float(np.hypot(px, py))
                    if r > 1e-12:
                        s = inner_radius / r
                        pos_inner.append([px * s, py * s])
                    else:
                        pos_inner.append([0.0, 0.0])
                outer_node_coords = np.asarray(pos_inner, dtype=float)
        except Exception:
            pass
        
        # 移除support position信息以节省空间
        # print(f"  Support positions: {len(outer_node_coords)} nodes")
        print(f"  Position range: x=[{np.min(outer_node_coords[:, 0]):.2f}, {np.max(outer_node_coords[:, 0]):.2f}]")
        print(f"                  y=[{np.min(outer_node_coords[:, 1]):.2f}, {np.max(outer_node_coords[:, 1]):.2f}]")
        
        # 使用壳体FEA计算支撑反力
        try:
            # 壳的内圈应与桁架外圈配合：直接以桁架 outer_nodes 坐标作为支撑位置
            support_reactions = self.shell_fea.solve_with_support_positions(outer_node_coords)
            print(f"  [OK] Shell FEA completed")
            print(f"  Max reaction force: {np.max(np.abs(support_reactions)):.0f} N")
            self._debug_log_support_state(
                stage='compute_shell_based_loads',
                support_positions=outer_node_coords,
                support_reactions=support_reactions,
                load_vector=None,
            )
            
        except Exception as e:
            print(f"  [ERROR] Shell FEA failed: {e}")
            raise RuntimeError(f"Shell FEA calculation failed: {e}")
        
        # 转换支撑反力为载荷向量格式（方向强制为径向向内，权重仅影响大小）
        load_vector = np.zeros(geometry.n_dof)

        for i, node_idx in enumerate(geometry.load_nodes):
            # 支撑点坐标与径向内法向量
            px, py = outer_node_coords[i]
            r = float(np.hypot(px, py))
            if r > 1e-12:
                nx, ny = -px / r, -py / r  # 向内法向量
            else:
                nx, ny = 0.0, -1.0

            # 壳体对支撑的反力（直接作用在桁架节点上）
            rx, ry = float(support_reactions[i, 0]), float(support_reactions[i, 1])
            fx = -rx
            fy = -ry
            load_vector[2 * node_idx] = fx
            load_vector[2 * node_idx + 1] = fy

        raw_vector = np.array(load_vector, copy=True)
        self._last_raw_load_vector = raw_vector
        filtered_vector = self._apply_load_filter(raw_vector)
        self.current_load_vector = np.array(filtered_vector, copy=True)

        self._debug_log_support_state(
            stage='compute_shell_based_loads_filtered',
            support_positions=outer_node_coords,
            support_reactions=support_reactions,
            load_vector=filtered_vector,
            node_indices=geometry.load_nodes,
            raw_load_vector=raw_vector,
        )

        # 计算等效基础压力（用于兼容性）
        base_pressure = self.material_data.rho_water * self.material_data.g * depth

        # 构建LoadData对象
        from .truss_system_initializer import LoadData
        return LoadData(
            load_vector=filtered_vector,
            base_pressure=base_pressure,
            depth=depth
        )
    
    def compute_load_vector(self, node_coords, outer_nodes, depth, return_jacobian: bool = False):
        """
        重新计算载荷向量（当几何改变时）
        
        这是你原有代码中用于SCP迭代的接口
        """
        if self.simple_mode:
            return self._compute_simple_load_vector(node_coords, outer_nodes, depth)
        if not self.enable_shell or self.shell_fea is None:
            raise RuntimeError("Shell FEA is required but not initialized")
        return self._compute_shell_load_vector(node_coords, outer_nodes, depth, return_jacobian=return_jacobian)

    def _compute_shell_load_vector(self, node_coords, outer_nodes, depth, return_jacobian: bool = False):
        """基于壳体的载荷向量计算"""
        coords = np.array(node_coords)
        
        # 提取外层节点坐标
        outer_node_coords = coords[outer_nodes]
        
        # 壳体FEA计算
        try:
            # 支撑位置直接使用桁架外圈节点坐标
            support_reactions = self.shell_fea.solve_with_support_positions(outer_node_coords)
            load_vector = np.zeros(len(coords) * 2)

            # 转换为载荷向量（方向强制为径向向内）
            outer_node_coords = coords[outer_nodes]
            for i, node_idx in enumerate(outer_nodes):
                px, py = outer_node_coords[i]
                r = float(np.hypot(px, py))
                if r > 1e-12:
                    nx, ny = -px / r, -py / r
                else:
                    nx, ny = 0.0, -1.0
                rx, ry = float(support_reactions[i, 0]), float(support_reactions[i, 1])
                fx = -rx
                fy = -ry
                load_vector[2*node_idx] = fx
                load_vector[2*node_idx + 1] = fy

            raw_vector = np.array(load_vector, copy=True)
            self._last_raw_load_vector = raw_vector
            filtered_vector = self._apply_load_filter(raw_vector)
            self.current_load_vector = np.array(filtered_vector, copy=True)

            self._debug_log_support_state(
                stage='compute_shell_load_vector',
                support_positions=outer_node_coords,
                support_reactions=support_reactions,
                load_vector=filtered_vector,
                node_indices=outer_nodes,
                raw_load_vector=raw_vector,
            )

            if return_jacobian:
                self._compute_shell_load_jacobian(coords, outer_nodes)
            else:
                self._last_shell_load_jacobian = None

            return filtered_vector
            
        except Exception as e:
            print(f"Shell load calculation failed: {e}")
            raise RuntimeError(f"Shell load calculation failed: {e}")

    def _compute_shell_load_jacobian(self, coords: np.ndarray, outer_nodes: List[int]) -> Optional[np.ndarray]:
        """Assemble ∂f/∂p where p 为支撑位置坐标 (x_i, y_i)。"""

        shell_state = self.shell_fea.get_last_solution() if self.shell_fea else None
        if not shell_state or shell_state.get("lambda") is None:
            self._last_shell_load_jacobian = None
            return None

        lambda_vec = np.asarray(shell_state["lambda"], dtype=float)
        u_vec = np.asarray(shell_state["u"], dtype=float)
        C = np.asarray(shell_state["C"], dtype=float)

        n_supports = len(outer_nodes)
        n_dof_truss = coords.shape[0] * 2
        shell_dof = u_vec.size

        if C.shape[0] != 2 * n_supports:
            raise RuntimeError("Mismatch between load nodes and shell constraint count.")

        jac = np.zeros((n_dof_truss, 2 * n_supports), dtype=float)

        for i in range(n_supports):
            for axis_idx, axis in enumerate(("x", "y")):
                dC = self.shell_fea.build_support_constraint_derivative(i, axis)
                rhs_top = -(dC.T @ lambda_vec)
                rhs_bottom = -(dC @ u_vec)
                rhs = np.concatenate([rhs_top, rhs_bottom])
                solution = self.shell_fea.solve_augmented_system(rhs)
                lambda_dot = solution[shell_dof:].reshape(n_supports, 2)
                col = 2 * i + axis_idx
                for j, node_idx in enumerate(outer_nodes):
                    jac[2 * node_idx, col] = lambda_dot[j, 0]
                    jac[2 * node_idx + 1, col] = lambda_dot[j, 1]

        self._last_shell_load_jacobian = jac
        return jac

    def get_last_shell_load_jacobian(self) -> Optional[np.ndarray]:
        """返回最近一次壳体载荷计算得到的 ∂f/∂p。"""

        return None if self._last_shell_load_jacobian is None else self._last_shell_load_jacobian.copy()

    # ---------------------------
    # Simple hydrostatic loads
    # ---------------------------
    def _compute_simple_loads(self, geometry, depth, radius, node_coords):
        """Compute simple hydrostatic nodal loads: per-node F_i = ρ g h_i.

        - h_i = max(0, depth - y_i) using current node y
        - Direction strictly radial inward at each load node
        - No angular redistribution (one node, one force)
        """
        coords = np.asarray(node_coords, dtype=float)
        ln = list(getattr(geometry, 'load_nodes', []))
        if not ln:
            return self._empty_load_data(depth)

        rho_g = float(self.material_data.rho_water * self.material_data.g)
        load_vector = np.zeros(geometry.n_dof, dtype=float)
        for nid in ln:
            x, y = float(coords[nid, 0]), float(coords[nid, 1])
            h = max(0.0, float(depth) - y)
            F = rho_g * h  # point force magnitude (debug/simple)
            r = float(np.hypot(x, y))
            if r > 1e-12:
                nx, ny = -x / r, -y / r
            else:
                nx, ny = 0.0, -1.0
            load_vector[2 * nid] = F * nx
            load_vector[2 * nid + 1] = F * ny

        raw_vector = np.array(load_vector, copy=True)
        self._last_raw_load_vector = raw_vector
        filtered_vector = self._apply_load_filter(raw_vector)
        self.current_load_vector = np.array(filtered_vector, copy=True)

        self._maybe_save_shell_displacement()

        from .truss_system_initializer import LoadData
        base_pressure = rho_g * float(depth)
        return LoadData(load_vector=filtered_vector, base_pressure=base_pressure, depth=depth)

    def _compute_simple_load_vector(self, node_coords, outer_nodes, depth):
        coords = np.asarray(node_coords, dtype=float)
        ln = list(outer_nodes)
        load_vector = np.zeros(coords.shape[0] * 2, dtype=float)
        if not ln:
            raw_vector = np.array(load_vector, copy=True)
            self._last_raw_load_vector = raw_vector
            filtered_vector = self._apply_load_filter(raw_vector)
            self.current_load_vector = np.array(filtered_vector, copy=True)
            return filtered_vector
        rho_g = float(self.material_data.rho_water * self.material_data.g)
        for nid in ln:
            x, y = float(coords[nid, 0]), float(coords[nid, 1])
            h = max(0.0, float(depth) - y)
            F = rho_g * h
            r = float(np.hypot(x, y))
            if r > 1e-12:
                nx, ny = -x / r, -y / r
            else:
                nx, ny = 0.0, -1.0
            load_vector[2 * nid] = F * nx
            load_vector[2 * nid + 1] = F * ny
        raw_vector = np.array(load_vector, copy=True)
        self._last_raw_load_vector = raw_vector
        filtered_vector = self._apply_load_filter(raw_vector)
        self.current_load_vector = np.array(filtered_vector, copy=True)
        return filtered_vector

    def _empty_load_data(self, depth):
        from .truss_system_initializer import LoadData
        zero = np.zeros(0, dtype=float)
        self._last_raw_load_vector = zero.copy()
        self.current_load_vector = zero.copy()
        self._load_history = []
        return LoadData(load_vector=zero, base_pressure=float(self.material_data.rho_water * self.material_data.g * depth), depth=depth)

    # ------------------------------------------------------------------
    # Shell displacement snapshots
    # ------------------------------------------------------------------
    def _maybe_save_shell_displacement(self) -> None:
        """Save shell displacement figure when shell FEA data is available."""
        if self.simple_mode or not self.enable_shell:
            return
        shell = getattr(self, 'shell_fea', None)
        if shell is None or not hasattr(shell, 'visualize_last_solution'):
            return

        out_dir = getattr(self, 'figure_output_dir', None)
        if not out_dir:
            return

        try:
            path = Path(out_dir)
            path.mkdir(parents=True, exist_ok=True)
            iteration = int(getattr(self, 'current_iteration', 0) or 0)
            fname = path / f"shell_disp_iter_{iteration:03d}.png"
            shell.visualize_last_solution(scale=None, save_path=str(fname))
        except Exception as exc:
            print(f"Warning: failed to save shell displacement snapshot: {exc}")

    def get_shell_info(self):
        """获取壳体信息（用于调试）"""
        if self.shell_fea:
            return self.shell_fea.get_mesh_info()
        else:
            return None

    def visualize_shell(self):
        """可视化壳体网格"""
        if self.shell_fea:
            self.shell_fea.visualize_mesh()
        else:
            print("Shell FEA not initialized")

    # ---------------------------
    # Debug helpers
    # ---------------------------

    def enable_debug_logging(self, path: str = 'debug_shell_support.log', reset: bool = True) -> None:
        """Enable detailed debug logging of shell support mapping/loads."""
        p = Path(path)
        if reset and p.exists():
            try:
                p.unlink()
            except OSError:
                pass
        self.debug_file = p
        header = (
            "# Shell support debug log\n"
            "# Columns: stage, point_index, node_id, x, y, r, nx, ny, "
            "lamb_x, lamb_y, dot, load_x_filtered, load_y_filtered, load_x_raw, load_y_raw, "
            "d_load_x, d_load_y, weights(list of node:weight pairs)\n"
        )
        self.debug_file.write_text(header, encoding='utf-8')
        self._debug_prev_loads = {}

    def _debug_log_support_state(self, stage: str, support_positions: np.ndarray,
                                 support_reactions: np.ndarray,
                                 load_vector: Optional[np.ndarray],
                                 node_indices: Optional[List[int]] = None,
                                 raw_load_vector: Optional[np.ndarray] = None) -> None:
        if self.debug_file is None or support_positions is None or support_reactions is None:
            return
        try:
            support_positions = np.asarray(support_positions, dtype=float)
            support_reactions = np.asarray(support_reactions, dtype=float)
            if node_indices is None:
                node_indices = list(range(len(support_positions)))
            weights = getattr(self.shell_fea, '_last_support_weights', [])
            with self.debug_file.open('a', encoding='utf-8') as fh:
                for i, node_idx in enumerate(node_indices):
                    px, py = map(float, support_positions[i])
                    rx = float(support_reactions[i, 0]) if support_reactions.ndim >= 2 else float(support_reactions[i])
                    ry = float(support_reactions[i, 1]) if support_reactions.ndim >= 2 else 0.0
                    r = float(np.hypot(px, py))
                    if r > 1e-12:
                        nx, ny = -px / r, -py / r
                    else:
                        nx, ny = 0.0, -1.0
                    dot = rx * nx + ry * ny
                    load_x = load_y = 0.0
                    if load_vector is not None and 2 * node_idx + 1 < load_vector.size:
                        load_x = float(load_vector[2 * node_idx])
                        load_y = float(load_vector[2 * node_idx + 1])
                    raw_x = raw_y = 0.0
                    if raw_load_vector is not None and 2 * node_idx + 1 < raw_load_vector.size:
                        raw_x = float(raw_load_vector[2 * node_idx])
                        raw_y = float(raw_load_vector[2 * node_idx + 1])
                    prev = self._debug_prev_loads.get(int(node_idx))
                    d_load_x = load_x - prev[0] if (prev and load_vector is not None) else 0.0
                    d_load_y = load_y - prev[1] if (prev and load_vector is not None) else 0.0
                    if load_vector is not None:
                        self._debug_prev_loads[int(node_idx)] = (load_x, load_y)
                    weight_list = []
                    if weights and i < len(weights):
                        weight_list = [f"({nid}:{w:.6f})" for nid, w in weights[i]]
                    fh.write(
                        f"{stage},{i},{node_idx},{px:.6f},{py:.6f},{r:.6f},{nx:.6f},{ny:.6f},"
                        f"{rx:.6f},{ry:.6f},{dot:.6f},{load_x:.6f},{load_y:.6f},{raw_x:.6f},{raw_y:.6f},"
                        f"{d_load_x:.6f},{d_load_y:.6f},[{';'.join(weight_list)}]\n"
                    )
        except Exception as exc:
            # Debug logging failures must not break primary workflow.
            print(f"[debug] Failed to log shell support state: {exc}")


def integrate_shell_into_existing_system():
    """
    演示如何将壳体集成到你现有系统中
    """
    print("Demo: Integrating Shell FEA into existing system")
    print("=" * 50)
    
    # 1. 模拟你的现有系统
    from .truss_system_initializer import TrussSystemInitializer
    
    # 创建初始化器（使用你的现有代码）
    initializer = TrussSystemInitializer(
        radius=5.0,
        n_sectors=16, 
        inner_ratio=0.6,
        depth=50,
        volume_fraction=0.2,
        enable_middle_layer=True,
        middle_layer_ratio=0.8
    )
    
    print(f"\n[OK] Truss system initialized:")
    print(f"  Load nodes: {len(getattr(initializer, 'load_nodes', initializer.outer_nodes))}")
    print(f"  Original load vector max: {np.max(np.abs(initializer.load_vector)):.0f} N")
    
    # 2. 替换LoadCalculator为壳体版本
    shell_params = {
        'outer_radius': initializer.radius,
        'depth': initializer.depth,
        'thickness': 0.1,  # 10cm厚度
        'n_circumferential': len(getattr(initializer, 'load_nodes', initializer.outer_nodes)),  # 与桁架节点对应
        'n_radial': 2,  # 薄壳，只要2层
        'E_shell': 210e9  # 210 GPa，比桁架刚度大
    }
    
    # 创建带壳体的载荷计算器
    shell_load_calc = LoadCalculatorWithShell(
        material_data=initializer.material_data,
        enable_shell=True,
        shell_params=shell_params
    )
    
    # 3. 重新计算载荷
    print(f"\n📊 Comparing load calculations:")
    
    # 传统方法
    shell_load_calc.enable_shell = False
    traditional_loads = shell_load_calc.compute_load_vector(
        initializer.nodes, initializer.outer_nodes, initializer.depth
    )
    
    # 壳体方法
    shell_load_calc.enable_shell = True
    shell_loads = shell_load_calc.compute_load_vector(
        initializer.nodes, initializer.outer_nodes, initializer.depth  
    )
    
    print(f"  Traditional max load: {np.max(np.abs(traditional_loads)):.0f} N")
    print(f"  Shell-based max load: {np.max(np.abs(shell_loads)):.0f} N")
    print(f"  Difference: {np.linalg.norm(shell_loads - traditional_loads):.0f} N")
    
    # 4. 可视化对比
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 传统载荷
        ln = getattr(initializer, 'load_nodes', initializer.outer_nodes)
        outer_coords = np.array([initializer.nodes[i] for i in ln])
        trad_forces_y = [traditional_loads[2*i+1] for i in ln]
        
        ax1.scatter(outer_coords[:, 0], outer_coords[:, 1], c=np.abs(trad_forces_y), 
                   cmap='Reds', s=100, alpha=0.8)
        ax1.set_title('Traditional Hydrostatic Loads')
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        
        # 壳体载荷  
        shell_forces_y = [shell_loads[2*i+1] for i in ln]
        
        ax2.scatter(outer_coords[:, 0], outer_coords[:, 1], c=np.abs(shell_forces_y),
                   cmap='Blues', s=100, alpha=0.8)
        ax2.set_title('Shell-based Loads')
        ax2.set_aspect('equal') 
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("  (Visualization skipped - matplotlib not available)")
    
    return shell_load_calc


if __name__ == "__main__":
    # 运行集成演示
    shell_calc = integrate_shell_into_existing_system()
    
    # 可视化壳体网格
    print(f"\n🔍 Visualizing shell mesh...")
    shell_calc.visualize_shell()
