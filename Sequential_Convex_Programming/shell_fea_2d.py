"""
2D Shell FEA Module
用于计算高刚度壳体的载荷传递和支撑反力

Date: 2025/9/4
"""
import numpy as np
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class ShellMaterialData:
    """壳体材料数据"""
    E: float = 210e9          # 弹性模量 (Pa) - 比桁架大10倍
    nu: float = 0.3           # 泊松比
    thickness: float = 0.1   # 厚度 (m)
    rho_water: float = 1025   # 水密度 (kg/m³)  
    g: float = 9.81           # 重力加速度 (m/s²)

@dataclass
class ShellMeshData:
    """壳体网格数据"""
    nodes: np.ndarray         # 节点坐标 [n_nodes, 2]
    elements: np.ndarray      # 单元连接 [n_elements, 3] (三角形单元)
    boundary_nodes: List[int] # 外边界节点索引
    boundary_edges: List[Tuple[int, int]] # 外边界边列表

class Shell2DFEA:
    """
    2D壳体有限元分析类
    
    功能：
    1. 生成半圆形壳体网格
    2. 计算静水压力分布载荷
    3. 根据支撑位置求解壳体响应
    4. 提取支撑反力
    """
    
    def __init__(self, outer_radius: float, depth: float, 
                 n_circumferential: int = 100, n_radial: int = 5,
                 material_data: Optional[ShellMaterialData] = None,
                 k_neighbors: int = 5,
                 sigma_factor: float = 1.5,
                 adaptive_sigma: bool = False,
                 epsilon_weight: float = 0.05):
        """
        初始化壳体FEA模块
        
        Parameters:
        -----------
        outer_radius : float
            外层半径 (m)
        depth : float  
            水深 (m)
        n_circumferential : int
            周向网格数
        n_radial : int
            径向网格数
        material_data : ShellMaterialData
            材料参数
        """
        self.outer_radius = outer_radius
        self.depth = depth
        self.n_circumferential = n_circumferential
        self.n_radial = n_radial
        
        # 材料数据
        self.material = material_data or ShellMaterialData()
        
        # 软权重参数（高斯核）
        self.k_neighbors = max(3, int(k_neighbors))
        self.sigma_factor = float(sigma_factor)
        self.adaptive_sigma = bool(adaptive_sigma)
        self.epsilon_weight = float(epsilon_weight)

        # 生成网格和预计算
        self._generate_shell_mesh()
        self._precompute_element_matrices()
        self._compute_pressure_loads()
        # 预计算边界节点角度（用于软权重）
        self._compute_boundary_angles()

        # 缓存最近一次求解结果（位移、反力、权重等）
        self._last_displacement: Optional[np.ndarray] = None
        self._last_reactions: Optional[np.ndarray] = None
        self._last_support_positions: Optional[np.ndarray] = None
        self._last_support_weights = None

        print(f"Shell2DFEA initialized:")
        print(f"  Outer radius: {outer_radius}m")
        print(f"  Depth: {depth}m") 
        print(f"  Mesh: {self.mesh.nodes.shape[0]} nodes, {self.mesh.elements.shape[0]} elements")
        print(f"  Shell stiffness: E = {self.material.E/1e9:.1f} GPa")
    
    def _generate_shell_mesh(self):
        """生成半圆形壳体网格"""
        print("Generating thin shell mesh...")
        
        # 生成节点
        nodes = []
        
        # 径向坐标 - 真正的薄壳：只在厚度范围内
        inner_radius = self.outer_radius - self.material.thickness
        radial_coords = np.linspace(inner_radius, self.outer_radius, self.n_radial)
        # np.linspace(start, stop, num)
        # 角度坐标  
        angles = np.linspace(0, np.pi, self.n_circumferential)
        
        # 生成节点坐标
        for r in radial_coords:
            for angle in angles:
                x = r * np.cos(angle)
                y = r * np.sin(angle) 
                nodes.append([x, y])
        
        nodes = np.array(nodes)
        
        # 生成三角形单元
        elements = []
        
        for i in range(self.n_radial - 1):
            for j in range(self.n_circumferential - 1):
                # 当前层的节点索引
                n1 = i * self.n_circumferential + j
                n2 = i * self.n_circumferential + j + 1  
                n3 = (i + 1) * self.n_circumferential + j
                n4 = (i + 1) * self.n_circumferential + j + 1
                
                # 两个三角形
                elements.append([n1, n2, n3])
                elements.append([n2, n4, n3])
        
        elements = np.array(elements)
        
        # 识别边界节点（最外层）
        boundary_nodes = list(range((self.n_radial - 1) * self.n_circumferential,
                                   self.n_radial * self.n_circumferential))

        boundary_edges: List[Tuple[int, int]] = []
        if len(boundary_nodes) >= 2:
            for idx in range(len(boundary_nodes) - 1):
                boundary_edges.append((boundary_nodes[idx], boundary_nodes[idx + 1]))
            # wrap-around edge closes the semicircle (last to first)
            boundary_edges.append((boundary_nodes[-1], boundary_nodes[0]))

        self.mesh = ShellMeshData(
            nodes=nodes,
            elements=elements, 
            boundary_nodes=boundary_nodes,
            boundary_edges=boundary_edges
        )
        
        print(f"  Generated mesh: {len(nodes)} nodes, {len(elements)} elements")

    def _compute_boundary_angles(self):
        """计算外边界节点的极角，用于支撑软权重分配"""
        bnodes = self.mesh.boundary_nodes
        coords = self.mesh.nodes
        thetas = []
        for n in bnodes:
            x, y = coords[n]
            theta = math.atan2(y, x)
            # 夹到 [0, pi]
            if theta < 0:
                theta = 0.0
            if theta > math.pi:
                theta = math.pi
            thetas.append(theta)
        self.boundary_angles = np.array(thetas)
    
    def _precompute_element_matrices(self):
        """预计算单元刚度矩阵"""
        print("Precomputing element stiffness matrices...")
        
        # 平面应力弹性矩阵
        E = self.material.E
        nu = self.material.nu
        t = self.material.thickness
        
        D = E * t / (1 - nu**2) * np.array([
            [1, nu, 0],
            [nu, 1, 0], 
            [0, 0, (1-nu)/2]
        ])
        
        self.D_matrix = D
        self.element_stiffness_matrices = []
        
        # 计算每个单元的刚度矩阵
        for elem_id, element in enumerate(self.mesh.elements):
            # 节点坐标
            nodes_coords = self.mesh.nodes[element]  # [3, 2]
            
            # 计算单元刚度矩阵
            K_elem = self._compute_triangle_stiffness(nodes_coords)
            self.element_stiffness_matrices.append(K_elem)
        
        print(f"  Precomputed {len(self.element_stiffness_matrices)} element matrices")
    
    def _compute_triangle_stiffness(self, coords: np.ndarray) -> np.ndarray:
        """计算三角形单元刚度矩阵"""
        # 节点坐标
        x1, y1 = coords[0]
        x2, y2 = coords[1] 
        x3, y3 = coords[2]
        
        # 面积
        area = 0.5 * abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))
        
        if area < 1e-12:
            return np.zeros((6, 6))  # 退化单元
        
        # B矩阵
        b1 = y2 - y3
        b2 = y3 - y1
        b3 = y1 - y2
        c1 = x3 - x2
        c2 = x1 - x3  
        c3 = x2 - x1
        
        B = 1/(2*area) * np.array([
            [b1, 0, b2, 0, b3, 0],
            [0, c1, 0, c2, 0, c3],
            [c1, b1, c2, b2, c3, b3]
        ])
        
        # 刚度矩阵 K = B^T * D * B * area
        K = B.T @ self.D_matrix @ B * area
        
        return K
    
    def _compute_pressure_loads(self):
        """计算静水压力并映射到壳体节点"""
        print("Computing hydrostatic pressure loads...")

        pressure_loads = np.zeros(len(self.mesh.nodes) * 2, dtype=float)

        rho_water = self.material.rho_water
        g = self.material.g

        boundary_nodes = np.asarray(self.mesh.boundary_nodes, dtype=int)
        boundary_edges = getattr(self.mesh, 'boundary_edges', [])
        if boundary_nodes.size == 0 or not boundary_edges:
            self.pressure_loads = pressure_loads
            print("  Total pressure load: 0 N")
            return

        boundary_edges = list(boundary_edges)

        for node_i, node_j in boundary_edges:
            p0 = self.mesh.nodes[node_i]
            p1 = self.mesh.nodes[node_j]
            edge_vec = p1 - p0
            edge_length = float(np.linalg.norm(edge_vec))
            if edge_length <= 0.0:
                continue

            midpoint = 0.5 * (p0 + p1)
            # Midpoint rule: integrate hydrostatic pressure along the boundary edge
            depth_local = max(0.0, self.depth - midpoint[1])
            pressure = rho_water * g * depth_local

            r_mid = float(np.linalg.norm(midpoint))
            if r_mid > 1e-12:
                nx = -midpoint[0] / r_mid
                ny = -midpoint[1] / r_mid
            else:
                nx, ny = 0.0, 0.0

            line_force = pressure * self.material.thickness * edge_length
            fx = line_force * nx
            fy = line_force * ny

            pressure_loads[2 * node_i] += 0.5 * fx
            pressure_loads[2 * node_i + 1] += 0.5 * fy
            pressure_loads[2 * node_j] += 0.5 * fx
            pressure_loads[2 * node_j + 1] += 0.5 * fy

        self.pressure_loads = pressure_loads

        total_force_vec = pressure_loads.reshape(-1, 2).sum(axis=0)
        total_force = float(np.linalg.norm(total_force_vec))
        print(f"  Total pressure load: {total_force:.0f} N")

    def solve_with_support_positions(self, support_positions: np.ndarray) -> np.ndarray:
        """Solve the shell problem with weighted supports and return reactions."""
        if support_positions is None or len(support_positions) == 0:
            return np.zeros((0, 2), dtype=float)

        support_positions = np.asarray(support_positions, dtype=float)
        support_weights = self._compute_support_weights(support_positions)
        if not support_weights:
            return np.zeros((0, 2), dtype=float)

        # Store for downstream debugging/inspection (shallow copy is fine).
        self._last_support_weights = list(support_weights)

        K_global = self._assemble_global_stiffness()
        n_dof = len(self.mesh.nodes) * 2
        C = self._build_support_constraint_matrix(support_weights)
        m = C.shape[0]

        A = np.zeros((n_dof + m, n_dof + m))
        A[:n_dof, :n_dof] = K_global
        A[:n_dof, n_dof:] = C.T
        A[n_dof:, :n_dof] = C

        b = np.zeros(n_dof + m)
        b[:n_dof] = self.pressure_loads

        try:
            sol = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            print("  Warning: Indefinite/singular saddle system, using pseudo-inverse")
            sol = np.linalg.pinv(A) @ b

        lamb = sol[n_dof:]
        # λ corresponds to the constraint forces applied by the structure on the supports;
        # invert the sign so we return the physical support reactions (supports on shell).
        reactions = (-lamb).reshape(-1, 2)

        # 缓存便于可视化与调试
        self._last_displacement = sol[:n_dof].reshape(-1, 2)
        self._last_reactions = reactions.copy()
        self._last_support_positions = support_positions.copy()
        self._last_support_weights = list(support_weights)

        total_pressure_vec = self.pressure_loads.reshape(-1, 2).sum(axis=0)
        total_reaction_vec = reactions.sum(axis=0)
        imbalance = total_reaction_vec + total_pressure_vec
        imbalance_norm = float(np.linalg.norm(imbalance))
        reference_norm = float(max(1.0, np.linalg.norm(total_pressure_vec)))
        if imbalance_norm > 1e-3 * reference_norm:
            print(f"  [warn] Reaction imbalance |Δ|={imbalance_norm:.2e} N")

        return reactions

    def visualize_last_solution(self, scale: Optional[float] = None,
                                save_path: Optional[str] = None,
                                show_reference: bool = False) -> None:
        """渲染最近一次求解的位移彩色云图（类似商业软件）。

        Parameters
        ----------
        scale : float
            位移放大系数，便于观察微小形变。
        show_reactions : bool
            是否在支撑位置绘制反力箭头。
        show_normals : bool
            是否绘制单位法向用于对比。
        """
        if self._last_displacement is None:
            print("No shell solution cached; call solve_with_support_positions first.")
            return

        try:
            import matplotlib.pyplot as plt
            from matplotlib.collections import PolyCollection, LineCollection
        except ImportError:
            print("Matplotlib not available for visualization")
            return

        nodes = self.mesh.nodes
        disp = self._last_displacement
        max_disp = float(np.max(np.linalg.norm(disp, axis=1)))
        if scale is None:
            if max_disp > 0.0:
                # 将最大位移放大到壳体半径约 5% 以便观察
                scale_use = 0.05 * self.outer_radius / max_disp
            else:
                scale_use = 1.0
        else:
            scale_use = float(scale)
        displaced = nodes + scale_use * disp

        elements = self.mesh.elements
        polys = displaced[elements]
        disp_mag = np.linalg.norm(disp, axis=1)
        cell_values = np.mean(disp_mag[elements], axis=1)

        fig, ax = plt.subplots(figsize=(8, 6))
        poly = PolyCollection(polys, array=cell_values, cmap='viridis', edgecolors='none')
        ax.add_collection(poly)
        cbar = fig.colorbar(poly, ax=ax, pad=0.02)
        cbar.set_label('|u| (m)')

        if show_reference:
            seg_ref = []
            seg_def = []
            for elem in elements:
                pts_ref = nodes[elem]
                pts_def = displaced[elem]
                for i in range(3):
                    j = (i + 1) % 3
                    seg_ref.append([pts_ref[i], pts_ref[j]])
                    seg_def.append([pts_def[i], pts_def[j]])
            ax.add_collection(LineCollection(seg_ref, colors='lightgray', linewidths=0.4, alpha=0.6))
            ax.add_collection(LineCollection(seg_def, colors='black', linewidths=0.6, alpha=0.8))

        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        title_scale = f"auto({scale_use:.2e})" if scale is None else f"{scale_use:.2f}"
        ax.set_title(f'Shell displacement (scale={title_scale})', fontweight='bold')
        ax.set_xlim(np.min(displaced[:, 0]) - 0.2, np.max(displaced[:, 0]) + 0.2)
        ax.set_ylim(np.min(displaced[:, 1]) - 0.2, np.max(displaced[:, 1]) + 0.2)

        plt.tight_layout()
        if save_path:
            try:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
            finally:
                plt.close(fig)
        else:
            plt.show()
    def get_mesh_info(self) -> dict:
        """获取网格资料，便于调试和可视化"""
        return {
            'nodes': self.mesh.nodes,
            'elements': self.mesh.elements,
            'boundary_nodes': self.mesh.boundary_nodes,
            'n_nodes': len(self.mesh.nodes),
            'n_elements': len(self.mesh.elements)
        }

    def visualize_mesh(self):
        """简单的网格可视化"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches

            fig, ax = plt.subplots(figsize=(10, 6))

            for element in self.mesh.elements:
                coords = self.mesh.nodes[element]
                triangle = patches.Polygon(coords, fill=False, edgecolor='blue', alpha=0.6)
                ax.add_patch(triangle)

            ax.scatter(self.mesh.nodes[:, 0], self.mesh.nodes[:, 1],
                      c='red', s=20, alpha=0.7, label='Nodes')

            boundary_coords = self.mesh.nodes[self.mesh.boundary_nodes]
            ax.scatter(boundary_coords[:, 0], boundary_coords[:, 1],
                      c='green', s=40, alpha=0.8, label='Boundary Nodes')

            arc = patches.Arc((0, 0), 2*self.outer_radius, 2*self.outer_radius,
                              angle=0, theta1=0, theta2=180,
                              linestyle='--', color='black', alpha=0.5)
            ax.add_patch(arc)

            ax.set_xlim(-1.2 * self.outer_radius, 1.2 * self.outer_radius)
            ax.set_ylim(-0.2 * self.outer_radius, 1.2 * self.outer_radius)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_title('Shell2D FEA Mesh')

            plt.show()

        except ImportError:
            print("Matplotlib not available for visualization")

    def _compute_support_weights(self, support_positions: np.ndarray) -> List[List[Tuple[int, float]]]:
        """Gaussian-weighted mapping from support positions to boundary nodes."""
        weights_all: List[List[Tuple[int, float]]] = []
        boundary_nodes = np.asarray(self.mesh.boundary_nodes, dtype=int)
        if boundary_nodes.size == 0:
            return weights_all
        theta_b = self.boundary_angles
        dtheta = math.pi / max(self.n_circumferential - 1, 1)
        k_use = max(3, min(self.k_neighbors, boundary_nodes.size))
        for pos in support_positions:
            x, y = float(pos[0]), float(pos[1])
            theta = math.atan2(y, x)
            theta = min(max(theta, 0.0), math.pi)
            d_all = np.abs(theta - theta_b)
            idx_sorted = np.argsort(d_all)[:k_use]
            local_angles = d_all[idx_sorted]
            if self.adaptive_sigma and local_angles.size > 0:
                W = float(np.max(local_angles))
                eps = max(1e-6, min(0.2, self.epsilon_weight))
                sigma = W / math.sqrt(2.0 * math.log(1.0 / eps)) if W > 0 else (self.sigma_factor * dtheta)
            else:
                sigma = max(1e-12, self.sigma_factor * dtheta)
            exparg = -0.5 * (local_angles / sigma) ** 2
            exparg -= float(np.max(exparg))
            w_raw = np.exp(exparg)
            s = float(np.sum(w_raw))
            if not np.isfinite(s) or s <= 0:
                w = np.full_like(w_raw, 1.0 / max(1, len(w_raw)))
            else:
                w = w_raw / s
            weights = [(int(boundary_nodes[idx_sorted[j]]), float(w[j])) for j in range(len(idx_sorted))]
            weights_all.append(weights)
        return weights_all

    def _build_support_constraint_matrix(self, support_weights: List[List[Tuple[int, float]]]) -> np.ndarray:
        n_supports = len(support_weights)
        n_dof = len(self.mesh.nodes) * 2
        C = np.zeros((2 * n_supports, n_dof))
        for i, weights in enumerate(support_weights):
            for node_idx, w in weights:
                C[2 * i, 2 * node_idx] += w
                C[2 * i + 1, 2 * node_idx + 1] += w
        return C

    def _assemble_global_stiffness(self) -> np.ndarray:
        """组装全局刚度矩阵"""
        n_dof = len(self.mesh.nodes) * 2
        K_global = np.zeros((n_dof, n_dof))

        for elem_id, element in enumerate(self.mesh.elements):
            K_elem = self.element_stiffness_matrices[elem_id]

            dofs = []
            for node in element:
                dofs.extend([2 * node, 2 * node + 1])

            for i in range(6):
                for j in range(6):
                    K_global[dofs[i], dofs[j]] += K_elem[i, j]

        return K_global


def test_shell_fea():
    """测试Shell2DFEA模块"""
    print("Testing Shell2DFEA module...")
    
    # 创建实例
    shell_fea = Shell2DFEA(
        outer_radius=5.0,
        depth=50.0,
        n_circumferential=200, # 周向网格数
        n_radial=5 # 径向网格数
    )
    
    # 可视化网格
    shell_fea.visualize_mesh()
    
    # 测试支撑计算
    test_positions = np.array([
        [5.0, 0.0],      # 右端点
        [0.0, 5.0],      # 顶点  
        [-5.0, 0.0],     # 左端点
        [2.5, 4.33]      # 中间点
    ])
    
    reactions = shell_fea.solve_with_support_positions(test_positions)
    
    print(f"\nTest results:")
    print(f"Support reactions shape: {reactions.shape}")
    print(f"Support reactions:\n{reactions}")
    
    return shell_fea


if __name__ == "__main__":
    shell = test_shell_fea()
