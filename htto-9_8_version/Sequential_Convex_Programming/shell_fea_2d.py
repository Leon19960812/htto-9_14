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
    E: float = 210e7          # 弹性模量 (Pa) - 比桁架大10倍
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
        
        self.mesh = ShellMeshData(
            nodes=nodes,
            elements=elements, 
            boundary_nodes=boundary_nodes
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
        """计算静水压力分布载荷"""
        print("Computing hydrostatic pressure loads...")
        
        # 对边界节点计算压力载荷
        pressure_loads = np.zeros(len(self.mesh.nodes) * 2)  # [fx, fy, fx, fy, ...]
        
        rho_water = self.material.rho_water
        g = self.material.g
        
        # 计算边界上相邻节点的弧长（均匀角度划分）
        dtheta = np.pi / (self.n_circumferential - 1)
        arc_length = self.outer_radius * dtheta  # 相邻边界节点间弧长
        # 采用最外两层半径的间距作为条带“径向高度”
        # 与 _generate_shell_mesh 中 radial_coords 一致：inner=R-t, outer=R，等距 n_radial 划分
        radial_spacing = (self.outer_radius - (self.outer_radius - self.material.thickness)) / max(self.n_radial - 1, 1)
        strip_area = arc_length * radial_spacing
        
        for node_id in self.mesh.boundary_nodes:
            x, y = self.mesh.nodes[node_id]
            
            # 节点深度
            node_depth = self.depth - y
            
            # 静水压力 
            pressure = rho_water * g * node_depth
            
            # 压力方向：径向向内
            r = np.sqrt(x**2 + y**2)
            if r > 1e-12:
                nx = -x / r  # 向内的法向量
                ny = -y / r
            else:
                nx, ny = 0, 0
            
            # 节点载荷（等效为外圈条带面积的均匀分配）
            tributary_area = strip_area
            force_magnitude = pressure * tributary_area
            
            pressure_loads[2*node_id] = force_magnitude * nx
            pressure_loads[2*node_id + 1] = force_magnitude * ny
        
        self.pressure_loads = pressure_loads
        
        total_force = np.sqrt(np.sum(pressure_loads[::2]**2) + np.sum(pressure_loads[1::2]**2))
        print(f"  Total pressure load: {total_force:.0f} N")
    
    def solve_with_support_positions(self, support_positions: np.ndarray) -> np.ndarray:
        """
        根据支撑位置求解壳体并返回支撑反力
        
        Parameters:
        -----------
        support_positions : np.ndarray
            支撑位置坐标 [n_supports, 2]
            
        Returns:
        --------
        np.ndarray
            支撑反力 [n_supports, 2] (fx, fy)
        """
        #print(f"Solving shell FEA with {len(support_positions)} support positions...")
        
        # 1. 为每个支撑点根据外圈边界节点计算高斯核软权重（k近邻）
        support_weights = self._compute_support_weights(support_positions)
        
        # 2. 组装全局刚度矩阵
        K_global = self._assemble_global_stiffness()

        # 3. 构建软权重的位移约束（MPC：Σ w_j u(n_j) = 0），用 Lagrange 乘子实现
        C = self._build_support_constraint_matrix(support_weights)
        n_dof = len(self.mesh.nodes) * 2
        m = C.shape[0]  # 约束个数 = 2 * n_supports

        # 增广鞍点系统 [K  C^T; C  0] [u;λ] = [f;0]
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

        u = sol[:n_dof]
        lamb = sol[n_dof:]

        # 4. 以 Lagrange 乘子作为支撑反力（每个支撑两分量）
        support_reactions = lamb.reshape(-1, 2)

        return support_reactions
    
    def _compute_support_weights(self, support_positions: np.ndarray) -> List[List[Tuple[int, float]]]:
        """为每个支撑位置计算基于高斯核的连续软权重（使用所有外边界节点）。
        说明：
        - 过去的做法是选取 k 近邻并归一化，这会在近邻集合切换时产生非连续跳变；
        - 这里改为对全部边界节点施加高斯权重并归一化，确保 f(θ) 对 θ 连续，
          从而提升线性化预测的稳定性，降低 rho 异常的概率。
        返回：列表，长度为 n_supports；每个元素是 [(node_idx, weight), ...]
        """
        weights_all: List[List[Tuple[int, float]]] = []
        bnodes = np.array(self.mesh.boundary_nodes)
        theta_b = self.boundary_angles  # [N_b]
        dtheta = math.pi / max(self.n_circumferential - 1, 1)
        # 为了自适应选择 σ，可仍然用局部 k 近邻的跨度来设置 σ，但计算权重时使用全部边界节点
        k = max(3, min(self.k_neighbors, len(bnodes)))

        for pos in support_positions:
            x, y = float(pos[0]), float(pos[1])
            theta = math.atan2(y, x)
            if theta < 0:
                theta = 0.0
            if theta > math.pi:
                theta = math.pi
            # 与全部边界节点的角差
            d_all = np.abs(theta - theta_b)
            # σ 选择：固定比例或基于局部 k 邻域跨度的自适应
            if self.adaptive_sigma:
                idx_sorted = np.argsort(d_all)
                d_nei = d_all[idx_sorted[:k]]
                W = float(np.max(d_nei)) if len(d_nei) > 0 else dtheta
                eps = max(1e-6, min(0.2, self.epsilon_weight))
                sigma = W / math.sqrt(2.0 * math.log(1.0/eps)) if W > 0 else (self.sigma_factor * dtheta)
            else:
                sigma = max(1e-12, self.sigma_factor * dtheta)
            # 高斯权重（对全部边界节点），数值稳定处理
            exparg = -0.5 * (d_all / sigma)**2
            exparg -= float(np.max(exparg))  # 稳定化，避免溢出
            w_raw = np.exp(exparg)
            s = float(np.sum(w_raw))
            if not np.isfinite(s) or s <= 0:
                # 回退：均匀分布到全部边界节点
                w = np.full_like(w_raw, 1.0 / max(1, len(w_raw)))
            else:
                w = w_raw / s
            # 打包 (node_idx, weight)（包含全部边界节点，连续）
            weights = [(int(bnodes[j]), float(w[j])) for j in range(len(bnodes))]
            weights_all.append(weights)
        return weights_all
    
    def _assemble_global_stiffness(self) -> np.ndarray:
        """组装全局刚度矩阵"""
        n_dof = len(self.mesh.nodes) * 2
        K_global = np.zeros((n_dof, n_dof))
        
        for elem_id, element in enumerate(self.mesh.elements):
            K_elem = self.element_stiffness_matrices[elem_id]
            
            # DOF映射
            dofs = []
            for node in element:
                dofs.extend([2*node, 2*node+1])
            
            # 组装
            for i in range(6):
                for j in range(6):
                    K_global[dofs[i], dofs[j]] += K_elem[i, j]
        
        return K_global
    
    def _build_support_constraint_matrix(self, support_weights: List[List[Tuple[int, float]]]) -> np.ndarray:
        """构建 MPC 约束矩阵 C，使 Σ w_j u(n_j)=0（x/y 两个方向各一条）"""
        n_supports = len(support_weights)
        n_dof = len(self.mesh.nodes) * 2
        C = np.zeros((2 * n_supports, n_dof))
        for i, weights in enumerate(support_weights):
            for n_idx, w in weights:
                # x 方向约束
                C[2*i, 2*n_idx] = C[2*i, 2*n_idx] + w
                # y 方向约束
                C[2*i+1, 2*n_idx+1] = C[2*i+1, 2*n_idx+1] + w
        return C
    
    def _reconstruct_global_displacement(self, u_reduced: np.ndarray, 
                                        free_dofs: List[int], support_node_ids: List[int]) -> np.ndarray:
        """重构全局位移向量"""
        n_dof = len(self.mesh.nodes) * 2
        u_global = np.zeros(n_dof)
        
        # 填入自由度位移
        for i, dof in enumerate(free_dofs):
            u_global[dof] = u_reduced[i]
        
        # 支撑节点位移为0（已经是默认值）
        
        return u_global
    
    def _compute_support_reactions(self, K_global: np.ndarray, u_global: np.ndarray,
                                  support_pairs: List[Tuple[int, int, float, float]]) -> np.ndarray:
        """计算支撑反力"""
        # 先计算每个节点的反力
        node_react = {}
        for n0, n1, _, _ in support_pairs:
            for nid in (n0, n1):
                if nid in node_react:
                    continue
                dofs = [2*nid, 2*nid+1]
                rx = float(np.dot(K_global[dofs[0], :], u_global) - self.pressure_loads[dofs[0]])
                ry = float(np.dot(K_global[dofs[1], :], u_global) - self.pressure_loads[dofs[1]])
                node_react[nid] = (rx, ry)
        # 将每个支撑的反力按权重从两节点汇总
        reactions = []
        for n0, n1, w0, w1 in support_pairs:
            rx0, ry0 = node_react[n0]
            rx1, ry1 = node_react[n1]
            reactions.extend([w0*rx0 + w1*rx1, w0*ry0 + w1*ry1])
        return np.array(reactions)
    
    def get_mesh_info(self) -> dict:
        """获取网格信息，用于调试和可视化"""
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
            
            # 绘制单元
            for element in self.mesh.elements:
                coords = self.mesh.nodes[element]
                triangle = patches.Polygon(coords, fill=False, edgecolor='blue', alpha=0.6)
                ax.add_patch(triangle)
            
            # 绘制节点
            ax.scatter(self.mesh.nodes[:, 0], self.mesh.nodes[:, 1], 
                      c='red', s=20, alpha=0.7, label='Nodes')
            
            # 突出显示边界节点
            boundary_coords = self.mesh.nodes[self.mesh.boundary_nodes]
            ax.scatter(boundary_coords[:, 0], boundary_coords[:, 1], 
                      c='green', s=40, alpha=0.8, label='Boundary Nodes')
            
            # 绘制外边界圆弧
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
