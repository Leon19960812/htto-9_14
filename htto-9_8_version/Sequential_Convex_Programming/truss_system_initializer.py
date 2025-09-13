"""

"""
import numpy as np
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

# ============================================
# 数据结构定义
# ============================================

@dataclass # decorator to automatically generate init, repr, etc.
class GeometryData:
    """几何数据（统一：载荷节点集 load_nodes）"""
    nodes: List[List[float]]                # 节点坐标
    elements: List[List[int]]               # 单元连接
    outer_nodes: List[int]                  # 兼容字段：历史“外层”节点索引
    load_nodes: List[int]                   # 载荷节点索引（无层语义，SCP使用）
    middle_nodes: List[int]                 # 中间层节点索引
    inner_nodes: List[int]                  # 内层节点索引
    n_nodes: int                            # 节点数
    n_elements: int                         # 单元数
    n_dof: int                              # 自由度度数

@dataclass
class MaterialData:
    """材料数据"""
    E_steel: float                          # 钢材弹性模?(Pa)
    rho_steel: float                        # 钢材密度 (kg/m³)
    rho_water: float                        # 水密?(kg/m³)
    g: float                                # 重力加速度 (m/s²)
    A_min: float                            # 最小截面积
    A_max: float                            # 最大截面积
    removal_threshold: float                # 移除阈?
@dataclass
class LoadData:
    """载荷数据"""
    load_vector: np.ndarray                 # 载荷向量
    base_pressure: float                    # 基础压力
    depth: float                            # 水深

@dataclass
class ConstraintData:
    """约束数据"""
    volume_constraint: float                # 体积约束
    volume_fraction: float                  # 体积分数
    fixed_dofs: List[int]                   # 固定自由度?    free_dofs: List[int]                    # 自由度自由度?
# ============================================
# 基础计算模块
# ============================================

class GeometryCalculator:
    """几何计算模块"""
    
    def __init__(self):
        pass
    
    def generate_ground_structure(self, radius: float, n_sectors: int, 
                                    inner_ratio: float, enable_middle_layer: bool = False,
                                    middle_layer_ratio: float = 0.85,
                                    connection_level: int = 4) -> GeometryData:
            """
            生成地面结构 - 支持2层或3?            
            Parameters:
            -----------
            radius : float
                外层半径 (m)
            n_sectors : int
                扇形数量
            inner_ratio : float
                内层半径比例
            enable_middle_layer : bool
                是否启用中间?            middle_layer_ratio : float
                中间层半径比?                
            Returns:
            --------
            GeometryData
                几何数据
            """
            if enable_middle_layer:
                print("Generating 3-layer Ground Structure...")
                return self._generate_3layer_structure(radius, n_sectors, inner_ratio, middle_layer_ratio, connection_level)
            else:
                print("Generating 2-layer Ground Structure...")
                return self._generate_2layer_structure(radius, n_sectors, inner_ratio)
    
    def _generate_2layer_structure(self, radius: float, n_sectors: int, 
                                 inner_ratio: float) -> GeometryData:
        """
        生成2层结?        """
        inner_radius = radius * inner_ratio
        nodes = []
        outer_nodes = []
        inner_nodes = []
        # 外层节点
        for i in range(n_sectors + 1):
            angle = i * math.pi / n_sectors
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            nodes.append([x, y])
            outer_nodes.append(len(nodes) - 1)
        
        # 内层节点
        for i in range(n_sectors + 1):
            angle = i * math.pi / n_sectors
            x = inner_radius * math.cos(angle)
            y = inner_radius * math.sin(angle)
            nodes.append([x, y])
            inner_nodes.append(len(nodes) - 1)
        
        # 生成单元连接
        elements = []
        
        # 外层弧形连接
        for i in range(len(outer_nodes) - 1):
            elements.append([outer_nodes[i], outer_nodes[i + 1]])
        
        # 内层弧形连接
        for i in range(len(inner_nodes) - 1):
            elements.append([inner_nodes[i], inner_nodes[i + 1]])
        
        # 径向连接
        for i in range(len(outer_nodes)):
            elements.append([outer_nodes[i], inner_nodes[i]])
        
        # 斜向连接
        for i in range(len(outer_nodes) - 1):
            elements.append([inner_nodes[i], outer_nodes[i + 1]])
            elements.append([outer_nodes[i], inner_nodes[i + 1]])
        
        # level 2 连接
        for i in range(len(outer_nodes) - 2):
            elements.append([outer_nodes[i], inner_nodes[i + 2]])
            elements.append([inner_nodes[i], outer_nodes[i + 2]])
        
        n_nodes = len(nodes)
        n_elements = len(elements)
        n_dof = 2 * n_nodes #自由度度总数
        
        print(f"2-layer Ground Structure: {n_elements} members")
        
        return GeometryData(
            nodes=nodes,
            elements=elements,
            outer_nodes=outer_nodes,
            load_nodes=outer_nodes,
            inner_nodes=inner_nodes,
            middle_nodes=[],
            n_nodes=n_nodes,
            n_elements=n_elements,
            n_dof=n_dof
        )
    
    def _generate_3layer_structure(self, radius: float, n_sectors: int, 
                                 inner_ratio: float, middle_layer_ratio: float,
                                 connection_level: int = 4) -> GeometryData:
        """
        生成3层结?        """
        inner_radius = radius * inner_ratio
        middle_radius = radius * middle_layer_ratio
        
        nodes = []
        outer_nodes = []
        middle_nodes = []
        inner_nodes = []
        
        # 外层节点（半圆周边）
        for i in range(n_sectors + 1):
            angle = i * math.pi / n_sectors
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            nodes.append([x, y])
            outer_nodes.append(len(nodes) - 1)
        
        # 中间层节?        for i in range(n_sectors + 1):
            angle = i * math.pi / n_sectors
            x = middle_radius * math.cos(angle)
            y = middle_radius * math.sin(angle)
            nodes.append([x, y])
            middle_nodes.append(len(nodes) - 1)
        
        # 内层节点
        for i in range(n_sectors + 1):
            angle = i * math.pi / n_sectors
            x = inner_radius * math.cos(angle)
            y = inner_radius * math.sin(angle)
            nodes.append([x, y])
            inner_nodes.append(len(nodes) - 1)
        
        # 生成单元连接
        elements = []
        if connection_level == 1 or connection_level == 4:
        # 1. 各层弧形连接
        # 外层弧形连接
            for i in range(len(outer_nodes) - 1):
                elements.append([outer_nodes[i], outer_nodes[i + 1]])
            
            # 中间层弧形连?            for i in range(len(middle_nodes) - 1):
                elements.append([middle_nodes[i], middle_nodes[i + 1]])
            
            # 内层弧形连接
            for i in range(len(inner_nodes) - 1):
                elements.append([inner_nodes[i], inner_nodes[i + 1]])
            
            # 2. 相邻层径向连?            # 外层-中间层径向连?            for i in range(len(outer_nodes)):
                elements.append([outer_nodes[i], middle_nodes[i]])
            
            # 中间?内层径向连接
            for i in range(len(middle_nodes)):
                elements.append([middle_nodes[i], inner_nodes[i]])
            
            # 3. 相邻层斜向连?            # 外层-中间层斜向连?            for i in range(len(outer_nodes) - 1):
                elements.append([middle_nodes[i], outer_nodes[i + 1]])
                elements.append([outer_nodes[i], middle_nodes[i + 1]])
            
            # 中间?内层斜向连接
            for i in range(len(middle_nodes) - 1):
                elements.append([inner_nodes[i], middle_nodes[i + 1]])
                elements.append([middle_nodes[i], inner_nodes[i + 1]])
        
        if connection_level == 2 or connection_level == 4:
            # 外层-中间?level 2
            for i in range(len(outer_nodes) - 2):
                elements.append([outer_nodes[i], middle_nodes[i + 2]])
                elements.append([middle_nodes[i], outer_nodes[i + 2]])
            
            # 中间?内层 level 2
            for i in range(len(middle_nodes) - 2):
                elements.append([middle_nodes[i], inner_nodes[i + 2]])
                elements.append([inner_nodes[i], middle_nodes[i + 2]])

        if connection_level == 3 or connection_level == 4:
            # 外层-中间?level 3
            for i in range(len(outer_nodes) - 3):
                elements.append([outer_nodes[i], middle_nodes[i + 3]])
                elements.append([middle_nodes[i], outer_nodes[i + 3]])
            
            # 中间?内层 level 3
            for i in range(len(middle_nodes) - 3):
                elements.append([middle_nodes[i], inner_nodes[i + 3]])
                elements.append([inner_nodes[i], middle_nodes[i + 3]])
        
        n_nodes = len(nodes)
        n_elements = len(elements)
        n_dof = 2 * n_nodes
        
        print(f"3-layer Ground Structure: {n_elements} members")
        print(f"  Outer layer: {len(outer_nodes)} nodes")
        print(f"  Middle layer: {len(middle_nodes)} nodes") 
        print(f"  Inner layer: {len(inner_nodes)} nodes")
        
        return GeometryData(
            nodes=nodes,
            elements=elements,
            outer_nodes=outer_nodes,
            load_nodes=outer_nodes,
            inner_nodes=inner_nodes,
            middle_nodes=middle_nodes,
            n_nodes=n_nodes,
            n_elements=n_elements,
            n_dof=n_dof
        )
    
    def compute_element_lengths(self, geometry: GeometryData) -> np.ndarray:
        """
        计算单元长度
        
        Parameters:
        -----------
        geometry : GeometryData
            几何数据
            
        Returns:
        --------
        np.ndarray
            单元长度数组
        """
        element_lengths = []
        for node1, node2 in geometry.elements:
            x1, y1 = geometry.nodes[node1]
            x2, y2 = geometry.nodes[node2]
            L = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            element_lengths.append(L)
        return np.array(element_lengths)
    
    def compute_element_geometry(self, node_coords: np.ndarray, 
                               elements: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算单元几何属?        
        Parameters:
        -----------
        node_coords : np.ndarray
            节点坐标数组
        elements : List[List[int]]
            单元连接列表
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            (单元长度数组, 单元方向余弦数组)
        """
        coords = np.array(node_coords)
        n_elements = len(elements)
        
        element_lengths = np.zeros(n_elements)
        element_directions = np.zeros((n_elements, 2))  # [cos, sin]
        
        for i, (node1, node2) in enumerate(elements):
            # 节点坐标
            p1 = coords[node1]
            p2 = coords[node2]
            
            # 长度
            length = np.linalg.norm(p2 - p1)
            element_lengths[i] = length
            
            # 方向余弦
            if length > 1e-12:
                cos_theta = (p2[0] - p1[0]) / length
                sin_theta = (p2[1] - p1[1]) / length
            else:
                cos_theta, sin_theta = 1.0, 0.0
                
            element_directions[i] = [cos_theta, sin_theta]
        
        return element_lengths, element_directions
    
    def update_node_coordinates(self, geometry: GeometryData, theta: np.ndarray, 
                              radius: float) -> np.ndarray:
        """
        根据角度更新节点坐标
        
        Parameters:
        -----------
        geometry : GeometryData
            几何数据
        theta : np.ndarray
            外层节点角度数组
        radius : float
            外层半径
            
        Returns:
        --------
        np.ndarray
            更新后的节点坐标
        """
        coords = np.array(geometry.nodes.copy()) # 将节点坐标转为Numpy数组
        
        # 更新载荷节点集合对应的坐标（等同于优化变量顺序）
        node_list = getattr(geometry, 'load_nodes', getattr(geometry, 'outer_nodes', []))
        for i, angle in enumerate(theta):
            if i >= len(node_list):
                break
            node_id = node_list[i]
            coords[node_id] = np.array([
                radius * np.cos(angle),
                radius * np.sin(angle)
            ])

        return coords
    
    

class StiffnessCalculator:
    """刚度计算模块"""
    
    def __init__(self, material_data: MaterialData):
        self.material_data = material_data
    
    def precompute_unit_stiffness_matrices(self, geometry: GeometryData, 
                                         element_lengths: np.ndarray) -> List[np.ndarray]:
        """
        预计算单位刚度矩?        
        Parameters:
        -----------
        geometry : GeometryData
            几何数据
        element_lengths : np.ndarray
            单元长度数组
            
        Returns:
        --------
        List[np.ndarray]
            单位刚度矩阵列表
        """
        print("Precomputing unit stiffness matrices...")
        
        unit_stiffness_matrices = []
        
        for elem_id, (node1, node2) in enumerate(geometry.elements):
            # 单元几何
            x1, y1 = geometry.nodes[node1]
            x2, y2 = geometry.nodes[node2]
            L = element_lengths[elem_id]
            
            # 方向余弦
            cos_theta = (x2 - x1) / L
            sin_theta = (y2 - y1) / L
            
            # 单位截面积刚度矩?(A = 1) EA/L
            k_factor = self.material_data.E_steel / L
            
            # 局部自由度度
            dofs = [2*node1, 2*node1+1, 2*node2, 2*node2+1]
            
            # 单元刚度矩阵（局部）
            k_local = k_factor * np.array([
                [cos_theta**2, cos_theta*sin_theta, -cos_theta**2, -cos_theta*sin_theta],
                [cos_theta*sin_theta, sin_theta**2, -cos_theta*sin_theta, -sin_theta**2],
                [-cos_theta**2, -cos_theta*sin_theta, cos_theta**2, cos_theta*sin_theta],
                [-cos_theta*sin_theta, -sin_theta**2, cos_theta*sin_theta, sin_theta**2]
            ])
            
            # 扩展到全局尺寸（不是全局刚度矩阵?            K_global = np.zeros((geometry.n_dof, geometry.n_dof))
            for i in range(4):
                for j in range(4):
                    K_global[dofs[i], dofs[j]] = k_local[i, j]
            
            unit_stiffness_matrices.append(K_global)
            #这是一个列表，列表的元素是每个单元在全局坐标系下的单位刚度矩?        
        print(f"Precomputed {len(unit_stiffness_matrices)} unit stiffness matrices")
        return unit_stiffness_matrices
    
    def compute_system_matrices(self, geometry: GeometryData, theta: np.ndarray,
                              radius: float, geom_calc: GeometryCalculator,
                              load_calc, depth: float) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        计算系统矩阵
        
        Parameters:
        -----------
        geometry : GeometryData
            几何数据
        theta : np.ndarray
            角度数组
        radius : float
            外层半径
        geom_calc : GeometryCalculator
            几何计算?        load_calc : LoadCalculator
            载荷计算?        depth : float
            水深
            
        Returns:
        --------
        Tuple[List[np.ndarray], np.ndarray]
            (单元刚度矩阵列表, 载荷向量)
        """
        # 更新节点坐标
        coords = geom_calc.update_node_coordinates(geometry, theta, radius)
        
        # 计算载荷向量 - 使用壳体FEA动态计算（方向强制为径向向内）
        outer_node_coords = np.array([coords[i] for i in geometry.load_nodes])
        support_reactions = load_calc.shell_fea.solve_with_support_positions(outer_node_coords)

        # 转换为载荷向量：仅保留壳体反力的径向分量大小，方向取径向向内
        f = np.zeros(geometry.n_dof)
        for i, node_idx in enumerate(geometry.load_nodes):
            px, py = outer_node_coords[i]
            r = float(np.hypot(px, py))
            if r > 1e-12:
                nx, ny = -px / r, -py / r
            else:
                nx, ny = 0.0, -1.0
            rx, ry = float(support_reactions[i, 0]), float(support_reactions[i, 1])
            mag_shell_on_support = max(0.0, rx * nx + ry * ny)
            fx = -mag_shell_on_support * nx
            fy = -mag_shell_on_support * ny
            f[2*node_idx] = fx
            f[2*node_idx + 1] = fy
        
        # 计算单元几何
        element_lengths, element_directions = geom_calc.compute_element_geometry(coords, geometry.elements)
        
        # 计算单位刚度矩阵列表
        K_list = []
        for i, (node1, node2) in enumerate(geometry.elements):
            length = element_lengths[i]
            c, s = element_directions[i]
            
            if length < 1e-12:
                # 退化单元，零刚?                K_unit = np.zeros((geometry.n_dof, geometry.n_dof))
            else:
                # 单位刚度系数
                k_coeff = self.material_data.E_steel / length
                
                # 局部刚度矩?                k_local = k_coeff * np.array([
                    [c*c, c*s, -c*c, -c*s],
                    [c*s, s*s, -c*s, -s*s],
                    [-c*c, -c*s, c*c, c*s],
                    [-c*s, -s*s, c*s, s*s]
                ])
                
                # 扩展到全局矩阵
                K_unit = np.zeros((geometry.n_dof, geometry.n_dof))
                dofs = [2*node1, 2*node1+1, 2*node2, 2*node2+1]
                
                for m in range(4):
                    for n in range(4):
                        K_unit[dofs[m], dofs[n]] = k_local[m, n]
            
            K_list.append(K_unit)
        
        return K_list, f
    
    def assemble_global_stiffness(self, geometry: GeometryData, A: np.ndarray,
                                element_lengths: np.ndarray, element_directions: np.ndarray) -> np.ndarray:
        """
        组装全局刚度矩阵
        
        Parameters:
        -----------
        geometry : GeometryData
            几何数据
        A : np.ndarray
            截面积数?        element_lengths : np.ndarray
            单元长度数组
        element_directions : np.ndarray
            单元方向余弦数组
            
        Returns:
        --------
        np.ndarray
            全局刚度矩阵
        """
        K_global = np.zeros((geometry.n_dof, geometry.n_dof))
        
        for i, (node1, node2) in enumerate(geometry.elements):
            # 回退：装配阶段不进行硬裁剪，始终参与装配
            # 仅对截面积施加物理下?A_min，避免零刚度
            Ai = max(A[i], self.material_data.A_min)
            # 单元刚度系数
            k_coeff = self.material_data.E_steel * Ai / element_lengths[i]
            
            # 方向余弦
            c, s = element_directions[i]
            
            # 单元刚度矩阵 (4x4)
            k_local = k_coeff * np.array([
                [c*c, c*s, -c*c, -c*s],
                [c*s, s*s, -c*s, -s*s],
                [-c*c, -c*s, c*c, c*s],
                [-c*s, -s*s, c*s, s*s]
            ])
            
            # 自由度度映?            dofs = [2*node1, 2*node1+1, 2*node2, 2*node2+1]
            
            # 组装到全局矩阵
            for m in range(4):
                for n in range(4):
                    K_global[dofs[m], dofs[n]] += k_local[m, n]
        
        return K_global

class ConstraintCalculator:
    """约束计算模块"""
    
    def __init__(self):
        pass
    
    def initialize_constraints(self, geometry: GeometryData, element_lengths: np.ndarray,
                             volume_fraction: float, A_max: float) -> ConstraintData:
        """
        初始化约束参?        
        Parameters:
        -----------
        geometry : GeometryData
            几何数据
        element_lengths : np.ndarray
            单元长度数组
        volume_fraction : float
            体积分数
        A_max : float
            最大截面积
            
        Returns:
        --------
        ConstraintData
            约束数据
        """
        total_length = np.sum(element_lengths)
        ground_structure_volume = total_length * A_max
        volume_constraint = ground_structure_volume * volume_fraction # 体积约束
        
        return ConstraintData(
            volume_constraint=volume_constraint,
            volume_fraction=volume_fraction,
            fixed_dofs=[],  # 稍后设置
            free_dofs=[]    # 稍后设置
        )
    
    def setup_boundary_conditions(self, geometry: GeometryData) -> Tuple[List[int], List[int]]:
        """
        设置边界条件
        
        Parameters:
        -----------
        geometry : GeometryData
            几何数据
            
        Returns:
        --------
        Tuple[List[int], List[int]]
            (固定自由度? 自由度自由度?
        """
        # 固定支撑：取载荷节点集两端
        ln = getattr(geometry, 'load_nodes', [])
        fixed_nodes = []
        if len(ln) >= 2:
            fixed_nodes = [ln[0], ln[-1]]
        fixed_dofs = []
        for node in fixed_nodes:
            fixed_dofs.extend([2*node, 2*node+1])
        
        # 自由度
        free_dofs = [i for i in range(geometry.n_dof) if i not in fixed_dofs]
        
        return fixed_dofs, free_dofs




# ============================================
# 统一的初始化?# ============================================

class TrussSystemInitializer:
    """
    重构版本的桁架系统初始化?    使用模块化的计算组件
    """
    
    def __init__(self, radius=2.0, n_sectors=12, inner_ratio=0.7, 
                 depth=50, volume_fraction=0.2, E_steel=210e6,
                enable_middle_layer=False, middle_layer_ratio=0.85,
                use_polar: bool = True, polar_config: dict = None):
        """
        初始化桁架系?        
        Parameters:
        -----------
        radius : float
            外层半径 (m)
        n_sectors : int
            扇形数量
        inner_ratio : float
            内层半径比例
        depth : float
            水深 (m)
        volume_fraction : float
            体积约束比例
        E_steel : float
            钢材弹性模?(Pa)
        """
        # 存储参数
        self.radius = radius
        self.n_sectors = n_sectors
        self.inner_ratio = inner_ratio
        self.depth = depth
        self.volume_fraction = volume_fraction
        # 中级层参?        self.enable_middle_layer = enable_middle_layer
        self.middle_layer_ratio = middle_layer_ratio
        
        # 创建材料数据
        self.material_data = MaterialData(
            E_steel=E_steel,
            rho_steel=7850,
            rho_water=1025,
            g=9.81,
            A_min=1e-6,
            A_max=1e-2,
            removal_threshold=1e-5
        )
        
        # 创建计算?        self.geometry_calc = GeometryCalculator()

        try:
            from load_calculator_with_shell import LoadCalculatorWithShell
            shell_params = {
                'outer_radius': self.radius,
                'depth': self.depth,
                'thickness': 0.01,
                'n_circumferential': 20,  # 统一使用20个周向网?                'n_radial': 4,  # 统一使用2个径向网?                'E_shell': self.material_data.E_steel * 1000
            }
            self.load_calc = LoadCalculatorWithShell(
                material_data=self.material_data,
                enable_shell=True,
                shell_params=shell_params
            )
            print("[OK] Using shell-based load calculation")
        except ImportError as e:
            print(f"[ERROR] Failed to import LoadCalculatorWithShell: {e}")
            raise ImportError("Shell_based load calculation is required")


        self.stiffness_calc = StiffnessCalculator(self.material_data)
        self.constraint_calc = ConstraintCalculator()
        
        # 执行初始化（极坐标几何，无回退）
        self._initialize_from_polar(polar_config or {})

    def _initialize_from_polar(self, polar_cfg: dict):
        """使用极坐标几何作为几何来源并适配为 GeometryData"""
        try:
            from polar_geometry import PolarConfig as _PolarConfig, PolarGeometry as _PolarGeometry
        except Exception as e:
            raise RuntimeError(f"polar_geometry import failed: {e}")

        # 构建配置（默认两环：outer/inner），允许外部传入覆盖
        rings = polar_cfg.get('rings') or [
            {'radius': self.radius, 'n_nodes': self.n_sectors + 1, 'type': 'outer'},
            {'radius': self.radius * self.inner_ratio, 'n_nodes': self.n_sectors + 1, 'type': 'inner'},
        ]
        pc = _PolarConfig(rings=rings)
        pg = _PolarGeometry(pc)

        # 提取节点与连接
        nodes_xy = []
        load_nodes = []
        for i, n in enumerate(pg.nodes):
            nodes_xy.append([float(n.x), float(n.y)])
            if getattr(n, 'node_type', '') == 'outer':
                load_nodes.append(i)
        elements = [[int(i), int(j)] for (i, j) in pg.connections]

        n_nodes = len(nodes_xy)
        n_elements = len(elements)
        n_dof = 2 * n_nodes

        # 适配到 GeometryData
        self.geometry = GeometryData(
            nodes=nodes_xy,
            elements=elements,
            outer_nodes=load_nodes,  # 兼容字段
            load_nodes=load_nodes,
            inner_nodes=[],
            middle_nodes=[],
            n_nodes=n_nodes,
            n_elements=n_elements,
            n_dof=n_dof
        )

        # 后续流程与传统初始化一致
        self.element_lengths = self.geometry_calc.compute_element_lengths(self.geometry)
        self.load_data = self.load_calc.compute_hydrostatic_loads(
            self.geometry, self.depth, self.radius, self.geometry.nodes
        )
        self.unit_stiffness_matrices = self.stiffness_calc.precompute_unit_stiffness_matrices(
            self.geometry, self.element_lengths
        )
        self.constraint_data = self.constraint_calc.initialize_constraints(
            self.geometry, self.element_lengths, self.volume_fraction, self.material_data.A_max
        )
        fixed_dofs, free_dofs = self.constraint_calc.setup_boundary_conditions(self.geometry)
        self.constraint_data.fixed_dofs = fixed_dofs
        self.constraint_data.free_dofs = free_dofs
        self._setup_legacy_attributes()
        self._print_initialization_info()
    
    def _initialize_system(self):
        """执行系统初始?""
        # 1. 生成几何结构
        self.geometry = self.geometry_calc.generate_ground_structure(
            self.radius, self.n_sectors, self.inner_ratio,
            self.enable_middle_layer, self.middle_layer_ratio
        )
        
        # 2. 计算单元长度
        self.element_lengths = self.geometry_calc.compute_element_lengths(self.geometry)
        
        # 3. 计算载荷
        self.load_data = self.load_calc.compute_hydrostatic_loads(
            self.geometry, self.depth, self.radius, self.geometry.nodes
        )
        
        # 4. 预计算刚度矩?        self.unit_stiffness_matrices = self.stiffness_calc.precompute_unit_stiffness_matrices(
            self.geometry, self.element_lengths
        )
        
        # 5. 初始化约?        self.constraint_data = self.constraint_calc.initialize_constraints(
            self.geometry, self.element_lengths, self.volume_fraction, self.material_data.A_max
        )
        
        # 6. 设置边界条件
        fixed_dofs, free_dofs = self.constraint_calc.setup_boundary_conditions(self.geometry)
        self.constraint_data.fixed_dofs = fixed_dofs
        self.constraint_data.free_dofs = free_dofs
        
        # 7. 为兼容性保留原有属?        self._setup_legacy_attributes()
        
        # 8. 打印初始化信?        self._print_initialization_info()
    
    def _setup_legacy_attributes(self):
        """设置兼容性属性，保证原有代码正常运行"""
        # 几何属?        self.nodes = self.geometry.nodes
        self.elements = self.geometry.elements
        self.outer_nodes = self.geometry.outer_nodes
        self.load_nodes = getattr(self.geometry, 'load_nodes', self.geometry.outer_nodes)
        self.inner_nodes = self.geometry.inner_nodes
        self.n_nodes = self.geometry.n_nodes
        self.n_elements = self.geometry.n_elements
        self.n_dof = self.geometry.n_dof

        #中间?        self.middle_nodes = self.geometry.middle_nodes
        
        # 载荷属?        self.load_vector = self.load_data.load_vector
        self.base_pressure = self.load_data.base_pressure
        
        # 材料属?        self.E_steel = self.material_data.E_steel
        self.rho_steel = self.material_data.rho_steel
        self.rho_water = self.material_data.rho_water
        self.g = self.material_data.g
        self.A_min = self.material_data.A_min
        self.A_max = self.material_data.A_max
        self.removal_threshold = self.material_data.removal_threshold
        
        # 约束属?        self.volume_constraint = self.constraint_data.volume_constraint
        self.fixed_dofs = self.constraint_data.fixed_dofs
        self.free_dofs = self.constraint_data.free_dofs
        
        # 结构几何参数 (保存原始参数供后续使?
        self.inner_radius = self.radius * self.inner_ratio

        # 新增：中间层半径（如果启用）
        if self.enable_middle_layer:
            self.middle_radius = self.radius * self.middle_layer_ratio
        else:
            self.middle_radius = None
        
        # 可视化所需的属?(初始化为None，优化完成后会设?
        self.final_areas = None
        self.final_compliance = None
        self.verification_passed = None
        self.final_angles = None
        
        # 优化状态属?(用于SCP算法)
        self.current_angles = None
        self.current_areas = None
        self.current_compliance = None
    
    def _print_initialization_info(self):
        """打印初始化信?""
        print(f"Structure: R={self.radius}m, Sectors={self.n_sectors}, Inner ratio={self.inner_ratio}")
        # middle layer is deprecated in runtime
        print(f"Depth: {self.depth}m, Pressure: {self.base_pressure/1000:.1f} kPa")
        print(f"Volume fraction: {self.volume_fraction:.1f}")
        print(f"Nodes: {self.n_nodes}, Elements: {self.n_elements}")
        print(f"DOF: {self.n_dof}")
        print(f"Load nodes: {len(getattr(self.geometry, 'load_nodes', []))}")
        print(f"边界条件设置:")
        ln = getattr(self.geometry, 'load_nodes', [])
        if len(ln) >= 2:
            print(f"  固定节点: {[ln[0], ln[-1]]}")
        print(f"  固定自由度: {len(self.fixed_dofs)} 个")
        print(f"  自由自由度: {len(self.free_dofs)} 个")
    
    def group_nodes_by_radius(self, theta: np.ndarray, radius: float = None) -> List[List[int]]:
        """
        根据半径分组节点 - 委托给专门的节点融合器（支持多层theta字典?        
        Parameters:
        -----------
        theta_dict : dict
            分层角度字典，格? {'theta_outer': array, 'theta_middle': array, 'theta_inner': array}
        radius : float, optional
            融合半径，如果为None则使用默认?            
        Returns:
        --------
        List[List[int]]
            需要融合的节点组列表（所有层?        """
        from node_merger import NodeMerger
        
        # 创建节点融合?        merger = NodeMerger(self.geometry, self.radius, self.constraint_calc)
        
        # 执行多层节点分组
        return merger.group_nodes_by_radius(theta, radius)
    
    def merge_node_groups(self, theta_dict: dict, A: np.ndarray, 
                        merge_groups: List[List[int]]) -> dict:
        """
        融合节点?- 委托给专门的节点融合器（支持多层theta字典?        
        Parameters:
        -----------
        theta_dict : dict
            当前分层角度字典
        A : np.ndarray
            当前截面积数?        merge_groups : List[List[int]]
            需要融合的节点?            
        Returns:
        --------
        dict
            融合信息
        """
        from node_merger import NodeMerger
        
        # 创建节点融合?        merger = NodeMerger(self.geometry, self.radius, self.constraint_calc)
        
        # 执行外层融合（一维theta?        result = merger.merge_node_groups(theta, A, merge_groups)
        
        # 转换为字典格式以保持兼容?        return {
            'merged_pairs': result.merged_pairs,
            'removed_members': result.removed_members,
            'theta_updated': result.theta_updated,  # 现在是字典格?            'A_updated': result.A_updated,
            'geometry_updated': result.geometry_updated,
            'structure_modified': result.structure_modified
        }
    

