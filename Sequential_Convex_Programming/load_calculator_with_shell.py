"""
扩展的载荷计算器 - 集成壳体FEA
  
Date: 2025/9/3
"""
import numpy as np
import math
from typing import List, Optional
from .shell_fea_2d import Shell2DFEA, ShellMaterialData

class LoadCalculatorWithShell:
    """
        载荷计算器
    """
    
    def __init__(self, material_data, enable_shell=True, shell_params=None, simple_mode: bool = False):
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
        
        if enable_shell and shell_params and not self.simple_mode:
            self._initialize_shell_fea(shell_params)
        
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
            outer_radius = shell_params.get('outer_radius', 5.0)
            depth = shell_params.get('depth', 50.0)
            thickness = shell_params.get('thickness', 0.01)
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
            self.shell_fea = Shell2DFEA(
                outer_radius=outer_radius,
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

            # 壳体对支撑的反力
            rx, ry = float(support_reactions[i, 0]), float(support_reactions[i, 1])
            # 仅取反力在径向方向的分量大小（非负），忽略切向分量
            mag_shell_on_support = max(0.0, rx * nx + ry * ny)
            # 等效桁架所受力为相反方向（作用反作用），方向严格径向向内
            fx = -mag_shell_on_support * nx
            fy = -mag_shell_on_support * ny
            load_vector[2*node_idx] = fx
            load_vector[2*node_idx + 1] = fy
        
        # 计算等效基础压力（用于兼容性）
        base_pressure = self.material_data.rho_water * self.material_data.g * depth
        
        # 构建LoadData对象
        from .truss_system_initializer import LoadData
        return LoadData(
            load_vector=load_vector,
            base_pressure=base_pressure,
            depth=depth
        )
    
    def compute_load_vector(self, node_coords, outer_nodes, depth):
        """
        重新计算载荷向量（当几何改变时）
        
        这是你原有代码中用于SCP迭代的接口
        """
        if self.simple_mode:
            return self._compute_simple_load_vector(node_coords, outer_nodes, depth)
        if not self.enable_shell or self.shell_fea is None:
            raise RuntimeError("Shell FEA is required but not initialized")
        return self._compute_shell_load_vector(node_coords, outer_nodes, depth)
    
    def _compute_shell_load_vector(self, node_coords, outer_nodes, depth):
        """基于壳体的载荷向量计算"""
        coords = np.array(node_coords)
        
        # 提取外层节点坐标
        outer_node_coords = coords[outer_nodes]
        
        # 壳体FEA计算
        try:
            # 支撑位置直接使用桁架外圈节点坐标
            support_reactions = self.shell_fea.solve_with_support_positions(outer_node_coords)
            
            # 转换为载荷向量（方向强制为径向向内）
            outer_node_coords = coords[outer_nodes]
            load_vector = np.zeros(len(coords) * 2)
            for i, node_idx in enumerate(outer_nodes):
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
                load_vector[2*node_idx] = fx
                load_vector[2*node_idx + 1] = fy
            
            return load_vector
            
        except Exception as e:
            print(f"Shell load calculation failed: {e}")
            raise RuntimeError(f"Shell load calculation failed: {e}")

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

        from .truss_system_initializer import LoadData
        base_pressure = rho_g * float(depth)
        return LoadData(load_vector=load_vector, base_pressure=base_pressure, depth=depth)

    def _compute_simple_load_vector(self, node_coords, outer_nodes, depth):
        coords = np.asarray(node_coords, dtype=float)
        ln = list(outer_nodes)
        load_vector = np.zeros(coords.shape[0] * 2, dtype=float)
        if not ln:
            return load_vector
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
        return load_vector

    def _empty_load_data(self, depth):
        from .truss_system_initializer import LoadData
        return LoadData(load_vector=np.zeros(0, dtype=float), base_pressure=float(self.material_data.rho_water * self.material_data.g * depth), depth=depth)
    
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
        'thickness': 0.01,  # 1cm厚度
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
