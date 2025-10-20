import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import math
import time
import os
import sys
scp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Sequential_Convex_Programming")
if scp_dir not in sys.path:
    sys.path.insert(0, scp_dir)
from scipy.sparse import csc_matrix
# 新增：引入统一载荷计算器
from load_calculator_with_shell import LoadCalculatorWithShell

class SDPTrussOptimizer:   
    def __init__(self, radius=2.0, n_sectors=12, inner_ratio=0.7, middle_layer_ratio = 0.8, 
                 depth=50, volume_fraction=0.1, initializer=None,
                 enable_shell=False, shell_params=None):
        """
        Parameters: 
        -----------
        radius : float                     外层半径 (m)
        n_sectors : int                    扇形分割数
        inner_ratio : float                内层半径比例
        depth : float                      水深 (m)
        volume_constraint : float          体积约束 (m³)
        enable_shell : bool                是否启用壳体分担荷载
        shell_params : dict                壳体参数
        -----------
        """
        self.enable_shell = enable_shell
        # 新增：兼容统一初始化器的分支
        if initializer is not None:
            print("Using unified initializer for SDP...")
            self._init_from_initializer(initializer)
            return
        # 基本参数
        self.radius = radius
        self.inner_radius = radius * inner_ratio
        self.middle_radius = radius * middle_layer_ratio
        self.n_sectors = n_sectors
        self.depth = depth
        self.volume_fraction = volume_fraction
        # 材料属性
        self.E_steel = 210e6 # Pa
        self.rho_steel = 7850  # kg/m³
        self.A_min = 1e-6   # 最小截面积
        self.A_max = 0.01   # 最大截面积
        self.removal_threshold = 1e-5  # 移除阈值
        # 显示阈值：仅用于可视化，默认为删除阈值的3倍
        self.display_threshold = 5 * self.removal_threshold
        # 生成结构
        self._generate_structure()
        # 计算单元长度
        self.element_lengths = np.array([
            math.sqrt((self.nodes[n1][0] - self.nodes[n2][0])**2 + (self.nodes[n1][1] - self.nodes[n2][1])**2)
            for n1, n2 in self.elements
        ])
        total_length = np.sum(self.element_lengths)
        ground_structure_volume = total_length * self.A_max
        self.volume_constraint = ground_structure_volume * volume_fraction
        self.n_nodes = len(self.nodes)
        self.n_elements = len(self.elements)
        self.n_dof = 2 * self.n_nodes
        # 新增：统一载荷计算器（与SCP一致）
        if shell_params is None:
            shell_params = {
                # Set outer radius so inner wall (outer_radius - thickness) matches truss radius
                'outer_radius': radius + 0.01,
                'depth': depth,
                'thickness': 0.01,
                'n_circumferential': n_sectors + 1,
                'n_radial': 2,
                'E_shell': 210e9
            }
        # 构造材料数据对象（只需rho_water和g）
        class Mat:
            rho_water = 1025
            g = 9.81
        self.load_calc = LoadCalculatorWithShell(
            material_data=Mat(),
            enable_shell=enable_shell,
            shell_params=shell_params
        )
        # 统一通过load_calc生成荷载
        class Geom:
            nodes = self.nodes
            outer_nodes = self.outer_nodes
            n_dof = self.n_dof
        load_data = self.load_calc.compute_hydrostatic_loads(
            Geom(), self.depth, self.radius
        )
        self.load_vector = load_data.load_vector
        self.base_pressure = load_data.base_pressure
        # 预计算单位刚度矩阵
        self._precompute_unit_stiffness_matrices()
        print(f"Structure: R={radius}m, Sectors={n_sectors}, Inner ratio={inner_ratio}")
        print(f"Depth: {depth}m, Pressure: {self.base_pressure/1000:.1f} kPa")
        print(f"Volume fraction: {volume_fraction:.1f}")
        print(f"Nodes: {self.n_nodes}, Elements: {self.n_elements}")
        print(f"DOF: {self.n_dof}")
    

    def _init_from_initializer(self, initializer):
        """从统一初始化器设置参数"""
        # 复制所有必要属性
        self.radius = initializer.radius
        self.inner_radius = initializer.inner_radius
        self.n_sectors = initializer.n_sectors
        self.depth = initializer.depth
        self.volume_fraction = initializer.volume_fraction
        
        # 材料属性
        self.E_steel = initializer.E_steel
        self.rho_steel = initializer.rho_steel
        self.A_min = initializer.A_min
        self.A_max = initializer.A_max
        self.removal_threshold = initializer.removal_threshold
        # 显示阈值：若初始化器未提供，则采用删除阈值的3倍
        self.display_threshold = getattr(initializer, 'display_threshold', 3 * self.removal_threshold)
        
        # 水压计算
        self.base_pressure = initializer.base_pressure
        
        # 几何和系统属性
        self.nodes = initializer.nodes
        self.elements = initializer.elements
        self.outer_nodes = initializer.outer_nodes
        self.inner_nodes = initializer.inner_nodes
        self.n_nodes = initializer.n_nodes
        self.n_elements = initializer.n_elements
        self.n_dof = initializer.n_dof
        self.element_lengths = initializer.element_lengths
        
        # 添加中间层节点支持
        if hasattr(initializer, 'middle_nodes'):
            self.middle_nodes = initializer.middle_nodes
            if hasattr(initializer, 'middle_radius'):
                self.middle_radius = initializer.middle_radius
        else:
            self.middle_nodes = []
            self.middle_radius = None
        
        # 载荷和约束
        self.load_vector = initializer.load_vector
        self.volume_constraint = initializer.volume_constraint
        self.free_dofs = initializer.free_dofs
        
        # 预计算的刚度矩阵
        self.unit_stiffness_matrices = initializer.unit_stiffness_matrices
    
        print(f"SDP initialized from unified system:")
        print(f"  Nodes: {self.n_nodes}, Elements: {self.n_elements}")
        print(f"  Volume constraint: {self.volume_constraint*1e6:.1f} cm³")
        
    def _generate_structure(self):
        """生成ground structure"""
        print("Generating Ground Structure...")
        
        # 生成节点
        self.nodes = []
        self.outer_nodes = []
        self.middle_nodes = []
        self.inner_nodes = []
        
        # 外层节点（半圆周边）
        for i in range(self.n_sectors + 1):
            angle = i * math.pi / self.n_sectors
            x = self.radius * math.cos(angle)
            y = self.radius * math.sin(angle)
            self.nodes.append([x, y])
            self.outer_nodes.append(len(self.nodes) - 1)
        
        # 中间层节点
        for i in range(self.n_sectors + 1):
            angle = i * math.pi / self.n_sectors
            x = self.middle_radius * math.cos(angle)
            y = self.middle_radius * math.sin(angle)
            self.nodes.append([x, y])
            self.middle_nodes.append(len(self.nodes) - 1)
            
        # 内层节点
        for i in range(self.n_sectors + 1):
            angle = i * math.pi / self.n_sectors
            x = self.inner_radius * math.cos(angle)
            y = self.inner_radius * math.sin(angle)
            self.nodes.append([x, y])
            self.inner_nodes.append(len(self.nodes) - 1)
        
        # 生成单元连接
        self.elements = []
        
        # 外层弧形连接
        for i in range(len(self.outer_nodes) - 1):
            self.elements.append([self.outer_nodes[i], self.outer_nodes[i + 1]])
        # 中间层弧形连接
        for i in range(len(self.middle_nodes) - 1):
            self.elements.append([self.middle_nodes[i], self.middle_nodes[i + 1]])
        # 内层弧形连接
        for i in range(len(self.inner_nodes) - 1):
            self.elements.append([self.inner_nodes[i], self.inner_nodes[i + 1]])
        
        # 径向连接
        for i in range(len(self.outer_nodes)):
            self.elements.append([self.outer_nodes[i], self.middle_nodes[i]])
        for i in range(len(self.middle_nodes)):
            self.elements.append([self.middle_nodes[i], self.inner_nodes[i]])

        # 斜向连接
        for i in range(len(self.outer_nodes) - 1):
            self.elements.append([self.inner_nodes[i], self.middle_nodes[i + 1]])
            self.elements.append([self.middle_nodes[i], self.inner_nodes[i + 1]])

        for i in range(len(self.outer_nodes) - 1):
            self.elements.append([self.outer_nodes[i], self.middle_nodes[i + 1]])
            self.elements.append([self.middle_nodes[i], self.outer_nodes[i + 1]])
        
        # level 2 连接
        for i in range(len(self.outer_nodes) - 2):
            self.elements.append([self.outer_nodes[i], self.middle_nodes[i + 2]])
            self.elements.append([self.middle_nodes[i], self.outer_nodes[i + 2]])

        for i in range(len(self.middle_nodes) - 2):
            self.elements.append([self.inner_nodes[i], self.middle_nodes[i + 2]])
            self.elements.append([self.middle_nodes[i], self.inner_nodes[i + 2]])

        # level 3 连接
        for i in range(len(self.outer_nodes) - 3):
            self.elements.append([self.outer_nodes[i], self.middle_nodes[i + 3]])
            self.elements.append([self.middle_nodes[i], self.outer_nodes[i + 3]])
        
        for i in range(len(self.middle_nodes) - 3):
            self.elements.append([self.middle_nodes[i], self.inner_nodes[i + 3]])
            self.elements.append([self.inner_nodes[i], self.middle_nodes[i + 3]])
        
        self.n_nodes = len(self.nodes)
        self.n_elements = len(self.elements)
        self.n_dof = 2 * self.n_nodes
        
        # 计算单元长度
        self.element_lengths = []
        for node1, node2 in self.elements:
            x1, y1 = self.nodes[node1]
            x2, y2 = self.nodes[node2]
            L = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            self.element_lengths.append(L)
        self.element_lengths = np.array(self.element_lengths)
        
        print(f"3-layer Ground Structure: {self.n_elements} members")
        print(f"  Outer layer: {len(self.outer_nodes)} nodes")
        print(f"  Middle layer: {len(self.middle_nodes)} nodes") 
        print(f"  Inner layer: {len(self.inner_nodes)} nodes")

        
    def _precompute_unit_stiffness_matrices(self):
        """预计算每个单元的单位刚度矩阵 - 添加数值稳定性检查"""
        print("Precomputing unit stiffness matrices...")
        
        self.unit_stiffness_matrices = []
        invalid_elements = []
        
        for elem_id, (node1, node2) in enumerate(self.elements):
            # 单元几何
            x1, y1 = self.nodes[node1]
            x2, y2 = self.nodes[node2]
            L = self.element_lengths[elem_id]
            
            # 检查长度有效性
            if L < 1e-12:
                print(f"Warning: Element {elem_id} has very small length: {L:.2e}")
                invalid_elements.append(elem_id)
                # 创建零刚度矩阵
                K_global = np.zeros((self.n_dof, self.n_dof))
                self.unit_stiffness_matrices.append(K_global)
                continue
            
            # 方向余弦
            cos_theta = (x2 - x1) / L
            sin_theta = (y2 - y1) / L
            
            # 检查方向余弦的有效性
            if np.isnan(cos_theta) or np.isnan(sin_theta) or np.isinf(cos_theta) or np.isinf(sin_theta):
                print(f"Warning: Element {elem_id} has invalid direction cosines")
                invalid_elements.append(elem_id)
                K_global = np.zeros((self.n_dof, self.n_dof))
                self.unit_stiffness_matrices.append(K_global)
                continue
            
            # 单位刚度矩阵 (E*A = 1)
            k_factor = self.E_steel / L
            
            # 检查刚度系数有效性
            if np.isnan(k_factor) or np.isinf(k_factor):
                print(f"Warning: Element {elem_id} has invalid stiffness factor: {k_factor}")
                invalid_elements.append(elem_id)
                K_global = np.zeros((self.n_dof, self.n_dof))
                self.unit_stiffness_matrices.append(K_global)
                continue
            
            # 局部自由度
            dofs = [2*node1, 2*node1+1, 2*node2, 2*node2+1]
            
            # 单元刚度矩阵（局部）
            k_local = k_factor * np.array([
                [cos_theta**2, cos_theta*sin_theta, -cos_theta**2, -cos_theta*sin_theta],
                [cos_theta*sin_theta, sin_theta**2, -cos_theta*sin_theta, -sin_theta**2],
                [-cos_theta**2, -cos_theta*sin_theta, cos_theta**2, cos_theta*sin_theta],
                [-cos_theta*sin_theta, -sin_theta**2, cos_theta*sin_theta, sin_theta**2]
            ])
            
            # 检查局部刚度矩阵
            if np.any(np.isnan(k_local)) or np.any(np.isinf(k_local)):
                print(f"Warning: Element {elem_id} has invalid local stiffness matrix")
                invalid_elements.append(elem_id)
                K_global = np.zeros((self.n_dof, self.n_dof))
                self.unit_stiffness_matrices.append(K_global)
                continue
            
            # 扩展到全局刚度矩阵
            K_global = np.zeros((self.n_dof, self.n_dof))
            for i in range(4):
                for j in range(4):
                    K_global[dofs[i], dofs[j]] = k_local[i, j]
            
            self.unit_stiffness_matrices.append(K_global)
        
        print(f"Precomputed {self.n_elements} unit stiffness matrices")
        if invalid_elements:
            print(f"Warning: {len(invalid_elements)} elements have numerical issues")
        
    def _apply_boundary_conditions(self, K_matrix, f_vector):
        """施加边界条件"""
        # 固定支撑：内层两端完全固定
        fixed_nodes = [self.inner_nodes[0], self.inner_nodes[-1]]
        
        fixed_dofs = []
        for node in fixed_nodes:
            fixed_dofs.extend([2*node, 2*node+1])
        
        # 自由度
        free_dofs = [i for i in range(self.n_dof) if i not in fixed_dofs]
        self.free_dofs = free_dofs
        
        # 缩减刚度矩阵和荷载向量
        if hasattr(K_matrix, 'value'):  # CVXPY variable
            # 创建选择矩阵来提取自由度
            n_free = len(free_dofs)
            selection_matrix = np.zeros((n_free, self.n_dof))
            for i, dof in enumerate(free_dofs):
                selection_matrix[i, dof] = 1.0
            
            # 使用矩阵乘法来提取子矩阵
            S = cp.Constant(selection_matrix)
            K_reduced = S @ K_matrix @ S.T
        else:  # numpy array
            K_reduced = K_matrix[np.ix_(free_dofs, free_dofs)]
        
        f_reduced = f_vector[free_dofs]
        
        return K_reduced, f_reduced
        
    def solve_sdp_optimization(self):
        """
        求解SDP问题 - 增强数值稳定性
        """
        print("\n" + "=" * 60)
        print("SOLVING SDP PROBLEM")
        print("=" * 60)
        print("Formulation: minimize f^T K^(-1) f")
        print("Method: Schur Complement + Semidefinite Programming")
        print("Variables: Cross-sectional areas A_i")
        print("Constraint: K = ∑(A_i * K_i) where K_i are unit stiffness matrices")
        print()
        
        # 预检查单位刚度矩阵
        print("Step 0: Checking unit stiffness matrices...")
        for i, K_unit in enumerate(self.unit_stiffness_matrices):
            if np.any(np.isnan(K_unit)) or np.any(np.isinf(K_unit)):
                print(f"Error: Unit stiffness matrix {i} contains NaN/Inf values")
                return None, None
        print("✓ All unit stiffness matrices are valid")
        
        # 检查载荷向量
        if np.any(np.isnan(self.load_vector)) or np.any(np.isinf(self.load_vector)):
            print("Error: Load vector contains NaN/Inf values")
            return None, None
        print("✓ Load vector is valid")
        
        # 1. 定义优化变量
        print("Step 1: Defining optimization variables...")
        A = cp.Variable(self.n_elements, pos=True)
        t = cp.Variable()
        
        # 2. 构建全局刚度矩阵作为A的线性组合
        print("Step 2: Building global stiffness matrix as linear combination...")
        
        # 使用更稳定的方式构建刚度矩阵
        K_global_list = []
        for i in range(self.n_elements):
            K_i = cp.Constant(self.unit_stiffness_matrices[i])
            K_global_list.append(A[i] * K_i)
        
        K_global = cp.sum(K_global_list)
        
        # 3. 施加边界条件
        print("Step 3: Applying boundary conditions...")
        K_reduced, f_reduced = self._apply_boundary_conditions(K_global, self.load_vector)
        
        # 4. 构建Schur complement约束
        print("Step 4: Building Schur complement constraint...")
        n_free = len(self.free_dofs)
        
        # 确保f_reduced是CVXPY兼容的
        f_reduced_cvxpy = cp.Constant(f_reduced)
        
        # 构建块矩阵 [K  f ]
        #            [f^T t]
        schur_matrix = cp.bmat([
            [K_reduced, f_reduced_cvxpy.reshape((-1, 1))],
            [f_reduced_cvxpy.reshape((1, -1)), cp.reshape(t, (1, 1))]
        ])
        
        print(f"   Schur matrix dimensions: {n_free+1} × {n_free+1}")
        print(f"   K_reduced shape: {n_free} × {n_free}")
        print(f"   f_reduced length: {len(f_reduced)}")
        
        # 5. 定义约束
        print("Step 5: Setting up constraints...")
        constraints = []
        
        # 半正定约束 (SDP核心)
        constraints.append(schur_matrix >> 0)
        print("   ✓ Added semidefinite constraint")
        
        # 截面积约束 - 使用更保守的下界
        A_min_safe = max(self.A_min, 1e-8)  # 避免过小的下界
        constraints.append(A >= A_min_safe)
        constraints.append(A <= self.A_max)
        print(f"   ✓ Added area bounds: [{A_min_safe:.2e}, {self.A_max:.2e}]")
        
        # 体积约束
        constraints.append(cp.sum(cp.multiply(A, self.element_lengths)) <= self.volume_constraint)
        print(f"   ✓ Added volume constraint: ≤ {self.volume_constraint*1e6:.1f} cm³")
        
        # 6. 目标函数：最小化compliance
        objective = cp.Minimize(t)
        print("   ✓ Objective: minimize compliance t")
        
        # 7. 构建和求解问题
        print("\nStep 6: Constructing SDP problem...")
        prob = cp.Problem(objective, constraints)
        
        print(f"Problem statistics:")
        print(f"   Variables: {self.n_elements} (areas) + 1 (compliance)")
        print(f"   SDP matrix size: {n_free + 1} × {n_free + 1}")
        print(f"   Linear constraints: {len(constraints) - 1}")
        print(f"   SDP constraints: 1")
        
        # 8. 求解
        print("\nStep 7: Solving SDP...")
        print("Using MOSEK solver...")

        if 'MOSEK' not in cp.installed_solvers():
            print("✗ MOSEK solver not found!")
            return None, None

        try:
            print(f"\nSolving with MOSEK...")
            start_time = time.time()
            
            # 使用更稳定的求解器参数
            prob.solve(
                solver=cp.MOSEK, 
                verbose=True,
                mosek_params={
                    'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-8,
                    'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-8,
                    'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-8
                }
            )
            
            solve_time = time.time() - start_time
            print(f"Solve time: {solve_time:.2f} seconds")
            print(f"Status: {prob.status}")
            
            if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                print(f"✓ SDP solved successfully!")
                print(f"✓ Optimal compliance: {prob.value:.6e}")
                
                # 检查解的质量
                if A.value is None:
                    print("✗ Solution variables are None")
                    return None, None
                    
                optimal_areas = A.value
                optimal_compliance = prob.value
                
                # 检查解的合理性
                if np.any(optimal_areas < 0):
                    print("✗ Negative areas found in solution")
                    return None, None
                
                if np.any(np.isnan(optimal_areas)) or np.any(np.isinf(optimal_areas)):
                    print("✗ NaN or Inf values in solution")
                    return None, None
                
                print(f"✓ Solution quality checks passed")
                
                # 验证结果
                self._verify_solution(optimal_areas, optimal_compliance)
                
                return optimal_areas, optimal_compliance
                
            else:
                print(f"✗ MOSEK failed with status: {prob.status}")
                return None, None
                
        except Exception as e:
            print(f"✗ MOSEK failed with exception: {e}")
            print("Debugging information:")
            print(f"  Problem variables: {prob.variables}")
            print(f"  Problem size: {prob.size_metrics}")
            return None, None
    
    def _verify_solution(self, areas, compliance):
        """验证SDP求解结果"""
        print("\n" + "-" * 40)
        print("SOLUTION VERIFICATION")
        print("-" * 40)
        
        # 重构刚度矩阵
        K_reconstructed = np.sum([areas[i] * self.unit_stiffness_matrices[i] 
                                 for i in range(self.n_elements)], axis=0)
        
        # 施加边界条件 - 直接使用numpy索引
        free_dofs = self.free_dofs
        K_red = K_reconstructed[np.ix_(free_dofs, free_dofs)]
        f_red = self.load_vector[free_dofs]
        
        try:
            # 检查刚度矩阵的条件数
            cond_num = np.linalg.cond(K_red)
            print(f"Stiffness matrix condition number: {cond_num:.2e}")
            
            if cond_num > 1e12:
                print("Warning: Ill-conditioned stiffness matrix")
            
            # 直接求解验证
            U_red = np.linalg.solve(K_red, f_red)
            compliance_direct = np.dot(f_red, U_red)
            
            print(f"SDP compliance:    {compliance:.6e}")
            print(f"Direct compliance: {compliance_direct:.6e}")
            print(f"Relative error:    {abs(compliance - compliance_direct)/compliance:.6e}")
            
            # 检查结构参数
            total_volume = np.sum(areas * self.element_lengths)
            effective_elements = np.sum(areas > self.removal_threshold)
            
            print(f"Total volume:      {total_volume*1e6:.1f} cm³")
            print(f"Volume constraint: {self.volume_constraint*1e6:.1f} cm³")
            print(f"Volume utilization: {total_volume/self.volume_constraint:.1%}")
            print(f"Effective elements: {effective_elements}/{self.n_elements}")
            
            # 检查最小和最大截面积
            active_areas = areas[areas > self.removal_threshold]
            if len(active_areas) > 0:
                print(f"Area range: [{np.min(active_areas)*1e6:.2f}, {np.max(active_areas)*1e6:.2f}] mm²")
            
            # 保存结果
            self.final_areas = areas
            self.final_compliance = compliance
            self.verification_passed = abs(compliance - compliance_direct)/compliance < 1e-3
            
            if self.verification_passed:
                print("✓ Verification PASSED")
            else:
                print("✗ Verification FAILED - Large discrepancy in compliance")
            
        except Exception as e:
            print(f"✗ Verification failed: {e}")
            self.verification_passed = False
    
    def visualize_results(self):
        """可视化SDP优化结果"""
        if not hasattr(self, 'final_areas'):
            print("✗ No optimization results to visualize!")
            print("Please run optimization first!")
            return
        
        print("\n" + "=" * 50)
        print("GENERATING VISUALIZATION")
        print("=" * 50)
        
        try:
            # 设置matplotlib后端
            import matplotlib
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt
            
            # 创建结果目录
            results_dir = "results"
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
                print(f"Created results directory: {results_dir}")
            
            fig = plt.figure(figsize=(18, 12))
            print("Created figure...")
            
            # 子图1：初始ground structure
            ax1 = plt.subplot(2, 3, 1)
            initial_areas = np.ones(self.n_elements) * np.mean(self.final_areas)
            self._plot_structure(ax1, initial_areas, "Initial Ground Structure", 
                               linewidth_mode='uniform')
            print("Plotted initial structure...")
            
            # 子图2：SDP优化结果
            ax2 = plt.subplot(2, 3, 2)
            self._plot_structure(ax2, self.final_areas, "SDP Optimized Structure", 
                               linewidth_mode='variable')
            print("Plotted optimized structure...")
            
            # 子图3：拓扑清理后
            ax3 = plt.subplot(2, 3, 3)
            cleaned_areas = self.final_areas.copy()
            cleaned_areas[cleaned_areas < self.removal_threshold] = 0
            self._plot_structure(ax3, cleaned_areas, "After Topology Cleanup", 
                               linewidth_mode='variable')
            print("Plotted cleaned structure...")
            
            # 子图4：截面积分布
            ax4 = plt.subplot(2, 3, 4)
            areas_mm2 = self.final_areas * 1e6
            valid_areas = areas_mm2[areas_mm2 > self.removal_threshold * 1e6]
            
            if len(valid_areas) > 0:
                num_bins = max(1, min(25, len(valid_areas)))
                ax4.hist(valid_areas, bins=num_bins, alpha=0.7,
                        color='skyblue', edgecolor='black')
                ax4.axvline(x=self.removal_threshold*1e6, color='red', linestyle='--',
                           label='Removal Threshold')
                a_max = float(getattr(self, 'A_max', 1e-2))
                if not np.isfinite(a_max) or a_max <= 0:
                    a_max = 1e-2
                a_max_mm2 = max(a_max * 1e6, float(np.max(valid_areas)))
                ax4.set_xlim(0, a_max_mm2)
                ax4.set_xticks(np.linspace(0, a_max_mm2, 11))
                ax4.set_xlabel('Cross-sectional Area (mm²)', fontsize=14)
                ax4.set_ylabel('Number of Members', fontsize=14)
                ax4.set_title('Area Distribution')
                ax4.tick_params(axis='both', labelsize=14)
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, 'No valid areas\nto display', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Area Distribution')
            print("Plotted area distribution...")
            
            # 子图5：荷载分布
            ax5 = plt.subplot(2, 3, 5)
            self._plot_loads(ax5)
            print("Plotted load distribution...")
            
            # 子图6：总结统计
            ax6 = plt.subplot(2, 3, 6)
            ax6.axis('off')
            
            effective_elements = np.sum(self.final_areas > self.removal_threshold)
            total_volume = np.sum(self.final_areas * self.element_lengths)
            
            # 检测结构类型
            is_3layer = hasattr(self, 'middle_nodes') and len(self.middle_nodes) > 0
            structure_type = "3-layer" if is_3layer else "2-layer"
            
            summary_text = f"""SDP Optimization Results

Method: Semidefinite Programming
Formulation: Schur Complement

Structure:
• Type: {structure_type} structure
• Radius: {self.radius}m
• Sectors: {self.n_sectors}
• Depth: {self.depth}m
• Total Members: {self.n_elements}

Results:
• Optimal Compliance: {self.final_compliance:.3e}
• Effective Members: {effective_elements}
• Removed Members: {self.n_elements - effective_elements}
• Total Volume: {total_volume*1e6:.1f} cm³
• Volume Limit: {self.volume_constraint*1e6:.1f} cm³
• Volume Utilization: {total_volume/self.volume_constraint:.1%}

Verification:
• Status: {'✓ Passed' if self.verification_passed else '✗ Failed'}
• Method: Direct stiffness calculation
            """
            
            ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            print("Added summary statistics...")
            
            plt.tight_layout()
            
            # 保存图像
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            fig_path = os.path.join(results_dir, f"fixed_sdp_results_{timestamp}.png")
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"✓ Results saved to: {fig_path}")
            
            # 显示图形
            print("Displaying figure...")
            plt.show(block=True)
            
            # 单独导出面积分布直方图（SDP）
            try:
                areas_mm2 = self.final_areas * 1e6
                valid_areas = areas_mm2[areas_mm2 > self.removal_threshold * 1e6]
                if len(valid_areas) > 0:
                    import matplotlib.pyplot as plt
                    fig_h, ax_h = plt.subplots(figsize=(8, 6))
                    num_bins = max(1, min(25, len(valid_areas)))
                    ax_h.hist(valid_areas, bins=num_bins, alpha=0.7,
                              color='skyblue', edgecolor='black')
                    ax_h.axvline(x=self.removal_threshold*1e6, color='red', linestyle='--',
                                 label='Removal Threshold')
                    a_max = float(getattr(self, 'A_max', 1e-2))
                    if not np.isfinite(a_max) or a_max <= 0:
                        a_max = 1e-2
                    a_max_mm2 = max(a_max * 1e6, float(np.max(valid_areas)))
                    ax_h.set_xlim(0, a_max_mm2)
                    ax_h.set_xticks(np.linspace(0, a_max_mm2, 11))
                    ax_h.set_xlabel('Cross-sectional Area (mm²)', fontsize=14)
                    ax_h.set_ylabel('Number of Members', fontsize=14)
                    #ax_h.set_title('')
                    ax_h.tick_params(axis='both', labelsize=14)
                    ax_h.legend()
                    ax_h.grid(True, alpha=0.3)
                    plt.tight_layout()
                    hist_path = os.path.join(results_dir, f"area_histogram_sdp_{timestamp}.pdf")
                    plt.savefig(hist_path, dpi=300, bbox_inches='tight')
                    print(f"✓ Area histogram saved to: {hist_path}")
                    plt.show()
                try:
                    np.savetxt(
                        os.path.join(results_dir, "final_areas.csv"),
                        np.asarray(self.final_areas, dtype=float),
                        delimiter=',',
                        header='area_m2',
                        comments=''
                    )
                except Exception as exp:
                    print(f"Failed to export SDP final areas: {exp}")
            except Exception as e:
                print(f"Failed to save SDP area histogram: {e}")

            # 打印最终统计
            self._print_final_statistics()
            
        except Exception as e:
            print(f"✗ Visualization failed: {e}")
            import traceback
            traceback.print_exc()
            
            # 至少打印统计信息
            try:
                self._print_final_statistics()
            except:
                print("Could not print statistics either")
    
    def _print_final_statistics(self):
        """打印最终统计信息"""
        if not hasattr(self, 'final_areas'):
            print("No results to display")
            return
            
        print("\n" + "=" * 60)
        print("FINAL OPTIMIZATION STATISTICS")
        print("=" * 60)
        
        effective_elements = np.sum(self.final_areas > self.removal_threshold)
        total_volume = np.sum(self.final_areas * self.element_lengths)
        
        # 检测结构类型
        is_3layer = hasattr(self, 'middle_nodes') and len(self.middle_nodes) > 0
        structure_type = "3-layer" if is_3layer else "2-layer"
        
        print(f"Method: True Semidefinite Programming")
        print(f"Solver: Successfully solved")
        print(f"Verification: {'✓ Passed' if self.verification_passed else '✗ Failed'}")
        print()
        
        print(f"Structure Configuration:")
        print(f"  Type: {structure_type} structure")
        print(f"  Radius: {self.radius}m")
        print(f"  Sectors: {self.n_sectors}")
        print(f"  Depth: {self.depth}m")
        print(f"  Total Candidate Members: {self.n_elements}")
        print()
        
        print(f"Optimization Results:")
        print(f"  Optimal Compliance: {self.final_compliance:.6e}")
        print(f"  Effective Members: {effective_elements}")
        print(f"  Removed Members: {self.n_elements - effective_elements}")
        print(f"  Removal Rate: {(self.n_elements - effective_elements)/self.n_elements:.1%}")
        print()
        
        print(f"Volume Statistics:")
        print(f"  Total Volume: {total_volume*1e6:.1f} cm³")
        print(f"  Volume Constraint: {self.volume_constraint*1e6:.1f} cm³")
        print(f"  Volume Utilization: {total_volume/self.volume_constraint:.1%}")
        print()
        
        active_areas = self.final_areas[self.final_areas > self.removal_threshold]
        if len(active_areas) > 0:
            print(f"Cross-sectional Area Statistics (Active Members):")
            print(f"  Count: {len(active_areas)}")
            print(f"  Average: {np.mean(active_areas)*1e6:.2f} mm²")
            print(f"  Range: [{np.min(active_areas)*1e6:.2f}, {np.max(active_areas)*1e6:.2f}] mm²")
            print(f"  Standard Deviation: {np.std(active_areas)*1e6:.2f} mm²")
        else:
            print("No active members found!")
        
        print("=" * 60)
    
    def _plot_structure(self, ax, areas, title, linewidth_mode='variable'):
        """绘制结构"""
        # 绘制参考圆弧
        from matplotlib.patches import Arc
        outer_arc = Arc((0, 0), 2*self.radius, 2*self.radius, 
                       angle=0, theta1=0, theta2=180, 
                       fill=False, color='black', linestyle='--', alpha=0.3)
        inner_arc = Arc((0, 0), 2*self.inner_radius, 2*self.inner_radius,
                       angle=0, theta1=0, theta2=180,
                       fill=False, color='gray', linestyle='--', alpha=0.3)
        ax.add_patch(outer_arc)
        ax.add_patch(inner_arc)
        
        # 如果有中间层，绘制中间层圆弧
        if hasattr(self, 'middle_radius') and self.middle_radius is not None:
            middle_arc = Arc((0, 0), 2*self.middle_radius, 2*self.middle_radius,
                            angle=0, theta1=0, theta2=180,
                            fill=False, color='gray', linestyle='--', alpha=0.3)
            ax.add_patch(middle_arc)
        
        # 绘制单元
        for i, ((node1, node2), area) in enumerate(zip(self.elements, areas)):
            # 仅用于显示的阈值，更“干净”的可视化
            if area > max(self.removal_threshold, getattr(self, 'display_threshold', self.removal_threshold)):
                x1, y1 = self.nodes[node1]
                x2, y2 = self.nodes[node2]
                
                area_ratio = area / self.A_max
                alpha = 1
                
                # 根据模式设置线宽
                if linewidth_mode == 'uniform':
                    linewidth = 2.5
                    color = 'gray'
                elif linewidth_mode == 'fine':
                    linewidth = 0.8 + 2.0 * area_ratio
                    color = 'darkblue'
                else:  # 'variable' 与 SCP 保持一致
                    linewidth = 1 + 1 * area_ratio
                    color = 'darkblue'
                
                ax.plot([x1, x2], [y1, y2], color=color, 
                       linewidth=linewidth, alpha=alpha)
        
        # 绘制节点
        nodes_array = np.array(self.nodes)
        
        # 外层节点（荷载点）
        outer_coords = nodes_array[self.outer_nodes]
        ax.scatter(outer_coords[:, 0], outer_coords[:, 1], 
                  c='red', s=60, marker='o', edgecolors='black', 
                  label='Load Points', zorder=5)
        
        # 内层节点
        inner_coords = nodes_array[self.inner_nodes]
        ax.scatter(inner_coords[:, 0], inner_coords[:, 1], 
                  c='green', s=40, marker='o', edgecolors='black', 
                  label='Inner Nodes', zorder=5)
        
        # 中间层节点（如果存在）
        if hasattr(self, 'middle_nodes') and len(self.middle_nodes) > 0:
            middle_coords = nodes_array[self.middle_nodes]
            ax.scatter(middle_coords[:, 0], middle_coords[:, 1], 
                      c='green', s=40, marker='o', edgecolors='black', 
                      zorder=5)
        
        # 支撑节点（只有内层两端固定）
        support_nodes = [self.inner_nodes[0], self.inner_nodes[-1]]
        support_coords = nodes_array[support_nodes]
        ax.scatter(support_coords[:, 0], support_coords[:, 1], 
                  c='blue', s=80, marker='s', edgecolors='black', 
                  label='Fixed Supports', zorder=6)
        
        ax.set_xlim(-1.2 * self.radius, 1.2 * self.radius)
        ax.set_ylim(-0.2 * self.radius, 1.2 * self.radius)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title(title, fontweight='bold')
    
    def _plot_loads(self, ax):
        """绘制荷载分布"""
        nodes_array = np.array(self.nodes)
        
        # 绘制荷载（修正：用真实方向）
        max_load = np.max(np.sqrt(self.load_vector[::2]**2 + self.load_vector[1::2]**2))  # 取合力最大值
        
        for i, node_idx in enumerate(self.outer_nodes):
            x, y = nodes_array[node_idx]
            load_x = self.load_vector[2*node_idx]
            load_y = self.load_vector[2*node_idx + 1]
            load_magnitude = np.sqrt(load_x**2 + load_y**2)
            if load_magnitude > 1e-6:
                arrow_scale = load_magnitude / max_load * 0.3
                dx = load_x / load_magnitude * arrow_scale
                dy = load_y / load_magnitude * arrow_scale
                ax.arrow(x, y, dx, dy,  # 用真实方向
                        head_width=0.05, head_length=0.05, 
                        fc='red', ec='red', alpha=0.8)
        # 绘制结构轮廓
        for i, ((node1, node2), area) in enumerate(zip(self.elements, self.final_areas)):
            if area > self.removal_threshold:
                x1, y1 = self.nodes[node1]
                x2, y2 = self.nodes[node2]
                ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.3, linewidth=0.5)
        # 绘制节点
        outer_coords = nodes_array[self.outer_nodes]
        ax.scatter(outer_coords[:, 0], outer_coords[:, 1], 
                  c='red', s=40, marker='o', edgecolors='black', alpha=0.7)
        inner_coords = nodes_array[self.inner_nodes]
        ax.scatter(inner_coords[:, 0], inner_coords[:, 1], 
                  c='green', s=30, marker='o', edgecolors='black', alpha=0.7)
        ax.set_xlim(-1.2 * self.radius, 1.2 * self.radius)
        ax.set_ylim(-0.2 * self.radius, 1.2 * self.radius)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title('Hydrostatic Pressure Loading', fontweight='bold')


def run_sdp_optimization(enable_shell=False, shell_params=None):
    """运行修复的SDP优化，支持壳体分担切换"""
    print("Starting Fixed SDP Truss Topology Optimization...")
    try:
        print("\n" + "=" * 60)
        print("INITIALIZING OPTIMIZER")
        print("=" * 60)
        optimizer = SDPTrussOptimizer(
            radius=5.0,
            n_sectors=16,
            inner_ratio=0.6,
            middle_layer_ratio=0.8,
            depth=50,
            volume_fraction=0.2,
            enable_shell=enable_shell,
            shell_params=shell_params
        )
        print("✓ Optimizer initialized successfully")
        print("\n" + "=" * 60)
        print("STARTING SDP OPTIMIZATION")
        print("=" * 60)
        optimal_areas, optimal_compliance = optimizer.solve_sdp_optimization()
        if optimal_areas is not None and optimal_compliance is not None:
            print("\n" + "=" * 80)
            print("OPTIMIZATION SUCCESSFUL!")
            print("=" * 80)
            print(f" Optimal compliance: {optimal_compliance:.6e}")
            print(f" Effective members: {np.sum(optimal_areas > optimizer.removal_threshold)}/{len(optimal_areas)}")
            print(f" Volume utilization: {np.sum(optimal_areas * optimizer.element_lengths)/optimizer.volume_constraint:.1%}")
            print("\n" + "=" * 60)
            print("GENERATING VISUALIZATION")
            print("=" * 60)
            optimizer.visualize_results()
            # 另存右上角“After Topology Cleanup”单图（PDF）
            try:
                cleaned_areas = optimizer.final_areas.copy()
                cleaned_areas[cleaned_areas < optimizer.removal_threshold] = 0
                fig_single, ax_single = plt.subplots(figsize=(10, 8))
                optimizer._plot_structure(ax_single, cleaned_areas, "", linewidth_mode='variable')
                single_path = os.path.join("results", f"sdp_after_cleanup_{time.strftime('%Y%m%d-%H%M%S')}.pdf")
                plt.tight_layout()
                plt.savefig(single_path, dpi=300, bbox_inches='tight')
                print(f"✓ Saved single cleaned figure to: {single_path}")
                plt.show()
            except Exception as e:
                print(f"Failed to save single cleaned figure: {e}")
            print("\n" + "=" * 80)
            print("ALL TASKS COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            return optimizer
        else:
            print("\n" + "!" * 80)
            print("SDP OPTIMIZATION FAILED!")
            print("!" * 80)
            return None
    except Exception as e:
        print("\n" + "!" * 80)
        print("FATAL ERROR DURING OPTIMIZATION!")
        print("!" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("!" * 80)
        return None

if __name__ == "__main__":
    print("Python Fixed SDP Truss Topology Optimization")
    print("=" * 50)
    # 运行优化
    optimizer = run_sdp_optimization(enable_shell=True)
    if optimizer is not None:
        print("\n🎉 Optimization completed successfully!")
        print(f"📊 Final structure has {np.sum(optimizer.final_areas > optimizer.removal_threshold)} effective members")
    else:
        print("\n❌ Optimization failed. Please check the error messages above.")
