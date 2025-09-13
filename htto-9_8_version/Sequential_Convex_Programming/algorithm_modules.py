"""
算法模块
包含所有优化算法相关的计算组件

包含?- 信赖域管?- 收敛检?- 梯度计算
- 系统矩阵计算
- 步长质量评估
- 约束构建
- 子问题求?
Author: LIANG YUAN
Date: 2025/9/4
"""
import numpy as np
import cvxpy as cp
from dataclasses import dataclass
from typing import List, Tuple, Optional

# ============================================
# 算法参数数据结构
# ============================================

@dataclass
class TrustRegionParams:
    """信赖域参?"""
    initial_radius: float = np.pi/180      # 初始信赖域半?    max_radius: float = np.pi/90         # 最大信赖域半径
    min_radius: float = np.pi/720         # 最小信赖域半径

    shrink_factor: float = 0.5            # 收缩因子
    expand_factor: float = 1.5           # 扩展因子
    accept_threshold: float = 0.01         # 接受步长的阈?    expand_threshold: float = 0.75        # 扩展信赖域的阈?
@dataclass
class GeometryParams:
    """几何约束参数"""
    min_angle_spacing: float = np.pi/180   # 最小角度间?
    boundary_buffer: float = np.pi/180    # 边界缓冲 
    # 相邻角度移动上限的全局帽（避免单步过大角度改变，降低ρ反号风险）
    neighbor_move_cap: float = np.deg2rad(2.0)  # 每步每个角最大改变量的绝对上限（基于绝对帽）
    neighbor_move_eps: float = 1e-3            # 相邻间距一半的减小量，保证严格小于一?
@dataclass
class OptimizationParams:
    """优化算法参数"""
    max_iterations: int = 30            # 最大迭代次?    convergence_tol: float = 1e-4         # 收敛容差
    gradient_fd_step: float = 1e-6        # 有限差分步长

# ============================================
# 信赖域管理器
# ============================================

class TrustRegionManager:
    """信赖域管理器"""
    
    def __init__(self, params: TrustRegionParams):
        self.params = params
        self.current_radius = params.initial_radius
    
    def update_radius(self, step_quality: float) -> bool:
        """更新信赖域半径并返回是否接受步长"""
        old_radius = self.current_radius
        
        if step_quality > self.params.expand_threshold:
            self.current_radius = min(
                self.params.expand_factor * self.current_radius,
                self.params.max_radius
            )
            accept_step = True
            status = "EXPAND"
            
        elif step_quality > self.params.accept_threshold:
            accept_step = True
            status = "ACCEPT"
            
        else:
            self.current_radius = max(
                self.params.shrink_factor * self.current_radius,
                self.params.min_radius
            )
            accept_step = False
            status = "REJECT"
        
        print(f"信赖域更? {old_radius:.4f} ?{self.current_radius:.4f} ({status})")
        return accept_step

# ============================================
# 收敛检查器
# ============================================

class ConvergenceChecker:
    """收敛检查器"""
    
    def __init__(self, tolerance: float, trust_region_manager: TrustRegionManager):
        self.tolerance = tolerance
        self.trust_region_manager = trust_region_manager
    
    def check_convergence(self, theta_old: np.ndarray, theta_new: np.ndarray,
                         A_old: Optional[np.ndarray] = None, 
                         A_new: Optional[np.ndarray] = None) -> bool:
        """检查收敛条?"""
        # 角度变化
        angle_change = np.linalg.norm(theta_new - theta_old) # L2范数（欧几里得范数）
        angle_relative_change = angle_change / max(np.linalg.norm(theta_old), np.pi/180.0)
        
        # 截面积变?        area_change = 0.0
        area_relative_change = 0.0
        if A_old is not None and A_new is not None:
            area_change = np.linalg.norm(A_new - A_old)
            area_relative_change = area_change / max(np.linalg.norm(A_old), 1.0)
        
        # 信赖域半径检?        radius_small = (self.trust_region_manager.current_radius < 
                       2 * self.trust_region_manager.params.min_radius)
        
        # 收敛判断
        converged = (
            (angle_relative_change < self.tolerance and
             area_relative_change < self.tolerance)
             )
        
        if converged:
            print(f"算法收敛:")
            print(f"  角度变化: {angle_change:.6e} (相对: {angle_relative_change:.6e})")
            if A_old is not None:
                print(f"  面积变化: {area_change:.6e} (相对: {area_relative_change:.6e})")
            print(f"  信赖域半? {self.trust_region_manager.current_radius:.6e}")
        
        return converged
    

# ============================================
# 梯度计算?# ============================================

class GradientCalculator:
    """梯度计算?"""
    
    def __init__(self, optimizer_ref):
        self.opt = optimizer_ref
        self.fd_step = OptimizationParams().gradient_fd_step # 有限差分步长
    
    def compute_gradients(self, theta: np.ndarray) -> Tuple[List[List[np.ndarray]], List[np.ndarray]]:
        """计算梯度"""
        # print(f"计算梯度，当前角? {theta * 180/np.pi}")
        
        n_angles = len(theta)
        grad_K_list = []  # grad_K_list[j] 包含所有单元对第j个角度的梯度
        grad_f = []       # grad_f[j] 是载荷向量对第j个角度的梯度
        
        for j in range(n_angles):
            # print(f"  计算对角?{j+1}/{n_angles} 的梯?..")
            grad_K_j, grad_f_j = self._compute_gradient_wrt_angle(theta, j)
            grad_K_list.append(grad_K_j)
            grad_f.append(grad_f_j)
        
        print("梯度计算完成")
        return grad_K_list, grad_f
    
    def _compute_gradient_wrt_angle(self, theta: np.ndarray, angle_idx: int) -> Tuple[List[np.ndarray], np.ndarray]:
        """计算对单个角度的梯度"""
        h = self.fd_step
        
        # 扰动角度
        theta_plus = theta.copy()
        theta_minus = theta.copy()
        theta_plus[angle_idx] += h
        theta_minus[angle_idx] -= h
        
        # 确保扰动后仍满足约束
        theta_plus_orig = theta_plus.copy()
        theta_minus_orig = theta_minus.copy()
        
        theta_plus = self._project_to_feasible(theta_plus)
        theta_minus = self._project_to_feasible(theta_minus)
        
        # 检查投影是否改变了角度
        if not np.allclose(theta_plus, theta_plus_orig):
            print(f"      ⚠️  角度 {angle_idx} 的扰动后投影改变了?)
        
        if not np.allclose(theta_minus, theta_minus_orig):
            print(f"      ⚠️  角度 {angle_idx} 的扰动后投影改变了?)
        
        try:
            # 计算扰动后的系统矩阵
            K_plus_list, f_plus = self._compute_system_matrices(theta_plus)
            K_minus_list, f_minus = self._compute_system_matrices(theta_minus)
            
            # 有限差分梯度
            grad_K_elements = []
            for i in range(self.opt.n_elements):
                grad_K_i = (K_plus_list[i] - K_minus_list[i]) / (2 * h)
                grad_K_elements.append(grad_K_i)
            
            grad_f = (f_plus - f_minus) / (2 * h)
            
            return grad_K_elements, grad_f
            
        except Exception as e:
            print(f"      ?角度 {angle_idx} 梯度计算失败: {e}")
            raise
    
    def _compute_system_matrices(self, theta: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """计算系统矩阵"""
        return self.opt.stiffness_calc.compute_system_matrices(
            self.opt.geometry, theta, self.opt.radius, 
            self.opt.geometry_calc, self.opt.load_calc, self.opt.depth
        )
    
    def _project_to_feasible(self, theta: np.ndarray) -> np.ndarray:
        """投影到可行域"""
        geometry_params = {
            'boundary_buffer': GeometryParams().boundary_buffer,
            'min_angle_spacing': GeometryParams().min_angle_spacing
        }
        
        theta_proj = theta.copy()
        
        # 边界投影
        theta_proj[0] = max(theta_proj[0], geometry_params['boundary_buffer'])
        theta_proj[-1] = min(theta_proj[-1], np.pi - geometry_params['boundary_buffer'])
        
        # 单调性投?        for i in range(len(theta_proj) - 1):
            if theta_proj[i+1] <= theta_proj[i] + geometry_params['min_angle_spacing']:
                theta_proj[i+1] = theta_proj[i] + geometry_params['min_angle_spacing']
        
        return theta_proj

# ============================================
# 系统计算?# ============================================

class SystemCalculator:
    """系统计算?""
    
    def __init__(self, optimizer_ref):
        self.opt = optimizer_ref
    
    def compute_system_matrices(self, theta: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """计算系统矩阵"""
        return self.opt.stiffness_calc.compute_system_matrices(
            self.opt.geometry, theta, self.opt.radius, 
            self.opt.geometry_calc, self.opt.load_calc, self.opt.depth
        )
    
    def compute_actual_compliance(self, theta, A: np.ndarray) -> float:
        """计算实际柔度（统一一?theta?""
        try:
            # 更新节点坐标（统一一?theta?            updated_coords = self.opt.geometry_calc.update_node_coordinates(
                self.opt.geometry, theta, self.opt.radius
            )

            # 计算新的单元几何属?            element_lengths, element_directions = self.opt.geometry_calc.compute_element_geometry(
                updated_coords, self.opt.geometry.elements
            )
            
            # 组装全局刚度矩阵（回退：不做装配期硬裁?分析侧抬底）
            K_global = self.opt.stiffness_calc.assemble_global_stiffness(
                self.opt.geometry, A, element_lengths, element_directions
            )
            
            # 计算新的载荷向量
            f_new = self.opt.load_calc.compute_load_vector(
                updated_coords, self.opt.geometry.load_nodes, self.opt.depth
            )

            
            # 应用边界条件
            K_red = K_global[np.ix_(self.opt.free_dofs, self.opt.free_dofs)]
            f_red = f_new[self.opt.free_dofs]
            
            # 求解位移并计算柔度（稳健求解，带正则化与伪逆回退?            def _stable_solve(K: np.ndarray, f: np.ndarray, kappa_max: float = 1e10):
                try:
                    # 首先尝试直接求解
                    return np.linalg.solve(K, f)
                except np.linalg.LinAlgError:
                    pass
                # 条件数检查与对角正则?                lam_base = float(np.maximum(1e-16, np.mean(np.diag(K)) if K.size > 0 else 1.0))
                lam = 1e-8 * lam_base
                for _ in range(5):
                    try:
                        K_reg = K + lam * np.eye(K.shape[0])
                        U = np.linalg.solve(K_reg, f)
                        cond_val = float(np.linalg.cond(K_reg))
                        if np.isfinite(cond_val) and cond_val <= kappa_max:
                            return U
                    except np.linalg.LinAlgError:
                        pass
                    lam *= 10.0
                # 伪逆回退（最后手段）
                try:
                    return np.linalg.pinv(K, rcond=1e-10) @ f
                except Exception:
                    return None

            U_red = _stable_solve(K_red, f_red)
            if U_red is None:
                print("?刚度矩阵奇异/病态，返回极大值作为惩?)
                return 1e12

            compliance = float(np.dot(f_red, U_red))
            if (not np.isfinite(compliance)) or (compliance <= 0.0):
                print("?柔度异常（非?非有限），返回极大值作为惩?)
                return 1e12
            return compliance
        
        except Exception as e:
            print(f"?柔度计算失败: {e}")
            return 1e12
    

# ============================================
# 步长质量评估?# ============================================

class StepQualityEvaluator:
    """步长质量评估?""
    
    def __init__(self, optimizer_ref):
        self.opt = optimizer_ref
    
    def evaluate_step_quality(self, theta_old, theta_new, 
                             A_new: np.ndarray, predicted_from_model: Optional[float] = None) -> float:
        """评估步长质量 - 支持单层和分层模?""
        # 计算实际目标函数?        system_calc = SystemCalculator(self.opt)
        actual_compliance = system_calc.compute_actual_compliance(theta_new, A_new)
        # 强校验：实际柔度必须为正且有限，否则直接当作评估失败
        if (not np.isfinite(actual_compliance)) or (actual_compliance <= 0.0) or (actual_compliance > 1e11):
            raise ValueError(f"实际柔度异常: {actual_compliance}")
        
        # 计算线性模型预测?        if predicted_from_model is not None and np.isfinite(predicted_from_model):
            predicted_compliance = float(predicted_from_model)
            # 严格模式：子问题返回的预测柔度必须为?            if predicted_compliance <= 0:
                raise ValueError(f"子问题返回的预测柔度非正: {predicted_compliance:.6e}")
            setattr(self.opt, '_pred_source', 'subproblem_t')
        else:
            predicted_compliance = self._compute_predicted_compliance(theta_old, theta_new, A_new)
        
        # 计算reduction ratio
        actual_reduction = self.opt.current_compliance - actual_compliance
        predicted_reduction = self.opt.current_compliance - predicted_compliance
        
        # 记录详细信息用于调试
        self._record_step_details(theta_old, theta_new, A_new, 
                                actual_compliance, predicted_compliance,
                                actual_reduction, predicted_reduction)
        
        # 使用相对阈值，严格判定预测下降是否过小
        rel_tol = max(1e-10, 1e-6 * abs(self.opt.current_compliance))
        if abs(predicted_reduction) < rel_tol:
            raise ValueError(f"预测下降过小：|predicted_reduction|={abs(predicted_reduction):.3e} < {rel_tol:.3e}")
        
        rho = actual_reduction / predicted_reduction
        
        print(f"步长质量评估:")
        print(f"  当前柔度: {self.opt.current_compliance:.6e}")
        print(f"  实际柔度: {actual_compliance:.6e}")
        print(f"  预测柔度: {predicted_compliance:.6e} [src={getattr(self.opt, '_pred_source', 'na')}]")
        if hasattr(self.opt, '_last_alpha'):
            try:
                print(f"  线搜索? {float(self.opt._last_alpha):.3f}")
            except Exception:
                pass
        print(f"  实际下降: {actual_reduction:.6e}")
        print(f"  预测下降: {predicted_reduction:.6e}")
        print(f"  质量比率: {rho:.4f}")
        if not np.isfinite(rho):
            print("⚠️  警告: rho 非数或无穷，视为无效步长")
        
        # 检查异常?        if abs(rho) > 100:
            print(f"⚠️  警告: rho值异?({rho:.2f})，可能存在问?)
        elif rho < -10:
            print(f"⚠️  警告: rho值过?({rho:.2f})，步长质量很?)
        
        # ρ 异常时，打印简要归因提示（详细数据写入 step_details?        if rho < 0 or abs(rho) > 5:
            print("  ℹ️  rho异常：已记录半步柔度、活跃集变化、投影、条件数与Top贡献，详见step_details")
        return rho
    
    def _record_step_details(self, theta_old, theta_new, A_new, 
                           actual_compliance, predicted_compliance,
                           actual_reduction, predicted_reduction):
        """记录步长详细信息"""
        if not hasattr(self.opt, 'step_details'):
            self.opt.step_details = []
        
        # 基本记录
        step_detail = {
            'iteration': self.opt.iteration_count + 1,
            'theta_old': theta_old.copy() if hasattr(theta_old, 'copy') else theta_old,
            'theta_new': theta_new.copy() if hasattr(theta_new, 'copy') else theta_new,
            'A_new': A_new.copy(),
            'current_compliance': self.opt.current_compliance,
            'actual_compliance': actual_compliance,
            'predicted_compliance': predicted_compliance,
            'actual_reduction': actual_reduction,
            'predicted_reduction': predicted_reduction,
            'rho': actual_reduction / predicted_reduction if abs(predicted_reduction) > 1e-12 else 0.0,
            'trust_radius': self.opt.trust_radius,
            'pred_source': getattr(self.opt, '_pred_source', 'na')
        }
        
        # 仅在ρ异常时做更详细的记录，避免终端刷?        try:
            rho_val = step_detail['rho']
            if rho_val < 0 or abs(rho_val) > 5:
                # 1) 半步柔度（固定θ，更新A?                half_step_compliance = SystemCalculator(self.opt).compute_actual_compliance(theta_old, A_new)
                step_detail['half_step_compliance'] = half_step_compliance
                step_detail['deltaC_A'] = float(half_step_compliance - self.opt.current_compliance)
                step_detail['deltaC_theta'] = float(actual_compliance - half_step_compliance)
                
                # 2) 活跃集变化（跨移除阈值）
                if hasattr(self.opt, 'current_areas') and self.opt.current_areas is not None:
                    thr = getattr(self.opt, 'removal_threshold', 0.0)
                    prev_active = (self.opt.current_areas > thr)
                    new_active = (A_new > thr)
                    removed = list(np.where(prev_active & (~new_active))[0])
                    added = list(np.where((~prev_active) & new_active)[0])
                    step_detail['active_set_removed'] = removed
                    step_detail['active_set_added'] = added
                
                # 3) 角度投影标记（简单比较投影与输入差异?                proj = getattr(self.opt.subproblem_solver, 'gradient_calc', None)
                if proj is not None and hasattr(proj, '_project_to_feasible'):
                    theta_old_proj = proj._project_to_feasible(theta_old)
                    theta_new_proj = proj._project_to_feasible(theta_new)
                    step_detail['angle_projection_flags'] = {
                        'old_proj_changed': bool(not np.allclose(theta_old, theta_old_proj)),
                        'new_proj_changed': bool(not np.allclose(theta_new, theta_new_proj))
                    }
                
                # 4) 条件数（半步与全步）
                try:
                    # 半步 (A^{k+1}, θ^k)
                    updated_coords = self.opt.geometry_calc.update_node_coordinates(self.opt.geometry, theta_old, self.opt.radius)
                    elen, edir = self.opt.geometry_calc.compute_element_geometry(updated_coords, self.opt.geometry.elements)
                    K_global_half = self.opt.stiffness_calc.assemble_global_stiffness(self.opt.geometry, A_new, elen, edir)
                    f_half = self.opt.load_calc.compute_load_vector(updated_coords, self.opt.geometry.load_nodes, self.opt.depth)
                    free = self.opt.free_dofs
                    K_red_half = K_global_half[np.ix_(free, free)]
                    step_detail['cond_K_half'] = float(np.linalg.cond(K_red_half))
                except Exception:
                    step_detail['cond_K_half'] = None
                try:
                    # 全步 (A^{k+1}, θ^{k+1})
                    updated_coords2 = self.opt.geometry_calc.update_node_coordinates(self.opt.geometry, theta_new, self.opt.radius)
                    elen2, edir2 = self.opt.geometry_calc.compute_element_geometry(updated_coords2, self.opt.geometry.elements)
                    K_global_full = self.opt.stiffness_calc.assemble_global_stiffness(self.opt.geometry, A_new, elen2, edir2)
                    f_full = self.opt.load_calc.compute_load_vector(updated_coords2, self.opt.geometry.load_nodes, self.opt.depth)
                    free = self.opt.free_dofs
                    K_red_full = K_global_full[np.ix_(free, free)]
                    step_detail['cond_K_full'] = float(np.linalg.cond(K_red_full))
                except Exception:
                    step_detail['cond_K_full'] = None
                
                # 5) Top-N 元素/角度贡献（近似，使用半步位移?                try:
                    u_half = np.linalg.solve(K_red_half, f_half[free])
                    # 元素?ΔC_A 的线性近似贡献：-(u^T dK/dA_i u) * ΔA_i
                    N = min(5, len(A_new))
                    elem_contribs = []
                    E = self.opt.stiffness_calc.material_data.E_steel
                    for i, (n1, n2) in enumerate(self.opt.geometry.elements):
                        L_i = elen[i]
                        c, s = edir[i]
                        k_unit = (E / L_i) * np.array([[c*c, c*s, -c*c, -c*s],
                                                       [c*s, s*s, -c*s, -s*s],
                                                       [-c*c, -c*s, c*c, c*s],
                                                       [-c*s, -s*s, c*s, s*s]])
                        dofs = [2*n1, 2*n1+1, 2*n2, 2*n2+1]
                        dK_g = np.zeros_like(K_global_half)
                        for m in range(4):
                            for n in range(4):
                                dK_g[dofs[m], dofs[n]] += k_unit[m, n]
                        dK_r = dK_g[np.ix_(free, free)]
                        dC_dAi = - float(u_half.T @ (dK_r @ u_half))
                        elem_contribs.append((i, dC_dAi * float(A_new[i] - self.opt.current_areas[i]) if hasattr(self.opt, 'current_areas') else 0.0))
                    elem_contribs.sort(key=lambda x: abs(x[1]), reverse=True)
                    step_detail['top_element_contribs'] = elem_contribs[:N]
                except Exception:
                    step_detail['top_element_contribs'] = None
                
                # 6) 角度Top-N贡献（使用缓存梯度，如可用）
                try:
                    N = min(5, len(theta_old))
                    if hasattr(self.opt, '_cached_gradients') and self.opt._cached_gradients is not None:
                        cached_grad_K_list, cached_grad_f = self.opt._cached_gradients
                        # 构?dC/dθ_j 解析式在半步点的近似
                        grad_theta = []
                        dK_global_accum = None
                        for j in range(len(theta_old)):
                            dK_g_j = None
                            for dK_elem in cached_grad_K_list[j]:
                                dK_g_j = dK_elem if dK_g_j is None else (dK_g_j + dK_elem)
                            dK_r_j = np.zeros_like(K_red_half) if dK_g_j is None else dK_g_j[np.ix_(free, free)]
                            df_r_j = cached_grad_f[j][free]
                            dC_dtheta_j = - float(u_half.T @ (dK_r_j @ u_half)) + 2.0 * float(u_half.T @ df_r_j)
                            grad_theta.append(dC_dtheta_j)
                        grad_theta = np.array(grad_theta)
                        contrib_theta = grad_theta * (theta_new - theta_old)
                        idx_sorted = np.argsort(-np.abs(contrib_theta))
                        top_list = [(int(i), float(contrib_theta[i])) for i in idx_sorted[:N]]
                        step_detail['top_angle_contribs'] = top_list
                    else:
                        step_detail['top_angle_contribs'] = None
                except Exception:
                    step_detail['top_angle_contribs'] = None
        except Exception:
            pass
        
        self.opt.step_details.append(step_detail)
    
    def _compute_predicted_compliance(self, theta_old, theta_new, 
                                       A_new: np.ndarray) -> float:
        """计算预测柔度
        优先使用与子问题一致的线性模型（?k 点线性化）来预测?        t_pred = f_red(A, θ)^T · K_red(A, θ)^{-1} · f_red(A, θ)
        若无缓存线性模型，则回退到一阶泰勒展开?        """
        try:
            # 优先路径：使用子问题缓存的线性模型（线性化点在 k?            if hasattr(self.opt, '_cached_linear_model') and self.opt._cached_linear_model is not None:
                cache = self.opt._cached_linear_model
                try:
                    # 0) 可行域投影，保持与子问题一?                    theta_project = getattr(self.opt.subproblem_solver, 'gradient_calc', None)
                    if theta_project is not None and hasattr(theta_project, '_project_to_feasible'):
                        theta_new_proj = theta_project._project_to_feasible(theta_new)
                    else:
                        theta_new_proj = theta_new
                    theta_k = cache['theta_k']
                    A_k = cache['A_k']
                    # 1) 用缓存的线性模型重?K_approx(A_new, θ_new) ?f_linearized(θ_new)
                    K_constant = np.zeros((self.opt.n_dof, self.opt.n_dof))
                    for i in range(self.opt.n_elements):
                        K_constant += A_k[i] * cache['K_current_list'][i]
                    # A 线性项
                    K_A_delta = np.zeros_like(K_constant)
                    for i in range(self.opt.n_elements):
                        K_A_delta += (A_new[i] - A_k[i]) * cache['K_current_list'][i]
                    # θ 线性项（使用缓存的加权梯度矩阵?                    K_theta_delta = np.zeros_like(K_constant)
                    for j in range(len(theta_k)):
                        K_theta_delta += (theta_new_proj[j] - theta_k[j]) * cache['weighted_grad_matrices'][j]
                    K_approx_num = K_constant + K_A_delta + K_theta_delta
                    # 载荷向量线性化
                    f_lin = cache['f_current'].copy()
                    for j in range(len(theta_k)):
                        f_lin += (theta_new_proj[j] - theta_k[j]) * cache['grad_f'][j]
                    # 2) 应用自由度裁剪并计算 t_pred
                    S = cache['selection_matrix']
                    K_red = S @ K_approx_num @ S.T
                    f_red = S @ f_lin
                    # 数值稳健性：尝试求解，奇异时回退到泰勒模?                    try:
                        u_red = np.linalg.solve(K_red, f_red)
                        t_pred = float(f_red.T @ u_red)
                        if t_pred <= 0:
                            raise ValueError("predicted compliance non-positive")
                        # 标注预测来源
                        setattr(self.opt, '_pred_source', 'linear_model')
                        return t_pred
                    except Exception:
                        raise
                except Exception:
                    raise

            # 0. һµĿͶӰԤ-ʵʲһ\n            theta_project = getattr(self.opt.subproblem_solver, 'gradient_calc', None)
            if theta_project is not None and hasattr(theta_project, '_project_to_feasible'):
                theta_old_proj = theta_project._project_to_feasible(theta_old)
                theta_new_proj = theta_project._project_to_feasible(theta_new)
                # 检查投影是否改变了theta - 支持分层模式
                projection_changed = False
                
                    for layer in theta_new:
                        if not np.allclose(theta_new_proj[layer], theta_new[layer]):
                            projection_changed = True
                            break
                else:
                    projection_changed = not np.allclose(theta_new_proj, theta_new)
                
                if projection_changed:
                    print("  ℹ️  预测使用投影后的Δθ (与子问题一?")
            else:
                theta_old_proj = theta_old
                theta_new_proj = theta_new
            
            # 1. 计算设计变量变化 - 支持分层模式
            
                # 分层模式
                theta_change = {}
                for layer in theta_new_proj:
                    theta_change[layer] = theta_new_proj[layer] - theta_old_proj[layer]
            else:
                # 单层模式
                theta_change = theta_new_proj - theta_old_proj
            A_change = A_new - self.opt.current_areas if hasattr(self.opt, 'current_areas') else np.zeros_like(A_new)
            
            # 2. 获取梯度信息
            if hasattr(self.opt, '_cached_gradients') and self.opt._cached_gradients is not None:
                cached_grad_K_list, cached_grad_f = self.opt._cached_gradients
                
                # 从缓存的梯度中提取柔度梯?                grad_theta, grad_A = self._extract_compliance_gradients_from_cached(cached_grad_K_list, cached_grad_f, theta_old, A_new)
                
            else:
                # 如果没有缓存梯度，计算有限差分梯?                print("  ⚠️  分层模式下的预测柔度：暂跳过梯度计算，使用当前柔度作为预?)
                # 临时回退：使用当前柔度作为预?                setattr(self.opt, '_pred_source', 'fallback')
                return self.opt.current_compliance * 0.999  # 稍微减小作为预测
            
            # 3. 验证梯度类型和形?            if grad_theta is None or grad_A is None:
                raise ValueError("梯度计算失败：返回值为 None")
            
            if not isinstance(grad_theta, np.ndarray) or not isinstance(grad_A, np.ndarray):
                raise ValueError(f"梯度类型错误：grad_theta={type(grad_theta)}, grad_A={type(grad_A)}")
            
            # 验证梯度形状 - 支持分层模式
            
                # 分层模式：跳过形状检查，稍后实现
                pass
            else:
                # 单层模式：原有检?                if grad_theta.shape != theta_change.shape:
                    raise ValueError(f"角度梯度形状不匹配：grad_theta.shape={grad_theta.shape}, theta_change.shape={theta_change.shape}")
            
            if grad_A.shape != A_change.shape:
                raise ValueError(f"面积梯度形状不匹配：grad_A.shape={grad_A.shape}, A_change.shape={A_change.shape}")
            
            # 4. 一阶泰勒展开：m^(k)(A^(k+1), θ^(k+1)) = m^(k)(A^(k), θ^(k)) + ∇m^(k) · [ΔA, Δθ]
            
                # 分层模式：暂时使用简化预?                print("  ⚠️  分层模式下的一阶泰勒展开：暂未实现，使用简化预?)
                predicted_compliance = self.opt.current_compliance * 0.999
            else:
                # 单层模式：原有计?                predicted_compliance = (self.opt.current_compliance + 
                                       np.dot(grad_theta, theta_change) + 
                                       np.dot(grad_A, A_change))
            
            # 5. 合理性检?- 如果预测柔度非正，直接报?            if predicted_compliance <= 0:
                raise ValueError(f"预测柔度非正: {predicted_compliance:.6e}，算法终?)
            
            # 标注预测来源
            setattr(self.opt, '_pred_source', 'taylor')
            return predicted_compliance
            
        except Exception as e:
            print(f"      ?预测柔度计算失败: {e}")
            raise
    
    def _extract_compliance_gradients_from_cached(self, cached_grad_K_list, cached_grad_f, theta_old, A_new):
        """从缓存的梯度中提取柔度梯?""
        # 参数验证
        if theta_old is None or A_new is None:
            raise ValueError("输入参数?None")
        
        # ֤ͳһһά theta
        if np.any(np.isnan(theta_old)):
            raise ValueError(" NaN ")
        if np.any(np.isnan(A_new)):
            raise ValueError(" NaN ")
        
        try:
            # 1) ڵǰ(^k, A^{k+1}) װϵͳλƣͳһһά theta
            updated_coords = self.opt.geometry_calc.update_node_coordinates(
                self.opt.geometry, theta_old, self.opt.radius
            )
            element_lengths, element_directions = self.opt.geometry_calc.compute_element_geometry(
                updated_coords, self.opt.geometry.elements
            )
            K_global = self.opt.stiffness_calc.assemble_global_stiffness(
                self.opt.geometry, A_new, element_lengths, element_directions
            )
            f_global = self.opt.load_calc.compute_load_vector(
                updated_coords, self.opt.geometry.outer_nodes, self.opt.depth
            )
            free = self.opt.free_dofs
            K_red = K_global[np.ix_(free, free)]
            f_red = f_global[free]
            u_red = np.linalg.solve(K_red, f_red)
            
            # 2) 用解析式 dC = -u^T (dK) u + 2 u^T (df) 计算对角度的梯度
            n_angles = len(theta_old)
            grad_theta = np.zeros(n_angles)
            for j in range(n_angles):
                # cached_grad_K_list[j] 是一个列表：每个元素对应一个单元在全局坐标下的 dK 贡献
                dK_global_j = None
                for dK_elem in cached_grad_K_list[j]:
                    dK_global_j = dK_elem if dK_global_j is None else (dK_global_j + dK_elem)
                if dK_global_j is None:
                    dK_red_j = np.zeros_like(K_red)
                else:
                    dK_red_j = dK_global_j[np.ix_(free, free)]
                df_red_j = cached_grad_f[j][free]
                grad_theta[j] = - float(u_red.T @ (dK_red_j @ u_red)) + 2.0 * float(u_red.T @ df_red_j)
            
            # 3) 面积梯度使用解析式：dC/dA_i = -u^T (dK/dA_i) u，假?df/dA = 0
            n_elements = len(A_new)
            grad_A = np.zeros(n_elements)
            E = self.opt.material_data.E_steel if hasattr(self.opt, 'material_data') else self.opt.stiffness_calc.material_data.E_steel
            for i, (node1, node2) in enumerate(self.opt.geometry.elements):
                L_i = element_lengths[i]
                c, s = element_directions[i]
                # 4x4 单元刚度对面积的导数（局部→全局一体化模板?                k_unit = (E / L_i) * np.array([
                    [c*c, c*s, -c*c, -c*s],
                    [c*s, s*s, -c*s, -s*s],
                    [-c*c, -c*s, c*c, c*s],
                    [-c*s, -s*s, c*s, s*s]
                ])
                # 装配到全局并裁剪到自由?                dofs = [2*node1, 2*node1+1, 2*node2, 2*node2+1]
                dK_global = np.zeros_like(K_global)
                for m in range(4):
                    for n in range(4):
                        dK_global[dofs[m], dofs[n]] += k_unit[m, n]
                dK_red_i = dK_global[np.ix_(free, free)]
                grad_A[i] = - float(u_red.T @ (dK_red_i @ u_red))
            
            return grad_theta, grad_A
        except Exception as e:
            print(f"    ?从缓存梯度提取柔度梯度失? {e}")
            # 回退到有限差分方?            return self._compute_finite_difference_gradients(theta_old, A_new)
    
    def _compute_finite_difference_gradients(self, theta: np.ndarray, A: np.ndarray):
        """计算有限差分梯度"""
        epsilon = 1e-6  # 有限差分步长
        
        # 角度梯度
        grad_theta = np.zeros_like(theta)
        for i in range(len(theta)):
            theta_plus = theta.copy()
            theta_plus[i] += epsilon
            theta_minus = theta.copy()
            theta_minus[i] -= epsilon
            
            # 计算前向和后向差?            try:
                compliance_plus = self.opt.system_calculator.compute_actual_compliance(theta_plus, A)
                compliance_minus = self.opt.system_calculator.compute_actual_compliance(theta_minus, A)
                
                # 验证返回?                if compliance_plus is None or compliance_minus is None:
                    raise ValueError(f"柔度计算失败：角?{i} 的有限差分计算返?None")
                
                grad_theta[i] = (compliance_plus - compliance_minus) / (2 * epsilon)
                
            except Exception as e:
                print(f"        ?角度 {i} 梯度计算失败: {e}")
                raise ValueError(f"角度 {i} 梯度计算失败: {e}")
        
        # 面积梯度
        grad_A = np.zeros_like(A)
        for i in range(len(A)):
            A_plus = A.copy()
            A_plus[i] += epsilon
            A_minus = A.copy()
            A_minus[i] -= epsilon
            
            # 计算前向和后向差?            try:
                compliance_plus = self.opt.system_calculator.compute_actual_compliance(theta, A_plus)
                compliance_minus = self.opt.system_calculator.compute_actual_compliance(theta, A_minus)
                
                # 验证返回?                if compliance_plus is None or compliance_minus is None:
                    raise ValueError(f"柔度计算失败：面?{i} 的有限差分计算返?None")
                
                grad_A[i] = (compliance_plus - compliance_minus) / (2 * epsilon)
                
            except Exception as e:
                print(f"        ?面积 {i} 梯度计算失败: {e}")
                raise ValueError(f"面积 {i} 梯度计算失败: {e}")
        
        # 验证最终结?        if grad_theta is None or grad_A is None:
            raise ValueError("有限差分梯度计算失败：返回值为 None")
        
        if not isinstance(grad_theta, np.ndarray) or not isinstance(grad_A, np.ndarray):
            raise ValueError(f"有限差分梯度类型错误：grad_theta={type(grad_theta)}, grad_A={type(grad_A)}")
        
        return grad_theta, grad_A

# ============================================
# 约束构建?# ============================================

class ConstraintBuilder:
    """约束构建?""
    
    def __init__(self, optimizer_ref):
        self.opt = optimizer_ref
        self.geometry_params = GeometryParams()
    
    def build_geometry_constraints(self, theta: cp.Variable) -> List[cp.Constraint]:
        """构建几何约束"""
        constraints = []
        
        n_angles = theta.shape[0] if hasattr(theta, 'shape') else len(theta)
        
        
        # 边界约束
        constraints.append(theta[0] >= self.geometry_params.boundary_buffer)
        constraints.append(theta[-1] <= np.pi - self.geometry_params.boundary_buffer)
        
        
        return constraints
    
    def build_trust_region_constraint(self, theta: cp.Variable, theta_k: np.ndarray, 
                                     trust_radius: float) -> cp.Constraint:
        """构建信赖域约?""
        return cp.norm(theta - theta_k, 2) <= trust_radius

    def build_neighbor_move_limits(self, theta: cp.Variable, theta_k: np.ndarray) -> List[cp.Constraint]:
        constraints: List[cp.Constraint] = []
        n = len(theta_k)
        gp = self.geometry_params
        caps = getattr(self.opt, 'theta_move_caps', None)
        if isinstance(caps, np.ndarray) and caps.shape[0] == n:
            m = np.maximum(0.0, caps.astype(float))
        else:
            m = np.full(n, max(0.0, gp.neighbor_move_cap - gp.neighbor_move_eps), dtype=float)
        constraints.append(theta - theta_k <= m)
        constraints.append(theta_k - theta <= m)
        return constraints
    
    
    

# ============================================
# 子问题求解器
# ============================================

class SubproblemSolver:
    """子问题求解器"""
    
    def __init__(self, optimizer_ref):
        self.opt = optimizer_ref
        self.gradient_calc = GradientCalculator(optimizer_ref)
        self.constraint_builder = ConstraintBuilder(optimizer_ref)
    
    
    
    
    
    
    
    
    def solve_linearized_subproblem(self, A_k: np.ndarray, theta_k: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        """求解线性化子问?""
        print(f"求解联合线性化优化，当前信赖域半径: {self.opt.trust_radius:.4f}")
        
        try:
            # 1. 计算当前点的系统矩阵和梯?            print("  步骤1: 计算系统矩阵和梯?..")
            K_current_list, f_current = self.gradient_calc._compute_system_matrices(theta_k)
            grad_K_list, grad_f = self.gradient_calc.compute_gradients(theta_k)
            
            # 2. 定义优化变量
            A = cp.Variable(self.opt.n_elements, pos=True)
            theta = cp.Variable(len(theta_k))
            t = cp.Variable()
            
            # 3. 构建分离的线性化刚度矩阵
            print("  步骤2: 构建分离线性化刚度矩阵...")
            
            # 当前点的刚度矩阵
            K_constant = np.zeros((self.opt.n_dof, self.opt.n_dof))
            for i in range(self.opt.n_elements):
                K_constant += A_k[i] * K_current_list[i]
            
            # A的线性项
            K_A_terms = []
            for i in range(self.opt.n_elements):
                K_A_terms.append((A[i] - A_k[i]) * cp.Constant(K_current_list[i]))
            
            # θ的线性项
            K_theta_terms = []
            for j in range(len(theta_k)):
                weighted_grad_matrix = np.zeros((self.opt.n_dof, self.opt.n_dof))
                for i in range(self.opt.n_elements):
                    weighted_grad_matrix += A_k[i] * grad_K_list[j][i]
                K_theta_terms.append((theta[j] - theta_k[j]) * cp.Constant(weighted_grad_matrix))
            
            # 联合线性化刚度矩阵
            K_approx = cp.Constant(K_constant) + cp.sum(K_A_terms) + cp.sum(K_theta_terms)
            
            # 4. 应用边界条件
            print("  步骤3: 应用边界条件...")
            
            n_free = len(self.opt.free_dofs)
            selection_matrix = np.zeros((n_free, self.opt.n_dof))
            for i, dof in enumerate(self.opt.free_dofs):
                selection_matrix[i, dof] = 1.0
            
            S = cp.Constant(selection_matrix)
            K_reduced = S @ K_approx @ S.T
            
            # 线性化载荷向量
            f_linearized = cp.Constant(f_current)
            for j in range(len(theta_k)):
                f_linearized += (theta[j] - theta_k[j]) * cp.Constant(grad_f[j])
            
            f_reduced = S @ f_linearized
            
            # 5. 构建SDP约束
            print("  步骤4: 构建SDP约束...")
            
            schur_matrix = cp.bmat([
                [K_reduced, f_reduced.reshape((-1, 1))],
                [f_reduced.reshape((1, -1)), t.reshape((1, 1))]
            ])
            
            # 所有约?            constraints = [
                schur_matrix >> 0,  # SDP约束
                *self.constraint_builder.build_geometry_constraints(theta),  # 几何约束
                self.constraint_builder.build_trust_region_constraint(theta, theta_k, self.opt.trust_radius),  # 信赖?                *self.constraint_builder.build_neighbor_move_limits(theta, theta_k),  # 相邻角度移动上限
                A >= self.opt.A_min,
                A <= self.opt.A_max
            ]

            # 覆盖旧约束：使用边界 + L2 信赖?+ 逐点步长帽（避免单调?分层依赖?            gp_local = self.constraint_builder.geometry_params
            _n_theta = len(theta_k)
            _caps = getattr(self.opt, 'theta_move_caps', None)
            if isinstance(_caps, np.ndarray) and _caps.shape[0] == _n_theta:
                m_vec = np.maximum(0.0, _caps.astype(float))
            else:
                m_vec = np.full(_n_theta, max(0.0, gp_local.neighbor_move_cap - gp_local.neighbor_move_eps), dtype=float)
            constraints = [
                schur_matrix >> 0,
                theta[0] >= gp_local.boundary_buffer,
                theta[-1] <= np.pi - gp_local.boundary_buffer,
                self.constraint_builder.build_trust_region_constraint(theta, theta_k, self.opt.trust_radius),
                theta - theta_k <= m_vec,
                theta_k - theta <= m_vec,
                A >= self.opt.A_min,
                A <= self.opt.A_max
            ]


            
            # 体积约束
            element_lengths = self.opt.geometry_calc.compute_element_lengths(self.opt.geometry)
            constraints.append(cp.sum(cp.multiply(A, element_lengths)) <= self.opt.volume_constraint)

            # AASI：若存在按上一轮计算的受压杆最小截面积下界 A_req，则加入线性不等式 A >= A_req
            # 并进行体积可行性保护：?sum(max(A_min, A_req) * L) > volume_constraint，则按比例缩?A_req
            try:
                if getattr(self.opt, 'phase', 'A') == 'C' and hasattr(self.opt, 'A_req_buckling') and self.opt.A_req_buckling is not None:
                    A_req_raw = np.asarray(self.opt.A_req_buckling, dtype=float)
                    if A_req_raw.shape[0] == self.opt.n_elements:
                        # 基于体积约束的可行性检查与缩放
                        A_req_clipped = np.maximum(self.opt.A_min, A_req_raw)
                        V_req = float(np.sum(A_req_clipped * element_lengths))
                        V_cap = float(self.opt.volume_constraint)
                        if V_req > V_cap:
                            # 计算缩放系数，留?%裕度
                            scale = max(1e-6, 0.95 * V_cap / V_req)
                            A_req_scaled = np.maximum(self.opt.A_min, scale * A_req_raw)
                            V_scaled = float(np.sum(A_req_scaled * element_lengths))
                            print("  ⚠️  AASI 下界总体积超出体积约束，按比例缩放以恢复可行性：")
                            print(f"     V_req = {V_req:.6e} m^3 > V_cap = {V_cap:.6e} m^3, scale = {scale:.4f}")
                            print(f"     V_scaled = {V_scaled:.6e} m^3 (占比 {V_scaled / V_cap:.1%})")
                            constraints.append(A >= cp.Constant(A_req_scaled))
                        else:
                            constraints.append(A >= cp.Constant(A_req_clipped))
                        print("  已加?AASI 屈曲约束: A >= A_req（含体积可行性保护）")
            except Exception:
                # 该约束失败不应中断主流程；继续以基础约束求解
                pass
            
            # 6. 求解
            print("  步骤5: 求解SDP...")
            objective = cp.Minimize(t)
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.MOSEK, verbose=False)
            
            if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                print(f"  ?联合优化成功，目标? {t.value:.6e}")
                # 缓存线性化模型的必要信息（?k 点）用于预测柔度
                try:
                    weighted_grad_matrices = []
                    for j in range(len(theta_k)):
                        weighted = np.zeros((self.opt.n_dof, self.opt.n_dof))
                        for i in range(self.opt.n_elements):
                            weighted += A_k[i] * grad_K_list[j][i]
                        weighted_grad_matrices.append(weighted)
                    self.opt._cached_linear_model = {
                        'A_k': A_k.copy(),
                        'theta_k': theta_k.copy(),
                        'K_current_list': K_current_list,   # list of numpy arrays
                        'weighted_grad_matrices': weighted_grad_matrices,  # list of numpy arrays
                        'f_current': f_current.copy(),
                        'grad_f': [g.copy() for g in grad_f],
                        'selection_matrix': selection_matrix.copy()
                    }
                except Exception as _:
                    # 缓存失败不影响主流程
                    pass
                
                return A.value, theta.value, t.value
            else:
                print(f"  ?联合优化失败: {prob.status}")
                return None
                
        except Exception as e:
            print(f"  ?联合优化出错: {e}")
            import traceback
            traceback.print_exc()
            return None

# ============================================
# 初始化管理器
# ============================================

class InitializationManager:
    """初始化管理器"""
    
    def __init__(self, geometry_params: GeometryParams, material_data):
        self.geometry_params = geometry_params
        self.material_data = material_data
    
    def initialize_node_angles(self, node_ids: List[int]) -> np.ndarray:
        """初始化节点角度"""
        angles = np.linspace(
            self.geometry_params.boundary_buffer,
            np.pi - self.geometry_params.boundary_buffer,
            len(node_ids)
        )
        print(f"初始化节点角度: {angles * 180/np.pi} degrees")
        return angles
    
    def initialize_areas(self, n_elements: int, element_lengths: np.ndarray, 
                        volume_constraint: float) -> np.ndarray:
        """初始化截面积"""
        uniform_area = volume_constraint / np.sum(element_lengths) * 0.8
        areas = np.full(n_elements, uniform_area)
        areas = np.clip(areas, self.material_data.A_min, self.material_data.A_max)
        
        print(f"初始截面? 均匀 {uniform_area*1e6:.2f} mm^2")
        return areas
    
    






