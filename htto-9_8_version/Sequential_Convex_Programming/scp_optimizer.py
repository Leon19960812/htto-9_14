"""

序列凸优化器主控制器

负责协调各个算法组件，控制优化流?"""

import numpy as np

import matplotlib.pyplot as plt

import warnings

from typing import Optional, Tuple

import json

import csv

import os

# 抑制libpng警告

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# 导入基础计算模块

from truss_system_initializer import TrussSystemInitializer

# 导入算法模块

from algorithm_modules import (

    TrustRegionParams, GeometryParams, OptimizationParams,

    TrustRegionManager, ConvergenceChecker, SubproblemSolver, 

    InitializationManager, SystemCalculator, StepQualityEvaluator

)

# 导入可视化
from visualization import TrussVisualization

class SequentialConvexTrussOptimizer:
    # 序列凸优化器主控制器

    def __init__(self, radius=2.0, n_sectors=12, inner_ratio=0.7, 
                 depth=50, volume_fraction=0.1,
                 enable_middle_layer=False, middle_layer_ratio=0.85,
                 enable_aasi: bool = True,
                 polar_rings: Optional[list] = None):
        # 初始化优化器

        print("初始化SCP...")

        # 保存参数
        self.radius = radius

        self.depth = depth

        self.enable_middle_layer = enable_middle_layer

        self.middle_layer_ratio = middle_layer_ratio

        # 是否启用 AASI 稳定性约束（用于对照消融实验?
        self.enable_aasi = enable_aasi

        # 1. 初始化基础系统

        polar_cfg = {'rings': polar_rings} if polar_rings is not None else None
        self.initializer = TrussSystemInitializer(
            radius=radius,
            n_sectors=n_sectors, 
            inner_ratio=inner_ratio,
            depth=depth,
            volume_fraction=volume_fraction,
            enable_middle_layer=enable_middle_layer,
            middle_layer_ratio=middle_layer_ratio,
            use_polar=True,
            polar_config=polar_cfg
        )
        
        # 2. 从初始化器获取所有必要属?
        self.__dict__.update(self.initializer.__dict__)

        # 4. 设置优化参数
        self._setup_optimization_params()

        # 5. 初始化优化状?
        self._initialize_optimization_state()

        # 6. 创建算法组件
        self._create_algorithm_modules()

        print("[OK] 序列凸优化器初始化完成")
        
        # 严格模式：出现异常不做兜底回退，直接抛错终?
        self.strict_mode = True

    def _setup_optimization_params(self):

        # 设置优化参数

        self.trust_region_params = TrustRegionParams()

        self.geometry_params = GeometryParams()

        self.optimization_params = OptimizationParams()

    def _initialize_optimization_state(self):

        """初始化优化状?""

        self.current_angles = None

        self.current_areas = None

        self.current_compliance = None

        self.trust_radius = self.trust_region_params.initial_radius

        self.iteration_count = 0

        # 添加信赖域半径跟?
        self.trust_radius_history = [self.trust_radius]

        self.trust_radius_changes = []  # 记录变化事件

        # 接受步后的柔度历史（单调不增?
        self.compliance_history = []

        # 分阶段控制与统计
        self.phase = 'A'  # A: 拓扑成形; B: 几何细化; C: 稳定性强?
        self._accepted_window = []  # 最近接受步?(removed_count, improvement%)

        # 线搜?步长跟踪
        self.alpha_history = []

        self.step_norm_history = []  # [(||dtheta||2, ||dA||2)]

    def _create_algorithm_modules(self):
        # 创建算法组件
        # 信赖域管理器
        self.trust_region_manager = TrustRegionManager(self.trust_region_params)

        # 收敛检查器
        self.convergence_checker = ConvergenceChecker(

            tolerance=self.optimization_params.convergence_tol,

            trust_region_manager=self.trust_region_manager

        )

        # 子问题求解器
        self.subproblem_solver = SubproblemSolver(self)

        # 系统计算?
        self.system_calculator = SystemCalculator(self)

        # 步长质量评估?
        self.step_evaluator = StepQualityEvaluator(self)

        # 初始化管理器
        self.initialization_manager = InitializationManager(

            geometry_params=self.geometry_params,

            material_data=self.material_data

        )

    def _initialize_optimization_variables(self) -> Tuple[np.ndarray, np.ndarray]:

        # 初始化优化变?- 保持向后兼容

        theta_k = self.initialization_manager.initialize_node_angles(self.geometry.load_nodes)
        A_k = self.initialization_manager.initialize_areas(

            self.geometry.n_elements,

            self.element_lengths,

            self.volume_constraint

        )

        # 记录初始角度用于相邻移动上限
        self.initial_angles = theta_k.copy()

        try:

            import numpy as _np

            self.theta_node_ids = _np.array(self.geometry.load_nodes[:len(theta_k)], dtype=int)
        except Exception:

            self.theta_node_ids = None

        return theta_k, A_k

    def _compute_member_forces_and_lengths(self, theta: np.ndarray, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # 计算每根构件的轴力 N_i 与长度 L_i（约定受压为正）
        # 1) 更新坐标与几何
        node_coords = self.geometry_calc.update_node_coordinates(self.geometry, theta, self.radius)

        lengths, directions = self.geometry_calc.compute_element_geometry(node_coords, self.geometry.elements)

        # 2) 组装刚度并求解位?        K_global = self.stiffness_calc.assemble_global_stiffness(self.geometry, A, lengths, directions)

        f_global = self.load_calc.compute_load_vector(node_coords, self.geometry.load_nodes, self.depth)

        K_red = K_global[np.ix_(self.free_dofs, self.free_dofs)]

        f_red = f_global[self.free_dofs]

        # 稳健求解位移：对角正则化 + 伪逆回退，避免奇异导致全零内?        def _stable_solve(K: np.ndarray, f: np.ndarray, kappa_max: float = 1e10):

            try:

                return np.linalg.solve(K, f)

            except np.linalg.LinAlgError:

                pass

            lam_base = float(np.maximum(1e-16, np.mean(np.diag(K)) if K.size > 0 else 1.0))

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

            try:

                return np.linalg.pinv(K, rcond=1e-10) @ f

            except Exception:

                return None

        U_red = _stable_solve(K_red, f_red)

        if U_red is None:

            # 无法可靠求解，返回零内力但保留长?            return np.zeros(self.n_elements), lengths

        U_full = np.zeros(self.n_dof)

        U_full[self.free_dofs] = U_red

        # 3) 轴力 N_i = (E*A/L) * ((u2-u1)·dir)

        E = getattr(self.material_data, 'E_steel', 210e9)

        N = np.zeros(self.n_elements)

        for i, (n1, n2) in enumerate(self.geometry.elements):

            u1 = np.array([U_full[2*n1], U_full[2*n1+1]])

            u2 = np.array([U_full[2*n2], U_full[2*n2+1]])

            du = u2 - u1

            axial_ext = float(np.dot(du, directions[i]))

            L = max(lengths[i], 1e-12)

            N[i] = (E * A[i] / L) * axial_ext

        return N, lengths

    def _build_aasi_buckling_lower_bounds(self, theta: np.ndarray, A: np.ndarray,

                                          eps: float = 1e-3, a_cr_min_ratio: float = 0.002,

                                          alpha: float = None, K_end: float = 1.0) -> np.ndarray:

        """基于 AASI 计算受压杆最小截面下?A_req?        仅对 A_i > a_cr_min 且受压的构件给出正的下界，其余取 A_min?""

        if alpha is None:

            alpha = 1.0/(4.0*np.pi)  # 圆实心近?I≈αA²

        E = getattr(self.material_data, 'E_steel', 210e9)

        N, L = self._compute_member_forces_and_lengths(theta, A)

        A_req = np.full(self.n_elements, self.A_min, dtype=float)

        if self.n_elements == 0:

            return A_req

        # 与删除阈值解耦：a_cr_min 不高于删除阈值，避免早期卡死

        a_cr_min = max(self.A_min, min(self.removal_threshold, a_cr_min_ratio * float(np.max(A))))

        C = (np.pi**2 * E * alpha) / (K_end**2)

        for i in range(self.n_elements):

            if A[i] > a_cr_min and N[i] > 0.0:  # 仅对受压且不太细的杆

                # A_req = sqrt(((1+eps)*N * (K L)^2) / (π^2 E α))

                A_need = np.sqrt(((1.0+eps) * N[i] * (K_end * L[i])**2) / max(C, 1e-30)) if C>0 else self.A_min

                # 施加上限避免高于删除阈值太多（减少对删除的抑制?                A_cap = 1.2 *
        self.removal_threshold

                A_req[i] = min(max(self.A_min, a_cr_min, A_need), A_cap)

        return A_req

    def _update_trust_radius(self):

        """更新信赖域半?""

        self.trust_radius = self.trust_region_manager.current_radius

    def solve_scp_optimization(self):

        """主优化方?""

        print("\n" + "=" * 80)

        print("GEOMETRY-TOPOLOGY JOINT OPTIMIZATION")

        print("=" * 80)

        # 1. 初始?        print("初始化优?..")

        theta_k, A_k = self._initialize_optimization_variables()

        # 计算初始柔度
        self.current_compliance = self.system_calculator.compute_actual_compliance(theta_k, A_k)

        # 🔧 修复：立即设置当前状态，避免None值错?
        self.current_angles = theta_k

        self.current_areas = A_k

        # 初始化接受历史（?次）
        self.compliance_history = [self.current_compliance]

        print(f"初始设置:")

        print(f"  节点? {len(theta_k)}")

        print(f"  单元? {self.n_elements}")

        print(f"  初始柔度: {self.current_compliance:.6e}")

        print(f"  信赖域半? {self.trust_radius:.4f}")

        # 2. 主优化循?        success_count = 0

        for self.iteration_count in range(self.optimization_params.max_iterations):

            print(f"\n{'='*60}")

            print(f"迭代 {self.iteration_count + 1}/{self.optimization_params.max_iterations}")

            print(f"{'='*60}")

            try:

                # 求解子问?                print("求解联合线性化子问?..")

                # 仅在相位 C 构造并记录 AASI 下界；A/B 阶段不生成（避免误解为约束已启用?                if
        self.phase == 'C':

                    try:

                        self.A_req_buckling = self._build_aasi_buckling_lower_bounds(theta_k, A_k, eps=1e-3, a_cr_min_ratio=0.002)

                        n_active = int(np.sum(self.A_req_buckling > self.A_min + 1e-16))

                        print(f"已生?AASI 屈曲下界（C阶段），激活构件数: {n_active}/{len(self.A_req_buckling)}")

                    except Exception as _e:

                        print(f"⚠️ AASI 下界生成失败: {_e}")

                        self.A_req_buckling = None

                else:

                    self.A_req_buckling = None

                result = self.subproblem_solver.solve_linearized_subproblem(A_k, theta_k)

                if result is None:

                    print("[ERROR] 子问题求解失?)

                    if getattr(self, 'strict_mode', False):

                        raise RuntimeError("Linearized subproblem failed")

                    else:

                        if self._handle_subproblem_failure():

                            continue

                        else:

                            break

                A_new, theta_new, compliance_new = result

                # 缓存梯度信息用于预测柔度计算

                try:

                    grad_theta, grad_A = self.subproblem_solver.gradient_calc.compute_gradients(theta_k)

                    # 验证梯度有效?                    if grad_theta is None or grad_A is None:

                        raise ValueError("梯度计算返回 None")

                    if not isinstance(grad_theta, (list, np.ndarray)) or not isinstance(grad_A, (list, np.ndarray)):

                        raise ValueError(f"梯度类型错误：grad_theta={type(grad_theta)}, grad_A={type(grad_A)}")

                    self._cached_gradients = (grad_theta, grad_A)

                    print(f"?梯度缓存成功")

                except Exception as e:

                    print(f"[ERROR] 梯度缓存失败: {e}")

                    print(f"   梯度计算失败，将使用有限差分方法")

                    self._cached_gradients = None

                # 线搜?+ 刚度正定守护：沿(Δθ,ΔA)回溯 alpha，确?K_ff 可Cholesky 且条件数合理

                def _spd_ok(theta_cand, A_cand, kappa_max: float = 1e9):

                    try:

                        coords = self._update_node_coordinates(theta_cand)

                        elen, edir = self._compute_element_geometry(coords)

                        K_global = self._assemble_global_stiffness(A_cand, elen, edir)

                        K_red = K_global[np.ix_(self.free_dofs, self.free_dofs)]

                        np.linalg.cholesky(K_red)

                        cond_val = float(np.linalg.cond(K_red))

                        if np.isfinite(cond_val) and cond_val <= kappa_max:

                            return True, cond_val

                        return False, cond_val

                    except Exception:

                        return False, float('inf')

                # 单层模式：直接计算theta变化

                dtheta = theta_new - theta_k

                base_dtheta_norm = float(np.linalg.norm(dtheta))

                dA = A_new - A_k

                base_dA_norm = float(np.linalg.norm(dA))

                alpha = 1.0

                beta = 0.5

                max_bt = 3

                chosen_cond = None

                trial_record = []

                for _ in range(max_bt + 1):

                    # 单层模式

                    theta_cand = theta_k + alpha * dtheta

                    A_cand = A_k + alpha * dA

                    ok, cond_val = _spd_ok(theta_cand, A_cand)

                    trial_record.append((alpha, cond_val, ok))

                    if ok:

                        chosen_cond = cond_val

                        break

                    alpha *= beta

                # 输出线搜索与步长信息（无论是否回溯）

                tried_str = " ?".join([f"{a:.3f}" for a, _, _ in trial_record])

                cond_str = f"{chosen_cond:.3e}" if chosen_cond is not None else "nan"

                print(f"  Line search (SPD guard): α_final={alpha:.3f} | tried: {tried_str} | cond(K_ff)≈{cond_str}")

                print(f"  Step norms: ||Δθ||2={base_dtheta_norm:.3e}, ||ΔA||2={base_dA_norm:.3e}")

                # 记录 SPD 阶段结果，供日志使用

                alpha_spd_final = float(alpha)

                spd_trials = trial_record.copy()

                # 记录到历?
        self.alpha_history.append(float(alpha))

                self.step_norm_history.append((base_dtheta_norm, base_dA_norm))

                # 用线搜索后的解替换候选，从而复用后续流?                theta_new = theta_k + alpha * dtheta

                A_new = A_k + alpha * dA

                # 保存最近α供评估模块/日志使用
        self._last_alpha = float(alpha)

                # 显示变化情况
        self._print_iteration_info(theta_k, theta_new, A_k, A_new)

                # 第二阶段：基于ρ的回溯（质量守护）

                rho_bt_max = 3

                rho_target = self.trust_region_params.accept_threshold

                alpha_quality = float(alpha)

                rho = None

                quality_trials = []

                for _ in range(rho_bt_max + 1):

                    # 用当?alpha 生成候?                    theta_q = theta_k + alpha_quality * dtheta

                    A_q = A_k + alpha_quality * dA

                    try:

                        rho_try = self.step_evaluator.evaluate_step_quality(

                            theta_k, theta_q, A_q,

                            predicted_from_model=(compliance_new if abs(alpha_quality-1.0) < 1e-12 else None)

                        )

                    except Exception as e:

                        print(f"  [ERROR] 步长质量评估失败(α={alpha_quality:.3f}): {e}")

                        # 若评估失败，视为质量很差，继续回?                        rho_try = -np.inf

                    quality_trials.append((alpha_quality, rho_try))

                    if np.isfinite(rho_try) and rho_try >= rho_target:

                        rho = rho_try

                        theta_new, A_new = theta_q, A_q

                        break

                    alpha_quality *= beta

                if rho is None and quality_trials:

                    # 未达到目标，使用最后一次（最小α）的结?                    alpha_quality, rho = quality_trials[-1]

                    theta_new = theta_k + alpha_quality * dtheta

                    A_new = A_k + alpha_quality * dA

                # 打印质量回溯记录

                trials_str = " | ".join([f"α={a:.3f}, ρ={r:.3f}" if np.isfinite(r) else f"α={a:.3f}, ρ=nan" for a, r in quality_trials])

                print(f"  Quality backtracking: {trials_str}")

                if abs(alpha_quality - alpha) > 1e-12:

                    print(f"  α adjusted by ρ: {alpha:.3f} ?{alpha_quality:.3f}")

                # 保存最终?                alpha = float(alpha_quality)
        self._last_alpha = float(alpha)

                # 纠正/记录 α 历史为最终?                try:

                    if hasattr(self, 'alpha_history'):

                        if len(self.alpha_history) > 0:

                            self.alpha_history[-1] = float(alpha)

                        else:

                            self.alpha_history.append(float(alpha))

                except Exception:

                    pass

                if hasattr(self, 'step_details') and self.step_details:

                    try:

                        self.step_details[-1]['alpha'] = float(alpha)

                    except Exception:

                        pass

                # 更新信赖?                old_radius = self.trust_radius

                accept_step = self.trust_region_manager.update_radius(rho)

                self._update_trust_radius()

                # 记录信赖域半径变?
        self.trust_radius_history.append(self.trust_radius)

                if self.trust_radius != old_radius:

                    change_type = "EXPAND" if self.trust_radius > old_radius else "SHRINK"

                    self.trust_radius_changes.append({

                        'iteration': self.iteration_count + 1,

                        'old_radius': old_radius,

                        'new_radius': self.trust_radius,

                        'type': change_type,

                        'rho': rho

                    })

                else:

                    change_type = "SAME"

                # 接受或拒绝步?                if accept_step:

                    # 先检查收敛（用更新前的值）

                    converged = self.convergence_checker.check_convergence(theta_k, theta_new, A_k, A_new)

                    if converged:

                        if self._aasi_stability_ok(theta_new, A_new):

                            print(f"\n🎉 算法收敛（含AASI稳定性）")

                        break

                    theta_k = theta_new

                    A_k = A_new

                    old_compliance = self.current_compliance

                    self.current_compliance = self.system_calculator.compute_actual_compliance(theta_new, A_new)

                    improvement = (old_compliance - self.current_compliance) / old_compliance * 100

                    success_count += 1

                    print(f"?接受步长 (第{success_count}次成?")

                    print(f"   柔度: {old_compliance:.6e} ?{self.current_compliance:.6e}")

                    print(f"   改进: {improvement:.2f}%")

                    # —?写入日志（接受步?—?                    try:

                        sd = self.step_details[-1] if hasattr(self, 'step_details') and self.step_details else {}

                        self._append_iteration_log({

                            'iteration': self.iteration_count + 1,

                            'phase': self.phase,

                            'alpha_spd_final': alpha_spd_final,

                            'alpha_final': float(alpha),

                            'spd_trials': ";".join([f"{a:.3f}:{('ok' if ok else 'fail')}:{(c if np.isfinite(c) else float('nan')):.3e}" for a, c, ok in spd_trials]) if spd_trials else "",

                            'quality_trials': ";".join([f"{a:.3f}:{(r if np.isfinite(r) else float('nan')):.3f}" for a, r in quality_trials]) if quality_trials else "",

                            'rho': float(rho) if np.isfinite(rho) else float('nan'),

                            'cond_chosen': float(chosen_cond) if chosen_cond is not None and np.isfinite(chosen_cond) else float('nan'),

                            'step_norm_dtheta': base_dtheta_norm,

                            'step_norm_dA': base_dA_norm,

                            'trust_radius_old': old_radius,

                            'trust_radius_new': self.trust_radius,

                            'trust_update_type': change_type,

                            'accept_step': True,

                            'current_compliance_before': old_compliance,

                            'actual_compliance': float(sd.get('actual_compliance', float('nan'))),

                            'predicted_compliance': float(sd.get('predicted_compliance', float('nan'))),

                            'improvement_percent': float(improvement)

                        })

                    except Exception:

                        pass

                    # —?阶段统计：删除数量与改进幅度 —?                    prev_active = int(np.sum(self.current_areas >
        self.removal_threshold)) if self.current_areas is not None else int(np.sum(A_k > self.removal_threshold))

                    new_active = int(np.sum(A_k > self.removal_threshold))

                    removed_count = max(0, prev_active - new_active)

                    self._accepted_window.append((removed_count, float(improvement)))

                    if len(self._accepted_window) > 5:

                        self._accepted_window.pop(0)

                    # 保存当前状?
        self.current_angles = theta_k

                    self.current_areas = A_k

                    # 记录接受后的柔度
        self.compliance_history.append(self.current_compliance)

                    # 回写接受标记到最后一?step_detail

                    if hasattr(self, 'step_details') and self.step_details:

                        self.step_details[-1]['accepted'] = True

                        self.step_details[-1]['accepted_compliance'] = self.current_compliance

                    # —?阶段切换 —?                    if len(self._accepted_window) >= 5:

                        win = self._accepted_window[-5:]

                        removed_sum = sum(x for x,_ in win)

                        avg_impr = sum(y for _,y in win) / len(win)

                        if self.phase == 'A' and removed_sum <= 3 and avg_impr < 0.5:

                            self.phase = 'B'

                            print("[Phase Switch] A ?B（拓扑基本定型，开始几何细化）")

                        if self.phase in ('A','B') and avg_impr < 0.3:

                            if self.enable_aasi:

                                self.phase = 'C'

                                print("[Phase Switch] 进入 C（启用稳定性约?AASI?)

                            else:

                                # 禁用 AASI 时不进入 C，相当于仅进?A/B 阶段的对照实?                                print("[Phase Switch] AASI 已禁用，跳过进入 C 阶段")

                    # 节点融合检?                    merge_groups = self.initializer.group_nodes_by_radius(theta_k)

                    if (self.phase in ('B', 'C')) and merge_groups:

                        merge_info = self.initializer.merge_node_groups(theta_k, A_k, merge_groups)

                        if merge_info['structure_modified']:

                            # 更新当前状?                            theta_k = merge_info['theta_updated']

                            A_k = merge_info['A_updated']

                            self.current_angles = theta_k

                            self.current_areas = A_k

                            # 更新几何结构
        self.initializer.geometry = merge_info['geometry_updated']

                            self.geometry = merge_info['geometry_updated']

                            self.elements = self.geometry.elements  # 更新elements引用
                            # 同步兼容性（legacy）属性，避免可视化和其他模块索引不一致
                            # 这些属性在初始化时由 geometry 派生，此处必须在几何更新后刷新
        self.nodes = self.geometry.nodes
                            self.load_nodes = self.geometry.load_nodes

                            self.inner_nodes = self.geometry.inner_nodes
                            if hasattr(self.geometry, 'middle_nodes'):
                                self.middle_nodes = self.geometry.middle_nodes
                            try:
                                import numpy as _np
                                self.theta_node_ids = _np.array(self.load_nodes[:len(theta_k)], dtype=int)
                            except Exception:
                                self.theta_node_ids = None

                            # 更新基本参数
        self.n_nodes = merge_info['geometry_updated'].n_nodes

                            self.n_dof = merge_info['geometry_updated'].n_dof

                            self.n_elements = merge_info['geometry_updated'].n_elements

                            # 更新边界条件
        self.fixed_dofs = merge_info['geometry_updated'].fixed_dofs

                            self.free_dofs = merge_info['geometry_updated'].free_dofs

                            # 更新element_lengths
        self.element_lengths = self.geometry_calc.compute_element_lengths(self.geometry)

                            # 重新初始化load_calc（因为load_nodes数量变化?
        self._reinitialize_load_calculator()

                            # 强制重线性化 - 清除所有缓存，因为几何结构已改?
        self._clear_linearization_cache()

                            # 重新计算预计算刚度矩阵（因为elements已改变）
        self.unit_stiffness_matrices = self.stiffness_calc.precompute_unit_stiffness_matrices(

                                self.geometry, self.element_lengths

                            )

                            # 合并后基线柔度与新几何保持一致，防止后续步长质量评估失配

                            try:

                                self.current_compliance = self.system_calculator.compute_actual_compliance(theta_k, A_k)

                                print(f"   合并后重新评估柔? {self.current_compliance:.6e}")

                            except Exception as _e:

                                print(f"   ⚠️ 合并后柔度重评失? {_e}")

                            print(f"   重新计算预计算刚度矩阵，?{len(self.unit_stiffness_matrices)} 个单?)                        

                    # ǶȲñ˳ȣ

                    try:

                        updated_coords_caps = self.geometry_calc.update_node_coordinates(self.geometry, theta_k, self.radius)

                        elen_caps, _edir = self.geometry_calc.compute_element_geometry(updated_coords_caps, self.geometry.elements)

                        # ӳ䣺node_id -> theta /ŻĽڵ㣩

                        idx_map = {}

                        for _j, _nid in enumerate(self.theta_node_ids if ('numpy' in str(type(self.theta_node_ids))) or (hasattr(self, 'theta_node_ids') and self.theta_node_ids is not None) else self.geometry.load_nodes[:len(theta_k)]):

                            idx_map[_nid] = _j

                        dmin = np.full(len(theta_k), np.inf, dtype=float)

                        for _ei, (_n1, _n2) in enumerate(self.geometry.elements):

                            L = float(elen_caps[_ei]) if _ei < len(elen_caps) else float('inf')

                            if np.isfinite(L):

                                if _n1 in idx_map:

                                    dmin[idx_map[_n1]] = min(dmin[idx_map[_n1]], L)

                                if _n2 in idx_map:

                                    dmin[idx_map[_n2]] = min(dmin[idx_map[_n2]], L)

                        gp = self.geometry_params

                        R = max(1e-12, float(self.radius))

                        k_fac = 1.0/3.0

                        m_arc = np.minimum(k_fac * dmin, R * gp.neighbor_move_cap)

                        theta_caps = np.clip(m_arc / R, 0.0, gp.neighbor_move_cap)

                        # ʱ

                        theta_caps[~np.isfinite(theta_caps)] = max(0.0, gp.neighbor_move_cap - gp.neighbor_move_eps)

                        self.theta_move_caps = theta_caps

                        # ¼ step_details ڵ

                        if hasattr(self, 'step_details') and self.step_details:

                            self.step_details[-1]['theta_move_caps'] = theta_caps.tolist()

                    except Exception as _e_caps:

                        # ʧʱ caps Ի˵ȫñ
        self.theta_move_caps = None

                else:

                    print("[ERROR] 拒绝步长")

                    print(f"   保持当前解，柔度: {self.current_compliance:.6e}")

                    # 回写拒绝标记到最后一?step_detail

                    if hasattr(self, 'step_details') and self.step_details:

                        self.step_details[-1]['accepted'] = False

                        self.step_details[-1]['accepted_compliance'] = self.current_compliance

                    # —?写入日志（拒绝步?—?                    try:

                        sd = self.step_details[-1] if hasattr(self, 'step_details') and self.step_details else {}

                        self._append_iteration_log({

                            'iteration': self.iteration_count + 1,

                            'phase': self.phase,

                            'alpha_spd_final': alpha_spd_final,

                            'alpha_final': float(alpha),

                            'spd_trials': ";".join([f"{a:.3f}:{('ok' if ok else 'fail')}:{(c if np.isfinite(c) else float('nan')):.3e}" for a, c, ok in spd_trials]) if spd_trials else "",

                            'quality_trials': ";".join([f"{a:.3f}:{(r if np.isfinite(r) else float('nan')):.3f}" for a, r in quality_trials]) if quality_trials else "",

                            'rho': float(rho) if np.isfinite(rho) else float('nan'),

                            'cond_chosen': float(chosen_cond) if chosen_cond is not None and np.isfinite(chosen_cond) else float('nan'),

                            'step_norm_dtheta': base_dtheta_norm,

                            'step_norm_dA': base_dA_norm,

                            'trust_radius_old': old_radius,

                            'trust_radius_new': self.trust_radius,

                            'trust_update_type': change_type,

                            'accept_step': False,

                            'current_compliance_before': self.current_compliance,

                            'actual_compliance': float(sd.get('actual_compliance', float('nan'))),

                            'predicted_compliance': float(sd.get('predicted_compliance', float('nan'))),

                            'improvement_percent': float(0.0)

                        })

                    except Exception:

                        pass

            except KeyboardInterrupt:

                print("\n⚠️  用户中断优化")

                break

            except Exception as e:

                print(f"[ERROR] 迭代 {self.iteration_count + 1} 失败: {e}")

                if getattr(self, 'strict_mode', False):

                    raise

                else:

                    if self._handle_iteration_failure():

                        continue

                    else:

                        break

        # 3. 优化完成
        self._print_optimization_summary(success_count)

        # 设置最终结果属?
        self._set_final_results()

        return self.current_areas, self.current_compliance

    def _handle_subproblem_failure(self) -> bool:

        """处理子问题求解失?""

        if self.trust_radius <= 1.1 * self.trust_region_params.min_radius:

            print("信赖域半径过小，停止优化")

            return False

        else:

            old_radius = self.trust_radius

            self.trust_radius *= 0.5

            self.trust_region_manager.current_radius = self.trust_radius

            # 记录信赖域半径变?
        self.trust_radius_history.append(self.trust_radius)

            self.trust_radius_changes.append({

                'iteration': self.iteration_count + 1,

                'old_radius': old_radius,

                'new_radius': self.trust_radius,

                'type': "SHRINK_FAILURE",

                'rho': 0.0  # 失败时rho?

            })

            print(f"缩小信赖域至 {self.trust_radius:.4f}，重?)

            return True

    def _handle_iteration_failure(self) -> bool:

        # 处理迭代失败

        old_radius = self.trust_radius

        self.trust_radius = max(self.trust_radius * 0.5, self.trust_region_params.min_radius)

        self.trust_region_manager.current_radius = self.trust_radius

        # 记录信赖域半径变?
        self.trust_radius_history.append(self.trust_radius)

        self.trust_radius_changes.append({

            'iteration': self.iteration_count + 1,

            'old_radius': old_radius,

            'new_radius': self.trust_radius,

            'type': "SHRINK_FAILURE",

            'rho': 0.0  # 失败时rho?

        })

        if self.trust_radius <= 1.1 * self.trust_region_params.min_radius:

            print("信赖域过小，停止优化")

            return False

        return True

    def _print_iteration_info(self, theta_k, theta_new, 

                             A_k: np.ndarray, A_new: np.ndarray):

        # 打印迭代信息

        # һάtheta

        if isinstance(theta_k, np.ndarray) and isinstance(theta_new, np.ndarray):

            total_change_sq = float(np.linalg.norm(theta_new - theta_k) ** 2)

            max_change = float(np.max(np.abs(theta_new - theta_k)))

            layer_changes = None

            theta_change = np.sqrt(total_change_sq)

            print(f"  变化情况:")

            print(f"    θ总变? {theta_change:.6e} (最大变? {max_change:.6e})")

            print(f"    θ变化: {theta_change:.6e} (最大变? {np.max(np.abs(theta_new - theta_k)):.6e})")

        A_change = np.linalg.norm(A_new - A_k)

        print(f"    A变化: {A_change:.6e} (最大变? {np.max(np.abs(A_new - A_k)):.6e})")

    def _print_optimization_summary(self, success_count: int):

        # 打印优化总结

        print(f"\n{'='*80}")

        print("优化完成")

        print(f"{'='*80}")

        print(f"总迭代次? {self.iteration_count + 1}")

        print(f"成功步长: {success_count}")

        print(f"最终柔? {self.current_compliance:.6e}")

        print(f"最终信赖域半径: {self.trust_radius:.6f}")

        # 保存最终结?
        self.initializer.final_angles = self.current_angles

        self.initializer.final_areas = self.current_areas

        self.initializer.final_compliance = self.current_compliance

        # 自动导出步长详细信息（只导出可序列化的关键字段）

        try:

            export_dir = os.path.join("results")

            os.makedirs(export_dir, exist_ok=True)

            # 先构造精简且可序列化的步骤列表

            raw_steps = getattr(self, 'step_details', [])

            def to_serializable(x):

                try:

                    if x is None:

                        return None

                    import numpy as _np

                    if isinstance(x, (_np.floating, _np.integer)):

                        return float(x)

                    if isinstance(x, _np.ndarray):

                        return x.tolist()

                    if isinstance(x, (list, tuple)):

                        return [to_serializable(v) for v in x]

                    if isinstance(x, (float, int, str, bool, dict)):

                        return x

                    return str(x)

                except Exception:

                    return None

            kept_fields = [

                'iteration','rho','pred_source','trust_radius',

                'current_compliance','actual_compliance','predicted_compliance',

                'actual_reduction','predicted_reduction',

                'half_step_compliance','deltaC_A','deltaC_theta',

                'cond_K_half','cond_K_full',

                'active_set_removed','active_set_added','angle_projection_flags',

                'top_element_contribs','top_angle_contribs','alpha'

            ]

            safe_steps = []

            for step in raw_steps:

                row = {}

                for k in kept_fields:

                    row[k] = to_serializable(step.get(k, None))

                safe_steps.append(row)

            # JSON 完整导出（精简后的?            json_path = os.path.join(export_dir, "step_details.json")

            tmp_json = json_path + ".tmp"

            with open(tmp_json, "w", encoding="utf-8") as f:

                json.dump(safe_steps, f, ensure_ascii=False, indent=2)

            os.replace(tmp_json, json_path)

            print(f"已导? {json_path}")

            # CSV 精简导出（每步关键信息）

            csv_path = os.path.join(export_dir, "step_details_summary.csv")

            fields = [

                'iteration','rho','pred_source','trust_radius',

                'current_compliance','actual_compliance','predicted_compliance',

                'actual_reduction','predicted_reduction',

                'half_step_compliance','deltaC_A','deltaC_theta',

                'cond_K_half','cond_K_full','alpha'

            ]

            tmp_csv = csv_path + ".tmp"

            with open(tmp_csv, "w", newline="", encoding="utf-8") as f:

                writer = csv.DictWriter(f, fieldnames=fields)

                writer.writeheader()

                for step in safe_steps:

                    row = {k: step.get(k, None) for k in fields}

                    writer.writerow(row)

            os.replace(tmp_csv, csv_path)

            print(f"已导? {csv_path}")

        except Exception as e:

            print(f"导出 step_details 失败: {e}")

    def _set_final_results(self):

        """设置最终结果属性，用于可视?""

        self.final_areas = self.current_areas

        self.final_compliance = self.current_compliance

        self.verification_passed = True

        self.final_angles = self.current_angles

    def _update_node_coordinates(self, theta) -> np.ndarray:

        # 更新节点坐标

        return self.geometry_calc.update_node_coordinates(self.geometry, theta, self.radius)

    def _compute_element_geometry(self, node_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        # 计算单元几何 - 为兼容性保留的方法

        return self.geometry_calc.compute_element_geometry(node_coords, self.geometry.elements)

    def _assemble_global_stiffness(self, A: np.ndarray, element_lengths: np.ndarray, 

                                  element_directions: np.ndarray) -> np.ndarray:

        # 组装全局刚度矩阵 - 为兼容性保留的方法

        return self.stiffness_calc.assemble_global_stiffness(

            self.geometry, A, element_lengths, element_directions

        )

    def _compute_load_vector(self, node_coords: np.ndarray) -> np.ndarray:

        """计算载荷向量 - 使用壳体FEA动态计?""

        # 统一通过 load_calc 接口调用，避免重复计?        return
        self.load_calc.compute_load_vector(node_coords, self.geometry.load_nodes, self.depth)

    def _clear_linearization_cache(self):

        # 清除线性化缓存，强制重线性化

        # 清除优化器中的缓?        if hasattr(self, '_cached_linear_model'):
        self._cached_linear_model = None

        if hasattr(self, '_cached_gradients'):

            self._cached_gradients = None

        # 清除算法模块中的缓存

        if hasattr(self, 'subproblem_solver') and hasattr(self.subproblem_solver, 'opt'):

            if hasattr(self.subproblem_solver.opt, '_cached_linear_model'):

                self.subproblem_solver.opt._cached_linear_model = None

            if hasattr(self.subproblem_solver.opt, '_cached_gradients'):

                self.subproblem_solver.opt._cached_gradients = None

        # 清除其他可能的缓?        if hasattr(self, 'gradient_calculator'):

            if hasattr(self.gradient_calculator, '_cached_gradients'):

                self.gradient_calculator._cached_gradients = None

        print("   清除线性化缓存，强制重线性化")

    def _reinitialize_load_calculator(self):

        """重新初始化载荷计算器（节点融合后需要更新Shell FEA网格?""

        try:

            from load_calculator_with_shell import LoadCalculatorWithShell

            shell_params = {

                'outer_radius': self.radius,

                'depth': self.depth,

                'thickness': 0.01,

                'n_circumferential': 20,  # 使用固定?0个周向网?                'n_radial': 4,  # 使用固定?个径向网?                'E_shell':
        self.material_data.E_steel * 1000

            }

            self.load_calc = LoadCalculatorWithShell(

                material_data=self.material_data,

                enable_shell=True,

                shell_params=shell_params

            )

            print("   重新初始化载荷计算器（Shell FEA网格已更新）")

        except Exception as e:

            print(f"   ⚠️  载荷计算器重新初始化失败: {e}")

            # 如果重新初始化失败，继续使用原来的计算器

    def _append_iteration_log(self, row: dict, filepath: str = 'optimization_log.csv'):

        # 将关键迭代参数追加写入CSV日志?        字段包含：迭代号、phase、α（SPD/最终）、试探记录、ρ、cond、步长范数、信赖域变化、是否接受、柔度等?

        try:

            headers = [

                'iteration', 'phase',

                'alpha_spd_final', 'alpha_final',

                'spd_trials', 'quality_trials',

                'rho', 'cond_chosen',

                'step_norm_dtheta', 'step_norm_dA',

                'trust_radius_old', 'trust_radius_new', 'trust_update_type',

                'accept_step',

                'current_compliance_before', 'actual_compliance', 'predicted_compliance',

                'improvement_percent'

            ]

            file_exists = os.path.exists(filepath)

            with open(filepath, 'a', newline='', encoding='utf-8') as f:

                writer = csv.DictWriter(f, fieldnames=headers)

                if not file_exists:

                    writer.writeheader()

                # 只保留已知字段，避免 DictWriter 报错

                safe_row = {k: row.get(k, '') for k in headers}

                writer.writerow(safe_row)

        except Exception as e:

            print(f"   ⚠️ 写入日志失败: {e}")

    def verify_solution(self, areas: np.ndarray, theta: np.ndarray):

        """验证解的正确性（用最终theta和A重新生成结构参数?""

        print("\n" + "-" * 40)

        print("SOLUTION VERIFICATION")

        print("-" * 40)

        try:

            # 1. 用theta重算节点坐标

            node_coords = self.geometry_calc.update_node_coordinates(self.geometry, theta, self.radius)

            # 2. 重算单元几何

            element_lengths, element_directions = self.geometry_calc.compute_element_geometry(node_coords, self.geometry.elements)

            # 3. 重算全局刚度矩阵

            K_global = self.stiffness_calc.assemble_global_stiffness(

                self.geometry, areas, element_lengths, element_directions

            )

            # 4. 重算载荷向量

            f_global = self.load_calc.compute_load_vector(node_coords, self.geometry.load_nodes, self.depth)

            # 5. 获取自由度索?            free_dofs = self.free_dofs

            # 6. 约化

            K_red = K_global[np.ix_(free_dofs, free_dofs)]

            f_red = f_global[free_dofs]

            # 7. 求解

            U_red = np.linalg.solve(K_red, f_red)

            compliance_direct = np.dot(f_red, U_red)

            # 8. SCP流程的柔?            compliance_scp = self.system_calculator.compute_actual_compliance(theta, areas)

            # 9. 打印对比

            print(f"SCP compliance:    {compliance_scp:.6e}")

            print(f"Direct compliance: {compliance_direct:.6e}")

            print(f"Relative error:    {abs(compliance_scp - compliance_direct)/compliance_scp:.6e}")

            # 其余统计

            total_volume = np.sum(areas * element_lengths)

            effective_elements = np.sum(areas > self.removal_threshold)

            print(f"Total volume:      {total_volume*1e6:.1f} cm³")

            print(f"Volume constraint: {self.volume_constraint*1e6:.1f} cm³")

            print(f"Volume utilization: {total_volume/self.volume_constraint:.1%}")

            print(f"Effective elements: {effective_elements}/{self.n_elements}")

            active_areas = areas[areas > self.removal_threshold]

            if len(active_areas) > 0:

                print(f"Area range: [{np.min(active_areas)*1e6:.2f}, {np.max(active_areas)*1e6:.2f}] mm²")

            # 设置验证结果
        self.verification_passed = abs(compliance_scp - compliance_direct)/compliance_scp < 1e-3

            if self.verification_passed:

                print("?Verification PASSED")

            else:

                print("?Verification FAILED - Large discrepancy in compliance")

        except Exception as e:

            print(f"?Verification failed: {e}")

            self.verification_passed = False

def run_scp_optimization():

    # 运行SCP优化

    print("Starting SCP Truss Optimization...")

    try:

        # 创建优化?        print("\n" + "=" * 60)

        print("INITIALIZING OPTIMIZER")

        print("=" * 60)

        optimizer = SequentialConvexTrussOptimizer(

            radius=2.0,\n            n_sectors=12,\n            inner_ratio=0.7,\n            depth=50.0,\n            volume_fraction=0.1,\n            enable_middle_layer=False,\n            middle_layer_ratio=0.85,\n            enable_aasi=False\n        )

        print("Optimizer initialized successfully")

        # 求解优化问题

        print("\n" + "=" * 60)

        print("STARTING SCP OPTIMIZATION")

        print("=" * 60)

        optimizer.solve_scp_optimization()

        if optimizer.current_areas is not None and optimizer.current_compliance is not None:

            print("\n" + "=" * 80)

            print("OPTIMIZATION SUCCESSFUL!")

            print("=" * 80)

            print(f" Optimal compliance: {optimizer.current_compliance:.6e}")

            print(f" Effective members: {np.sum(optimizer.current_areas > optimizer.removal_threshold)}/{len(optimizer.current_areas)}")

            print(f" Volume utilization: {np.sum(optimizer.current_areas * optimizer.element_lengths)/optimizer.volume_constraint:.1%}")

            # 验证?            optimizer.verify_solution(optimizer.current_areas, optimizer.current_angles)

            optimizer.initializer.verification_passed = optimizer.verification_passed

            # 可视化结?            print("\n" + "=" * 60)

            print("GENERATING VISUALIZATION")

            print("=" * 60)

            visualizer = TrussVisualization()

            visualizer.visualize_results(optimizer)

            # 另存After Topology Cleanup单图

            visualizer.plot_single_figure(

                optimizer,

                figure_type="cleaned",

                save_path="results/after_cleanup.pdf",

                figsize=(10, 8)

            )

            # 单独导出面积分布直方图（SCP?            visualizer.plot_single_figure(

                optimizer,

                figure_type="area_histogram",

                save_path="results/area_histogram_scp.pdf",

                figsize=(8, 6)

            )

            # 单独导出信赖域演化图

            print("\n" + "=" * 60)

            print("GENERATING TRUST REGION EVOLUTION PLOT")

            print("=" * 60)

            # 生成PNG格式的信赖域演化?            visualizer.plot_trust_region_evolution_only(

                optimizer,

                save_path="results/trust_region_evolution.png",

                figsize=(14, 10),

                dpi=300,

                show_plot=False,

                format='png'

            )

            # 生成PDF格式的信赖域演化图（适合论文发表?            visualizer.plot_trust_region_evolution_only(

                optimizer,

                save_path="results/trust_region_evolution.pdf",

                figsize=(12, 8),

                dpi=300,

                show_plot=False,

                format='pdf'

            )

            # 生成信赖?柔度对比?            visualizer.plot_trust_region_evolution_with_compliance(

                optimizer,

                save_path="results/trust_region_compliance_comparison.png",

                figsize=(16, 12),

                dpi=300,

                show_plot=False,

                format='png'

            )

            print("\n" + "=" * 80)

            print("ALL TASKS COMPLETED SUCCESSFULLY!")

            print("=" * 80)

            return optimizer

        else:

            print("\n" + "!" * 80)

            print("SCP OPTIMIZATION FAILED!")

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

    optimizer = run_scp_optimization()

    if optimizer is not None:

        print("\n[SUCCESS] Optimization completed successfully!")

        print(f"Final structure has {np.sum(optimizer.final_areas > optimizer.removal_threshold)} effective members")

    else:

        print("\n[ERROR] Optimization failed. Please check the error messages above.")

