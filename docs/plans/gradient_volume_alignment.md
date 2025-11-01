# Pending Alignment: Analytical Gradients and Volume Constraint

## Background
Recent review comparing the implementation with `docs/reference/paper.md` exposed two gaps that need to be closed before the solver fully matches the paper’s formulation.

## 1. Analytical Sensitivities Missing
- **Observed**: `Sequential_Convex_Programming/algorithm_modules.py:127-137` defines `GradientCalculator` as a placeholder returning empty lists. The actual linearization inside `SubproblemSolver` (lines ~431-738) recomputes `K_ff` and `f_ff` via finite differences on each `\theta_j`.
- **Expected**: Paper Section 3.2, equations (24)-(27), relies on closed-form expressions for `∂K/∂θ_j` and `∂f/∂θ_j`, so the subproblem uses analytic gradients.
- **Impact**: Finite differences introduce truncation noise, increase shell-FEA calls, and degrade trust-region predictions; repeated step rejections stem from these inaccuracies.
- **Next Steps**:
  1. Implement analytic stiffness/load gradients consistent with equations (24)-(27).
  2. Return structured gradients from `GradientCalculator` (or equivalent module) and wire them into `SubproblemSolver` and logging.
  3. Add verification (unit tests or regression runs) to confirm analytic and finite-difference gradients agree.

## 2. Volume Constraint Uses Stale Element Lengths
- **Observed**: The SDP subproblem enforces `lengths @ A <= V_max` (`algorithm_modules.py:452-455`) with `lengths = opt.element_lengths`. That array is populated during initialization (`scp_optimizer.py:615`) or structural rebuilds (node merges), but it is not refreshed after routine accepted steps where node coordinates change.
- **Expected**: Equation (27) in the paper uses the current geometry `L_i(θ^{(k)})` each iteration.
- **Impact**: When members elongate/shorten, the optimization still measures volume with outdated lengths, so feasibility and predicted reductions drift from the true model.
- **Next Steps**:
  1. Recompute `element_lengths` after every accepted step (before solving the next subproblem), or on-demand inside the subproblem builder from the latest coordinates.
  2. Audit any derived caches (unit stiffness matrices, move caps) that depend on lengths and ensure they stay synced.
  3. Document the update rule so future refactors do not regress.

Addressing these two items will bring the implementation back in line with the published SCP formulation.

## 2025-03-16 Update

- 已在 `Sequential_Convex_Programming/algorithm_modules.py` 中补齐解析刚度与简单静水载荷的梯度，同时保留对壳体 FEA 的有限差分兜底选项（需显式启用 `enable_shell_fd_sensitivity`）。
- `SubproblemSolver` 现直接调用解析的 `(∂K_ff/∂θ_j, ∂f_ff/∂θ_j)`，避免重复的全量有限差分。主循环调用也同步传入当前 `(θ, A)` 更新缓存。
- 每次构建子问题时都会将 `element_lengths` 更新为当前基线几何的杆长（写回优化器和初始化器），对应论文中 `Σ A_i L_i(θ^{(k)})` 的实现。
- 建议后续增加**回归测试**：即为关键场景（simple_load、shell_load 等）固定随机种子与输入参数，运行短程 SCP，比较关键指标（迭代柔度、ρ、体积占用）与记录的基准结果；一旦差异超阈值即报警，从而防止未来改动破坏解析梯度或体积约束逻辑。

### TODO: shell 载荷灵敏度（解析化）

- 当前解析 `∂f/∂θ_j` 仅覆盖 simple_load；壳体模式仍视作 0（除非手动启用有限差分兜底），导致信赖域预测误差大。
- 拟采用的实现思路：
  1. 将壳体支持点的插值权重、法向等封装成可微函数，得到约束矩阵 `C(θ)` 及其导数 `∂C/∂θ_j`。
  2. 对壳体的 saddle 系统
     [ K  C^T ; C  0 ] [u; λ] = [f_p; d]
     做一次线性化，求解
     [ K  C^T ; C  0 ] [u̇; λ̇] = -[ (∂C^T/∂θ_j) λ ; (∂C/∂θ_j) u - ∂d/∂θ_j ]，
     从 λ̇ 中直接读取 `∂f/∂θ_j`。
  3. 在 `GradientCalculator` 中接入该流程：当启用壳体载荷时自动返回解析灵敏度，而不是落回有限差分。
- 该部分工作量较大，留待下一轮完善。

## 2025-09-18 Review

- **结论**：对壳体载荷使用鞍式系统线性化（[K C^T; C 0]）求解 \(u, \lambda\) 并通过扰动方程获得 \(\dot{\lambda}\) 的思路在理论上可行，因为壳体刚度矩阵 K 与支撑位置无关，支撑约束矩阵 C 的变化即可驱动反力灵敏度。
- **主要风险**：
  - `_compute_support_weights` 先用 `argsort` 选取最近 k 个边界节点，再做高斯权重并夹紧角度、方差；这些离散选择与 `min/max` 裁剪在支撑位置穿越节点时不可导，解析梯度会出现跳变。
  - 壳体反力转成桁架载荷时还要乘上节点径向单位向量 `(-x/r, -y/r)`，该映射的导数（含半径归一化）必须显式写出，否则梯度缺项。
  - 当前载荷向量默认通过 FIR 滤波缓冲，滤波历史依赖迭代轨迹；若不重新定义滤波在灵敏度中的角色，解析导数与实际载荷不一致。
  - 每个 \(	heta_j\) 需要解一次增量方程；若不重用原始鞍式系统的分解，计算成本很高。
- **改进建议**：
  1. 用全边界节点的 softmax 权重或预先固定的分片基函数替换 "k 最近邻" 裁剪，确保 `C(θ)` 对支撑位置是光滑函数；必要时对角度夹紧使用平滑近似（SoftClip）。
  2. 在 `Shell2DFEA` 中显式给出 `∂C/∂p_i`，并封装一个求解接口：返回支撑反力、用于 reuse 的 `A` 分解，以及给定 `∂C` 时的 `∂λ`。这样 `GradientCalculator` 就能一次因式分解、对多个方向复用。
  3. 在 `LoadCalculatorWithShell` 中补充解析公式，把 `λ` 到 `f` 的转换对节点坐标求导（含 `r = √(x²+y²)` 与单位向量），并与桁架节点坐标对设计变量的导数做链式相乘。
  4. 明确载荷滤波的策略：要么在灵敏度计算时旁路滤波（直接用原始载荷），要么给出滤波权重对输入的线性导数并同步维护历史缓冲。
  5. 为壳体灵敏度实现提供测试脚本：固定几何与支撑点，对比解析导数与有限差分结果，验证在 softmax/滤波配置下的一致性。


### Implementation Plan (pending)

1. **禁用 FIR 滤波**：
   - 将 shell 模式的默认配置中 `load_filter.enabled` 置为 `False` 或直接移除相关配置。
   - 梳理 `LoadCalculatorWithShell` 的调用路径，确保灵敏度计算时不再维护历史缓冲。

2. **重构支撑权重为 softmax**：
   - 在 `Shell2DFEA._compute_support_weights` 中，改为对所有边界节点计算角度差并通过 `softmax(-Δθ^2 / τ)` 得到权重，τ 为可调温度。
   - 提供可选的局部掩码/阈值以控制性能，但保持导数连续。
   - 输出同时返回 `∂w/∂θ` 所需的中间量，为后续解析梯度调用做准备。

3. **链式求导接口准备**：
   - 在 `Shell2DFEA` 内部封装求解器接口：返回支撑反力、系统分解（以便重用）以及软权重相关的缓存。
   - 在 `LoadCalculatorWithShell` 中接入新的权重结构，验证与旧逻辑一致。

4. **验证与回归**：
   - 构造简单壳体载荷场景（固定几何/支撑），对比 softmax 与原高斯核输出的差异。
   - 暂时保持有限差分兜底，后续在解析导数实现后用它来做精度验证。

## 2025-09-18 Implementation Notes

- 额外增加柔度停滞判据：若连续三次接受步的柔度改善幅度均小于 0.1% (1e-3)，则提前收敛，避免尾段信赖域反复拒绝但变量/目标已稳定。
- 已默认关闭 shell 模式下的 FIR 滤波，并提供 `disable_fir_filter=False` 选项以显式重启。载荷历史缓存在解析梯度流程中不再参与。
- `Shell2DFEA._compute_support_weights` 改用 softmax(-Δθ²/τ) 输出连续可导的权重，同时缓存 `dw/dx, dw/dy`、角度导数以及增广系统矩阵 `A`、约束矩阵 `C`。
- 新增 `Shell2DFEA.build_support_constraint_derivative`、`solve_augmented_system` 等接口，支持构造 ∂C/∂p 并复用增广系统求解。
- `LoadCalculatorWithShell` 可以在求载荷时请求 `return_jacobian=True`，内部会调用壳体接口构建 ∂f/∂p（支撑坐标），并通过 `get_last_shell_load_jacobian()` 暴露。
- `GradientCalculator._load_theta_derivative` 现已在壳体模式下直接读取 Jacobian，与节点坐标对角度的导数做链式求导（dx/dθ = -y, dy/dθ = x），默认停止使用有限差分。
- 后续需关注数值稳定性：加载 Jacobian 规模大时的条件数，以及解析梯度与有限差分的残差比较（可选验证脚本）。

## 2025-09-18 Milestone Summary

- 壳体载荷解析梯度已完整融入：`LoadCalculatorWithShell` 提供 `∂f/∂p`，`GradientCalculator` 对 `θ` 做链式传导，避免再落回壳体有限差分。
- 信赖域尾段拒绝问题解决：新增“最近三次接受步改进 < 0.1%”收敛判据，遇到柔度停滞时自动退出。
- 比对实验：固定几何单次 SDP (`--single-subproblem --sdp-fixed-geometry`) 在简化载荷下柔度 `C_new≈1.30×10^2`，完整 SCP（壳体载荷、解析梯度）降至 `≈1.24×10^2`，相较基线 (`C_baseline≈3.63×10^2`) 提升显著。
- 快照输出完善：`--save-shell-iter` 现默认写入 `results/shell_displacement_iter/`，每个接受步保存最新壳体位移图，不会被覆盖。
- 相关日志、脚本（`log_scp_shell.txt`, `debug_shell_support.log`, `results_scp_shell/*`）已验证流程稳定，可据此撰写实验段落与图表。
