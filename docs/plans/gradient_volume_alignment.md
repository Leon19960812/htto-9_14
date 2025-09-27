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
