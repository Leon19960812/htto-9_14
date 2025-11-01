# 极坐标统一重构计划（更新版）

> 2025-09 重写执行计划（最终版）

目的：因历史编码/断行导致的结构性损坏，决定对核心模块进行“最小可编译重写”，与一维 θ + PolarGeometry 目标对齐；在可运行基线之上迭代回填功能。

## 执行步骤（Milestones）

1) 基线骨架（Day 1）
- `scp_optimizer.py`：清爽主流程（初始化、迭代、日志、信赖域记录）；允许占位子问题，先跑通。
- `algorithm_modules.py`：数据类与 6 个组件（TR 管理、收敛检查、初始化、系统计算、步长评估、子问题求解）的最小可用实现；约束仅“边界 + L2 TR + 逐点步长帽”。
- `truss_system_initializer.py`：保留数据类；几何生成弱化，逐步迁移到 `polar_geometry.py`。

2) 极坐标接入（Day 2）
- 统一几何：`PolarGeometry` 提供节点、单元、坐标、长度等；优化变量是一维 θ；支撑节点不入变量。
- 载荷装配适配极坐标接口，保留原装配函数签名。

3) 子问题 + 信赖域（Day 2）
- `SubproblemSolver` 使用 cvxpy 构建凸子问题；优先 MOSEK，回退 ECOS/SCS。
- 步长质量评估（ρ）+ 信赖域半径更新；记录 α/半径历史；可选 SPD 守护。

4) AASI 屈曲下界（Day 3）
- Phase C 计算受压杆最小截面下界并体积可行性缩放；失败降级不影响主流程。

5) 收尾（Day 3）
- 可视化/日志统一；移除 layered 残留；简化注释，完善文档。

## 交付与验收
- `py_compile` 通过、`run_scp.py` 跑通；最小子问题可解并输出日志。
- 变量维度、坐标来源与文档一致；有求解器回退与功能降级兜底。

## 风险与对策
- 求解器不可用：回退 SCS/ECOS；或导出问题数据用于离线求解。
- 数值病态：收紧信赖域、启用步长帽、可选最小相邻间距兜底（默认关闭）。

## 工程守护
- `.editorconfig`/`.gitattributes` 统一 UTF-8 + LF；提供 `tools/verify_encoding.py` 做批量转换。
- 可选 pre-commit 钩子：提交前检查并拒绝非 UTF‑8 文本。

## 背景
- 历史代码存在“分层 theta（outer/middle/inner）+ 字典变量”的设计，带来索引映射、支撑节点、融合后重映射等大量复杂性和问题源。
- 新方向：参考 trussopt 的思想，统一以极坐标节点系（r, θ）描述几何，theta 使用“一维数组”统一建模；固定支撑恒定，不参与变量。

## 目标
- 统一一维 theta：变量仅包含“非固定”节点的角度，维护 `theta_index -> node_id` 映射。
- ground structure 来源统一：使用 `polar_geometry.py` 生成并维护节点与连接，弃用旧的分层生成。
- 约束系统极简：
  - 边界：对每个自由节点施加 `boundary_buffer ≤ θ ≤ π - boundary_buffer`（逐点盒约束）。
  - 信赖域：L2 trust region + 逐点步长帽 `neighbor_move_cap`（可调）。
  - 非必需不引入单调性；如需兜底，可用“静态邻接对最小间距”但默认关闭。
- 载荷/边界：支撑节点恒定（不入变量），外层载荷通过 `outer` 类型节点集导出索引以兼容原载荷接口。

## 模块调整
- `polar_geometry.py`（几何唯一真源）：
  - 提供节点坐标、连接、`outer` 节点索引、自由/固定 DOF。
  - 提供 `get_optimization_variables()` 与 `update_from_optimization(theta)`。
- `algorithm_modules.py`
  - ConstraintBuilder：移除单调性与分层逻辑；保留边界、信赖域、逐点步长帽；可选“静态邻接对 min_spacing”（默认关闭）。
  - SystemCalculator/GradientCalculator/SubproblemSolver：统一使用一维 `theta`，坐标来源改为 `PolarGeometry`。
- `scp_optimizer.py`
  - 在初始化阶段注入 `PolarGeometry` 并提供最小适配到现有装配/载荷接口；逐步弃用 `TrussSystemInitializer` 的几何生成。
- `node_merger.py`
  - 改为一维 `theta` + `node_id` 映射工作流；融合后重建变量与 DOF 映射；支撑优先保留为代表节点。

## 迁移总表（对照）
- 旧 layered θ 与字典变量 → 统一一维 θ（仅自由节点），维护 `theta_index -> node_id`。
- 旧几何生成 → 由 `polar_geometry.py` 生成与维护（节点/单元/DOF/outer 索引）。
- 约束：去除单调性/分层残留，仅保留“边界 + L2 信赖域 + 逐点步长帽”；最小相邻间距为可选兜底。

## 风险与回退
- 过短单元/病态刚度：若出现，可临时收紧信赖域或开启“静态邻接对最小间距”。
- shapely 可用性：如环境受限，提供基于半径窗与角度范围的简化过滤以替代 polygon.contains。

## 当前状态
- 文档统一到一维 theta 方案；多层相关计划已移除。
- 下一步：在优化器接入 `PolarGeometry` 并调整 `ConstraintBuilder`。
## Shell Load Calculator Refactor (2025-09)

- Replace `_compute_pressure_loads()` with a mesh-independent pressure integration so total hydrostatic force stays constant when `n_circumferential` or `n_radial` change.
- Rework `solve_with_support_positions()` to recover reactions via DOF partitioning: solve `K_ff u_f = f_f`, then evaluate `r_c = K_cf u_f - f_c` using the nearest boundary node per support.
- After each `node_merge`, rebuild the support mapping from the updated geometry so merged supports share a single constrained DOF and reaction entry.
- Update the shell-aware load calculator to consume the new reaction vector directly (no manual sign flips) and continue projecting forces onto truss load nodes.
- Add regression checks that log the total hydrostatic load and compare it against the summed reactions, warning when the mismatch exceeds a tolerance.
