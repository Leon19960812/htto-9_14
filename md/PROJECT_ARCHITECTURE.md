# 项目架构（更新）

> 2025-09 代码重写路线（摘要）

- 决策：对核心模块进行最小可编译重写（rewrite），以彻底消除历史编码/断行/缩进破损带来的结构性问题，并与“一维 θ + PolarGeometry”的目标保持一致。
- 范围：`scp_optimizer.py`、`algorithm_modules.py`、`truss_system_initializer.py` 三个文件重写为干净骨架；逐步回填功能；其他模块按需适配。
- 基础设施：全仓库统一 UTF-8 + LF（.editorconfig/.gitattributes 已生效）；提供 `tools/verify_encoding.py` 与（可选）pre-commit 钩子以防再发。

## 重写里程碑（Roadmap）

1) 基线骨架（Day 1）
- `scp_optimizer.py`：保留初始化参数与主流程框架（日志、迭代计数、信赖域跟踪），去除破损段；可先用 no-op/占位求解，保证可编译+可运行。
- `algorithm_modules.py`：
  - 数据类：`TrustRegionParams`、`GeometryParams`、`OptimizationParams`（干净字段，每行一项）。
  - 组件：`TrustRegionManager`、`ConvergenceChecker`、`InitializationManager`、`SystemCalculator`、`StepQualityEvaluator`、`SubproblemSolver`（最小实现）。
  - 约束：仅保留“边界盒约束 + L2 信赖域 + 逐点步长帽”。
- `truss_system_initializer.py`：保留数据类（Geometry/Material/Load/Constraint），几何生成弱化（将逐步由 `polar_geometry.py` 接管）。

2) 极坐标接入（Day 2）
- 所有坐标与长度统一由 `polar_geometry.py` 提供：`update_from_optimization(theta)`、`get_cartesian_coordinates()` 等。
- 载荷向量组装适配极坐标节点，保留原装配接口兼容。

3) 子问题与信赖域（Day 2）
- `SubproblemSolver`：构建凸子问题（cvxpy）；目标与约束按最小集成；优先使用 MOSEK，可回退 ECOS/SCS。
- 步长质量评估 + 信赖域半径更新；记录 `alpha_history`、`trust_radius_history`。

4) AASI 屈曲下界（Day 3）
- 在 Phase C 可选启用下界计算与体积可行性缩放；失败不阻断主流程（降级继续）。

5) 收尾与配套（Day 3）
- 可视化与日志对齐；移除 layered（outer/middle/inner）残留路径；精简注释与文档。

## 验收标准（每阶段）
- `py_compile` 全部通过；`run_scp.py` 可运行并输出迭代/日志（即便是占位结果）。
- 约束与变量维度一致，坐标来源统一自 `polar_geometry.py`。
- 具备最小的“边界 + 信赖域 + 步长帽 + 体积约束”子问题能力。
- 针对数值与可用性的降级路径明确（求解器回退、AASI 可选）。

## 预提交保护（可选）
- 将 `tools/verify_encoding.py` 挂为 pre-commit：提交前检查/转换到 UTF-8（LF）。
- 编辑器建议：VS Code 设置 `files.encoding=utf8`、`files.eol=\n`、`python.analysis.typeCheckingMode=basic`。

## 概览
- 目标：基于序列凸优化（SCP）的桁架几何+拓扑联合优化。
- 几何表示：统一极坐标（r, θ），theta 为一维数组（仅包含自由节点）；固定支撑恒定。
- ground structure：由 `Sequential_Convex_Programming/polar_geometry.py` 生成并维护。

## 目录结构（相关）
```
htto-9_8_version/
├── Sequential_Convex_Programming/
│  ├── scp_optimizer.py                 # 优化器主控
│  ├── algorithm_modules.py             # 约束/子问题/系统计算
│  ├── polar_geometry.py                # 极坐标几何与 ground structure（目标）
│  ├── truss_system_initializer.py      # 旧几何初始化（仅兼容期保留）
│  ├── load_calculator_with_shell.py    # 载荷计算
│  ├── node_merger.py                   # 节点融合
│  └── visualization.py                 # 可视化
└── sdp_truss_optimizer_fixed.py        # SDP 对照（固定几何）
```

## 关键组件
- `polar_geometry.py`
  - 核心数据源：节点（r, θ, node_type, is_fixed）、连接（elements）、DOF 分配。
  - 接口：`get_optimization_variables()`、`update_from_optimization(theta)`、`get_cartesian_coordinates()`、导出 `outer` 节点索引。
- `algorithm_modules.py`
  - ConstraintBuilder：仅保留边界（boundary_buffer）、L2 信赖域、逐点步长帽；默认不使用单调性与分层约束。
  - SystemCalculator/GradientCalculator/SubproblemSolver：统一接收一维 θ，坐标来自 `PolarGeometry`。
- `scp_optimizer.py`
  - 集成 `PolarGeometry` 为几何真源；逐步弃用 `TrussSystemInitializer` 的几何生成。
- `node_merger.py`
  - 一维 θ + `node_id` 映射；融合后重建变量与 DOF；支撑优先保留代表
- `visualization.py`
  - 输出最终结构、载荷分布，并新增 `compliance_evolution.png` 展示柔度演化曲线

## 需要修改的文件（按顺序）
1) 三大核心重写：`scp_optimizer.py`、`algorithm_modules.py`、`truss_system_initializer.py`
2) 接入 `polar_geometry.py` 作为几何唯一真源
3) `node_merger.py`：删除分层 θ 残留，适配一维 θ + node 映射
4) 删除/精简旧的 layered 逻辑（按需）

## 约束策略
- 边界：逐点 `boundary_buffer ≤ θ ≤ π - boundary_buffer`
- 信赖域：`||θ − θ_k||₂ ≤ r` + 逐点步长帽 `|θ_i − θ_{k,i}| ≤ cap_i`
- 可选对称：`--enforce-symmetry` 启用镜像节点等式约束；载荷/支撑不对称时自动降级
- 可选兜底：静态“邻接对 min_spacing”约束（默认关闭）

## 现状
- 文档已统一到“一维 theta + PolarGeometry”的路线。
- 下一步：在优化器接入 `PolarGeometry` 与约束极简化。
