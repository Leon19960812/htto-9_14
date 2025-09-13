# 项目架构（更新）

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
  - 一维 θ + `node_id` 映射；融合后重建变量与 DOF；支撑优先保留代表。

## 需要修改的文件（按顺序）
1) 文档与清理（本次完成）
2) `scp_optimizer.py`：注入 `PolarGeometry` 并最小适配到现有装配/载荷接口
3) `algorithm_modules.py`：
   - ConstraintBuilder 去除单调性与分层残留
   - 所有坐标更新统一走 `PolarGeometry`
4) `node_merger.py`：删除字典式 θ 路径
5) 移除 `TrussSystemInitializer` 的 ground structure 生成函数（仅保留共用工具如装配）

## 约束策略
- 边界：逐点 `boundary_buffer ≤ θ ≤ π - boundary_buffer`
- 信赖域：`||θ − θ_k||₂ ≤ r` + 逐点步长帽 `|θ_i − θ_{k,i}| ≤ cap_i`
- 可选兜底：静态“邻接对 min_spacing”约束（默认关闭）

## 现状
- 文档已统一到“一维 theta + PolarGeometry”的路线。
- 下一步：在优化器接入 `PolarGeometry` 与约束极简化。
