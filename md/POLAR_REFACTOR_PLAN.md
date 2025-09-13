# 极坐标统一重构计划（更新版）

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

## 迁移步骤与里程碑
1) 文档与清理（本次）：
   - 移除多层 theta 相关文档与提法；明确“统一一维 theta + PolarGeometry”目标。
2) 最小集成：
   - 在优化器中引入 `PolarGeometry`（适配现有 `GeometryData` 字段），坐标更新改走 `PolarGeometry`；不改动装配与载荷接口。
   - ConstraintBuilder 改为“边界 + 信赖域 + 逐点步长帽”。
3) 收尾：
   - 删除 layered 残留引用与分支；弃用 `TrussSystemInitializer` 的 ground structure 生成函数。

## 风险与回退
- 过短单元/病态刚度：若出现，可临时收紧信赖域或开启“静态邻接对最小间距”。
- shapely 可用性：如环境受限，提供基于半径窗与角度范围的简化过滤以替代 polygon.contains。

## 当前状态
- 文档统一到一维 theta 方案；多层相关计划已移除。
- 下一步：在优化器接入 `PolarGeometry` 并调整 `ConstraintBuilder`。

