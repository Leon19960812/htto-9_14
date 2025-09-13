# 极坐标重构开发跟踪（更新）

## 时间线
- 开始日期：2025-01-13
- 当前状态：进行中（文档清理完成，逐步接入极坐标与极简约束）

## 本期目标
- 统一一维 theta：支撑恒定，变量仅包含非固定节点
- 使用 `polar_geometry.py` 作为唯一 ground structure 来源（逐步集成）
- 约束采用“边界 + L2 信赖域 + 逐点步长帽”，默认不启用单调性与分层逻辑

## 已完成
- [x] 文档清理：移除/更新与多层 theta 相关内容
- [x] 统一路线设计文档（PLAN/DEVELOPMENT/ARCHITECTURE）
- [x] 代码：一维 theta 柔度路径统一（去除 layered 坐标更新）
- [x] 代码：子问题内覆盖使用极简约束（边界 + L2 信赖域 + 逐点步长帽 + 面积界）
- [x] 代码：逐点步长帽支持（优先 `theta_move_caps`，无则回退全局 cap）
- [x] 代码：NodeMerger 无层化（删除字典式 theta，统一一维 theta）

## 进行中
- [ ] 在优化器注入 `PolarGeometry`（适配为现有装配/载荷接口字段）
- [ ] 清理 `algorithm_modules.py` 残留的分层判断与无用函数块
- [ ] 引入并使用 `theta_node_ids`（一维 θ 到 node_id 的映射），统一坐标写回、步长帽与融合

## 待办（下一步）
- [ ] 自适应 `theta_move_caps`：基于 incident 最短杆长计算（已初步接入，继续稳健化）
- [ ] 坐标更新统一走 `PolarGeometry.update_from_optimization` / `get_cartesian_coordinates`
- [ ] `node_merger.py` 收敛到一维 theta + node_id 映射（删除字典式路径）
- [ ] 渐进弃用 `TrussSystemInitializer` 的 ground structure 生成

## 风险与对策
- 病态刚度：收紧 trust region 或启用“静态邻接对 min_spacing（可选）”
- shapely 依赖：不可用时使用半径窗 + 角度过滤的简化几何包含策略

## 备注
- 不再引入“层”的运行时概念；所有约束与数据结构均以一维 theta 为中心
# 极坐标重构开发跟踪（更新）

> TODO NOW (open this file after restart)
> 1) Wire theta_node_ids end-to-end (Batch 1)
>    - scp_optimizer.py: after node merge, rebuild and save `self.theta_node_ids` to reflect new node ids.
>    - Ensure per-node caps loop uses `theta_node_ids` only (no outer_nodes slicing).
>    - Pass mapping where needed (NodeMerger currently derives optimized ids internally; keep consistent or return mapping).
> 2) Switch load nodes (Batch 2)
>    - Provide `geometry.load_nodes` in initializer (temp: equal to previous outer_nodes set).
>    - Replace load_calc.compute_load_vector(..., outer_nodes, ...) with load_nodes.
> 3) Text cleanup (Batch 3)
>    - Remove leftover layered comments/markers ("removed layered branch") and extra blank lines in algorithm_modules.py and scp_optimizer.py.
> 4) PolarGeometry integration (Batch 4)
>    - Inject PolarGeometry; adapt nodes/elements/load_nodes/fixed_dofs/free_dofs; keep assembly/load interfaces.
> 5) Optional: linearized min member length constraint (off by default).
