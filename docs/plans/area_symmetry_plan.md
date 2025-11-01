# 面积对称友好的节点合并方案

## 背景
- 当前 NodeMerger 仅依据边长阈值做聚类，不考虑 `node_mirror_map`。
- 合并后几何可能失去镜像关系，导致 `_build_member_symmetry_pairs` 找不到镜像构件，面积对称被禁用。

## 目标
1. 在节点合并过程中保持镜像对称，避免单侧删除节点或构件。
2. 出现异常时给出可观察的日志与选择（跳过/补齐），而不是直接关闭面积对称。
3. 尽量不改变现有优化流程，参数仍可配置。

## 技术方案概述
1. **对称感知的候选生成**
   - 复用 `_prepare_symmetry_constraints` 产生的 `node_mirror_map` 和镜像构件信息。
   - 在 `find_merge_groups` 得到候选后，新增配对逻辑：对每个 group，映射到镜像节点，组装成成对 group；若镜像元素缺失，根据策略跳过或补齐。
2. **同步执行合并**
   - `merge_node_groups` 接受成对 group：左右同时合并，更新 `geometry.nodes` 与 `elements` 时保持一致。
   - 若镜像侧无实际更新，打印提示日志，方便调试。
3. **构件镜像修复**
   - 合并后调用 `_enforce_member_symmetry()`，遍历所有构件 `(i1,i2)`，通过 `node_mirror_map` 检查 `(m1,m2)` 是否存在；缺失时按策略补齐或删除。
4. **测试验证支持**
   - 构建小型对称案例验证逻辑，确认面积对称不会被禁用。
   - 在现有场景中运行，观察 `log_scp.txt` 与导出 CSV，确保几何与面积保持镜像。

## 实施步骤
1. **准备阶段**
   - 在 `_prepare_symmetry_constraints` 完成后，将 `node_mirror_map` 缓存在优化器对象中，供 NodeMerger 使用。
   - 定义描述左右组的新数据结构（如 `SymmetricMergeGroup`）。
2. **实现对称配对**
   - 在调用 `merge_node_groups` 前增加 `pair_merge_groups_by_symmetry()`：
     - 输入：`merge_groups`, `node_mirror_map`, `theta_node_ids`。
     - 输出：成对 group 列表与异常 group 列表。
   - 对异常 group 记录日志，并根据配置选择“跳过”或“补齐”。
3. **调整 NodeMerger**
   - 扩展 `merge_node_groups` 支持左右组同步操作：统一更新 `theta`、`A`、`geometry`，保证索引顺序镜像。
4. **构件镜像修复**
   - 新增 `_enforce_member_symmetry()`，遍历元素列表并对缺失镜像的构件执行补齐或删除。
5. **测试与验证**
   - 编写针对配对逻辑与镜像修复的单元测试。
   - 使用现有日志案例回放，确认面积对称始终启用且导出的 `theta/area` 数据保持对称。

## 后续工作
- 评估镜像修复对数值稳定性的影响，必要时在修复策略中加入最小面积阈值。
- 将关键开关（阈值、补齐策略）暴露在配置层，便于后续调参与 A/B 实验。
