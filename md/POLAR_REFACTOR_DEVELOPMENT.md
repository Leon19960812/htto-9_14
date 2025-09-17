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
- [x] 在优化器注入 `PolarGeometry`（由 `TrussSystemInitializer` 构建并挂载）
- [ ] 清理 `algorithm_modules.py` 残留的分层判断与无用函数块
- [ ] 引入并使用 `theta_node_ids`（一维 θ 到 node_id 的映射），统一坐标写回、步长帽与融合

## 待办（下一步）
- [ ] 自适应 `theta_move_caps`：基于 incident 最短杆长计算（已初步接入，继续稳健化）
- [x] 坐标更新统一走 `PolarGeometry.update_from_optimization` / `get_cartesian_coordinates`（通过 `update_from_partial_optimization` 适配 `load_nodes` 顺序）
- [ ] `node_merger.py` 收敛到一维 theta + node_id 映射（删除字典式路径）
- [ ] 渐进弃用 `TrussSystemInitializer` 的 ground structure 生成

## 风险与对策
- 病态刚度：收紧 trust region 或启用“静态邻接对 min_spacing（可选）”
- shapely 依赖：不可用时使用半径窗 + 角度过滤的简化几何包含策略

## 备注
- 不再引入“层”的运行时概念；所有约束与数据结构均以一维 theta 为中心

## 进度更新（2025-09-14）
- 已完成
  - 强制使用 MOSEK（cvxpy），移除 ECOS/SCS 回退；失败直接报错。
  - `polar_geometry.py`：严格使用 shapely + gcd 生成 ground structure，删除简化几何包含回退。
  - `truss_system_initializer.py`：重写为最小骨架，几何来源统一 `PolarGeometry`；暴露 `polar_geometry` 给优化器。
  - 优化器已通过适配层切换到 `PolarGeometry` 坐标更新：
    - 使用 `polar_geometry.update_from_partial_optimization(theta, geometry.load_nodes)`
    - 然后 `polar_geometry.get_cartesian_coordinates()` 作为装配坐标。

- 待办/风险
  - `theta_node_ids`：需要对齐优化器的 θ 顺序与 `PolarGeometry` 的非固定节点顺序，完成后可以改为 `update_from_optimization` 全量更新。
  - 融合后需重建 `theta_node_ids` 并刷新 `load_nodes/DOF/element_lengths/单位刚度核`；本期将在节点融合回合后触发。

## 可视化与输出（问题与计划，2025-09-14）

已发现的问题
- 图像冗余与信息重复：optimized/cleaned 两张图差异小，不能直观展示“最终 A 与 θ”。
- 支撑节点绘制与计算不一致：历史上以 `load_nodes` 两端作为支撑，融合/排序变化后会导致“支撑”发生位移的错觉。
- 载荷点可见性：`load_nodes[0]` 在个别结果中未显示，可能是节点融合/索引重建后与绘制数据不同步，或标记过小/被遮挡。
- Ground Structure 图语义偏差：当前图在优化后生成，未明确代表“优化前的基线”。

解决计划（Batch VIZ-1）
1) 合并结果图（final_structure.png）
   - 用最终 θ 更新坐标；按 A 比例映射线宽（增大动态范围）；
   - load 点（外圈）红圆，支撑点（内圈两端）蓝三角；
   - 提供 `hide_below` 选项（默认关闭）以隐藏极小杆；
   - 移除 optimized/cleaned 两张图，避免信息重复。
2) Ground Structure 基线图（pre_optimization）
   - 优化前在初始化完成后立即保存基线（或在 CLI 中先绘一次）；
   - 明确标注为“未优化前”的结构模板（uniform 线宽）。
3) 支撑节点不动性（可视化与计算一致）
   - 计算侧已改为“内圈两端优先”；
   - 可视化侧统一用几何法识别内圈两端；
   - 节点融合时，若融合组包含支撑节点，则以“支撑节点”为代表，其他节点并入，不改变支撑坐标（NodeMerger 规则化）。
4) 载荷点可见性
   - load_nodes 合并/索引重建后，保证绘制使用的 `nodes_array` 与 `geometry.load_nodes` 同步；
   - 增大红点尺寸与 zorder，并在保存图前打印首末载荷点 id/坐标用于核对。
5) ρ 在回溯时的预测
   - 对 α<1 的候选，用线性缩放的近似 `t(α)` 作为预测，避免 ρ=1 的退化（保持 SDP 不重解）。

落地步骤（执行顺序）
1) VIZ-1A：新增 `final_structure.png` 生成逻辑，调整线宽/标记尺寸，合并图；
2) VIZ-1B：在 CLI `--save-figs` 下，优化前保存 `ground_structure.png`，优化后保存 `final_structure.png`；
3) NM-Rule：在 `node_merger.py` 中规则化“包含支撑节点的融合组：以支撑为代表”；
4) LOAD-CHK：在可视化保存时打印载荷点数量与首末坐标；
5) RHO-BT：在 α 回溯环节加入基于线性化的 `t(α)` 近似，完善 ρ 预测；
6) 验证：以 8 或 10 扇形、10–20 步迭代生成一套图，确认可读性与一致性。

## 现状记录（2025-09-14 晚）

状态
- PolarGeometry 作为“唯一几何真源”路线已落实：节点融合后调用 `PolarGeometry.rebuild_from_geometry(geometry)` 同步。
- NodeMerger 已加入支撑融合规则：若融合组包含支撑（内圈两端），以支撑为代表且保持坐标不动。
- 生成图已合并为单图 `final_structure.png`（最终 θ + A），并在优化前另存 `ground_structure.png`（基线）。

仍存在的问题
- 越界连杆：final_structure 中仍能看到不符合环域（polygon.contains/boundary.contains）的连杆。推断为 θ 更新/融合后未重新对 connections 做环域过滤。
- 载荷方向/幅值异常：静水压力经壳体传导至外圈 `load_nodes` 的支撑反力，转化为桁架载荷时方向/幅值不稳定（有外向箭头、大小不准确）。推测与“先投影取正再定向”的实现有关，且与最新坐标/索引同步有关。
- 集合与索引重建不足：融合后 `inner_nodes/outer_nodes` 未在 GeometryData 中持久重建；`theta_node_ids` 与 `load_nodes` 的角色需要严格区分并同步。

下次执行计划（Batch GS+LOAD-1）
- GS-FILTER（环域过滤）
  - 在 PolarGeometry 增加并调用：FilterConnectionsByRingPolygon()（基于当前内/外圈节点构造 ring polygon）；在 θ 更新与融合同步后执行；
  - 过滤 self.connections → 重建 self.elements（长度），随后再同步 GeometryData。
- LOAD-FIX（载荷矢量）
  - 以壳体支撑反力 R_shell 为基线，桁架等效载荷设为 F_truss = -R_shell；
  - 可选“仅取径向内分量”开关：F_truss = -max(0, R_shell·n_in)·n_in（默认关闭或保留比例）；
  - 保存图时打印载荷点首/末 id 与坐标/向量，核对方向与幅值。
- SETS-REBUILD（集合/索引重建）
  - 融合后在 GeometryData 重建 inner_nodes/outer_nodes（按半径阈值+极角排序）、load_nodes；同步回 PolarGeometry 后从 PG 派生 Geometry；
  - 变量集 `theta_node_ids` 与 `load_nodes` 严格分离，保证 θ 分量不包含支撑节点并对齐 PG 的非固定顺序。
- 保持“单一真源”
  - 所有坐标/线性化/绘图均从同步后的 PolarGeometry 出发；GeometryData 仅作 FE/载荷适配的快照。

复现实验（当前）
- 根目录运行：
  - `python run_scp.py --radius 2.0 --n-sectors 12 --inner-ratio 0.6 --depth 10 --volume-fraction 0.2 --max-iterations 10 --save-figs results/paper/scp_run`
- 观察：
  - `ground_structure.png` 为优化前基线；
  - `final_structure.png` 中可见少量越界连杆；
  - `load_distribution.png` 中个别载荷箭头向外/大小不准；
  - 终端打印包含载荷点首/末 id 与坐标，辅助定位 load_nodes 与坐标同步是否一致。

验收标准（下次）
- 过滤后，连杆全部位于 ring polygon 内（或边界上），越界数=0；
- 载荷方向统一向径向内（或按矢量设定一致），幅值与壳体反力一致或按投影/缩放规则一致；
- 融合后集合/索引一致：PG 与 GeometryData 在 nodes/elements/load_nodes/inner/outer 上 1:1 对应；
- 最终仅输出 `final_structure.png`（清晰表达 θ 与 A）、`ground_structure.png`（优化前基线），其余辅助图按需生成。

## Ground Structure 连接规则（无层，2025-09-15）

为避免在运行期引入“层”的概念，同时获得更合理且稀疏的连接网络，已将 `polar_geometry.py` 的连接生成改为以下无层规则：

- 角差阈值窗口（周向稀疏）
  - 将全部节点按极角 θ 升序排序。
  - 仅保留角差 Δθ(i,j) ≤ K·Δθ_avg 的连接（默认 K=3，Δθ_avg 为去重后的平均角步长），其余长跨向弦不生成；与环数无关。

- 径向家族去冗余（保留相邻半径）
  - 近似同一射线内（|Δθ| ≤ 0.5·平均角步长）的节点归为一个“径向家族”。
  - 在每个家族内按半径升序，仅保留相邻半径之间的连接；删除跨越中间半径的“长径向”连接（等价于 gcd 去冗余，但无整数网格假设且不引入层标签）。

- 域内性判定（包含边界）
  - 使用 `ring_polygon.buffer(ε).covers(seg)` 过滤（ε ≈ 1e-7·R_max），吸收贴边/沿边的数值误差；越界段不会通过。

备注与影响：
- 无层标签：节点融合/θ 更新后规则自动适配，无需同步“层”。
- 密度显著下降且连通性良好：K=3 通常即可形成局部三角网；若出现空洞可临时将 K 调至 4 验证。
- 与“gcd 思想”的一致性：径向家族的“只连相邻半径”本质上就是 gcd=1 的去冗余，但以几何容差表达，避免浮点量化误删。

## 子问题（联合线性化）可行性与数值策略（2025-09-15）

- 联合线性化：与论文一致，子问题在当前设计点 `(A^k, θ^k)` 处对 K 与 f 同时做一阶泰勒展开：
  - `K_lin = Σ A_i K_i(θ^k) + Σ (θ_j-θ_j^k) Kθ_j(A^k, θ^k)`（实现中核按 1/L 规范化、对称化）
  - `f_lin = f(θ^k) + Σ (θ_j-θ_j^k) fθ_j(θ^k)`（实现中对 f 乘 1/√E 缩放以改善数值）
- 变量集合扩展后的可行性：当前实现保持“全 θ 联合线性化”模式，不再引入 `J_f/J_o` 分组与 γI 兜底；后续若需稳定化将另行记录。

2025-09-16 更新
- 单步/迭代子问题恢复“全 θ 线性化”，去除 `J_f/J_o` 分组与 γI 补偿；文档保持一致。
- 有限差分计算 `Kθ_j`、`fθ_j` 时，每次扰动后立即回滚基线 θ，避免 PolarGeometry 状态漂移。
- 观察点：`--single-subproblem` 再运行时柔度应恢复下降，周向杆件不再被异常削除。
- 几何约束整体化：
  - 全局角序（对 PolarGeometry 的全体非支撑自由节点按角度升序），不分环；
  - 最小角间距仅对“近似同环”（半径差 ≤ 1e−6·R_max）的相邻对生效，跨环仅要求非递减，避免多环同角导致不可行；
  - 两端 `boundary_buffer` 和 L2 信赖域、单点帽与论文一致。
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
## 对称约束集成计划（2025-09-16）

- 背景：当前子问题仅施加边界/信赖域等约束，需要为半圆结构引入轴对称性。
1. 在 `PolarGeometry` 或初始化阶段梳理对称轴（默认 y 轴），生成每个环的镜像节点对，过滤掉支撑及轴线上单点，将结果缓存到优化器。
2. 在 `_initialize_optimization_variables` 之后，将节点对映射到 `theta` 变量索引，写入 `self.symmetry_pairs`，并对奇数节点的中心点施加 `theta=π/2` 固定策略。
3. 在 `SubproblemSolver.solve_linearized_subproblem` 中读取 `self.symmetry_pairs`，对每一对 `(i,j)` 添加线性等式约束 `theta[i] + theta[j] == π`（数值上使用 `float(np.pi)`），确保对称节点角度同时更新。
4. 提供布尔开关（CLI `--enforce-symmetry` -> 优化器 `enable_symmetry`），默认关闭；当检测到载荷或支撑不对称时给出告警并跳过约束。
5. 调整日志/导出：在 step details 中记录对称配对数，在可视化前验证 `x` 坐标镜像误差，超限则提示回滚或放宽信赖域。
6. 验证路径：以 `--simple-loads` 场景运行 2~3 步迭代，确认 `theta` 和最终结构满足镜像；再对人为扰动的载荷测试自动降级分支。
7. 载荷保持原有计算逻辑：壳体/简单模式均按当前几何逐节点给出径向力值，不做强制对称化；仅在节点集合不成镜像时禁用对称约束。

8. 新增自动导出的 compliance_evolution.png，基于 compliance_history 展示接受步柔度的变化，可用于论文附图。

## 节点融合策略讨论（2025-09-17）
- 参考文献《Truss Geometry and Topology Optimization with Global Stability Constraints》，计划将节点融合阈值设为相连杆件的欧氏长度 < 0.25 m。
- 判据从现有的“投影到固定半径”改为使用真实几何：利用节点间实际杆件长度 L_min 触发融合，而非环/索引推断。
- 合并对象覆盖所有节点：若组内包含支撑节点，直接保留支撑节点作为代表，以确保支撑位置不移动。
- 阶段划分将调整：节点融合应在 Phase A 即可生效，避免因密集节点导致的数值退化阻止进入 Phase B。
- 下一步将重构 `NodeMerger` 与调用路径，使其基于 `theta_node_ids` + 实际坐标执行，并允许在融合前做 dry-run 检查。

### 改造计划
1. 重构 `NodeMerger` 接口：以 `theta_node_ids` + 实际坐标为输入，支持全体节点并保持支撑节点为代表；新增 0.25 m 阈值配置。
2. 在初始化阶段构建节点 L_min：基于当前杆件长度计算每个节点的最短相连杆件，为融合判据提供输入。
3. 调整优化流程：在每次接受步后（Phase A 起生效）运行融合检测，dry-run 输出候选合并组，再在确认逻辑通过后执行合并并刷新几何/约束。
4. 记录和验证：融合后重新导出 `theta_history`/`area_history`，并加入 sanity check（节点数、DOF、支撑节点位置）。
