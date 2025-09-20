# Node Merge & Symmetry Constraint Flow

## Baseline (Before Any Merge)
- `SequentialConvexTrussOptimizer._initialize_optimization_variables` creates the optimizer's working node list:
  - Builds `theta_all = arctan2(y, x)` for every geometry node.
  - Drops nodes whose DOFs are fixed (`geometry.fixed_dofs`).
  - Sorts the remaining node ids by ascending `theta`; the result is stored in `self.theta_node_ids`.
  - The associated ordered `theta` values become the first `theta_k` vector.
- `self.polar_geometry` is instantiated from `PolarGeometry` (outer ⇒ middle ⇒ inner ring order).  Each `PolarNode.id` equals its positional index in `geometry.nodes` at creation.
- `_prepare_symmetry_constraints(theta_ids)` runs once at the end of initialization:
  - Collects the free-node ids (`theta_ids`) and looks up the matching polar nodes via `self.polar_geometry.nodes`.
  - Groups nodes by `(node_type, rounded radius)` and requires mirror angles about π/2.
  - Stores symmetry relations as pairs of indices **inside the θ-vector order** (`self.symmetry_pairs`) and center-locked indices (`self.symmetry_fixed_indices`).

## Merge Detection (`find_merge_groups`)
- Executed through `TrussSystemInitializer.find_merge_groups`, which instantiates a `NodeMerger` on the current `geometry` snapshot.
- `NodeMerger.find_merge_groups`:
  - Runs a DSU over `geometry.elements`, uniting endpoints whose Euclidean chord `(length < merge_threshold)`.
  - Chooses the representative of each cluster using priority: support node ⇒ node already in `theta_ids` ⇒ smallest index.
  - Returns ordered groups `[target, member1, member2, ...]` in terms of **current geometry node ids**.

## Merge Execution (`merge_node_groups`)
- Called from the optimizer with inputs at iteration `k`: `(theta_k, theta_ids_current, A_k, merge_groups)`.
- Inside `NodeMerger.merge_node_groups`:
  1. Collects `remove_nodes` and builds `target_coords` for each representative:
     - Support representatives keep their original coordinates.
     - Others move to a weighted centroid based on incident member areas (fallback to arithmetic mean).
  2. Iterates original node ids in ascending order to build `new_nodes`:
     - Skips any id in `remove_nodes`.
     - Keeps survivors in order, writing the new coordinate if the node was a group target.
     - Records a monotone mapping `old_to_new` such that remaining nodes keep relative order while indices are compacted.
  3. Reconstructs `geometry.elements`:
     - Maps each old `(n1, n2)` via `old_to_new`.
     - Drops degenerate edges and consolidates duplicate edges by summing their cross-sectional areas.
  4. Calls `_remap_geometry_sets` to rewrite `load_nodes`, `inner_nodes`, `outer_nodes`, `middle_nodes`, `support_nodes` using `old_to_new` (duplicates removed, order preserved).
  5. Recomputes boundary conditions from the constraint calculator, storing fresh `fixed_dofs/free_dofs` on the geometry.
  6. Updates the θ bookkeeping:
     - Maps each entry in the previous `theta_ids` through `old_to_new`; vanished nodes are dropped.
     - For the surviving ids, recomputes `theta_vals = arctan2(y, x)` using the **new coordinates**.
     - Sorts by those angles to deliver `theta_ids_updated` (ascending angle) and `theta_updated` (matching order).
  7. Packs everything in a `MergeResult`, keeping track of the raw `(target, member)` pairs and removed member ids.

## Optimizer Side Effects After a Merge
- `self.initializer.merge_node_groups` forwards to `NodeMerger` and mirrors the geometry-wide changes into the initializer (nodes, elements, DOFs, lengths, stiffness matrices).
- Back in `SequentialConvexTrussOptimizer` (main loop):
  1. Receives `merge_info`.
  2. If `structure_modified`:
     - Replaces `theta_k` and `A_k` with the arrays from `merge_info`.
     - Sets `self.theta_node_ids = theta_ids_new` (already sorted by new polar angle).
     - Flags `symmetry_refresh_needed = self.enable_symmetry`.
     - Copies the merged geometry snapshot onto `self.geometry`, `self.nodes`, `self.elements`, etc., ensuring all cached references use the compacted indices.
  3. Rebuilds secondary data (load calculator, stiffness caches, compliance baseline).
  4. Calls `self.polar_geometry.rebuild_from_geometry(self.geometry)`:
     - Creates fresh `PolarNode` objects with ids `0..n_nodes-1`, computing `radius` and `theta` straight from the merged Cartesian coordinates.
     - Reapplies support metadata via `set_support_nodes`, so the polar model now mirrors the merged indexing.
  5. If symmetry had been enabled, `_prepare_symmetry_constraints(self.theta_node_ids)` executes **after** the rebuild.  This recomputes pairings using the remapped ids and new polar angles.

## Symmetry Constraint Lifecycle
- **Before merge**: pairs and fixed indices are aligned with the θ-vector built from the original geometry.
- **After merge**:
  - `theta_ids` shrink to the survivors and are resorted by the new geometric angles.
  - `polar_geometry` rebuilt from the compacted node list means node ids, node types, and radii reflect the merged configuration.
  - `_prepare_symmetry_constraints` operates on these post-merge objects, so any mismatch must stem from downstream usage (e.g., pairing assumptions inconsistent with merged topology) rather than stale indices.

## Observed Mismatch (Needs Follow-up)
- Even though the bookkeeping refreshes as described above, the exported “SCP Final Structure” still shows asymmetric free-node placement after merges.
- That indicates either:
  1. The geometric merge itself is producing non-mirror coordinates (e.g., weighted centroid biased), or
  2. The symmetry constraint pairs are computed but not enforced in subsequent iterations (e.g., solver ignores them, or indices do not match θ decision variables fed into the SDP).
- Further diagnosis will need to trace how `self.symmetry_pairs` is translated into actual constraint matrices inside the subproblem assembly (beyond the scope of this note).

## Why Symmetry Gets Disabled After Node Merge
- `_prepare_symmetry_constraints` re-validates mirror conditions for key node sets every time it runs.  The first failure encountered (load nodes or support nodes) switches `self.enable_symmetry` to `False` and stops adding any symmetry pairs.
- After the merge at iteration 32 the log prints “载荷节点不满足镜像，对称约束已停用。” meaning the remapped `geometry.load_nodes` list lost its left/right pairing.
  - `NodeMerger._remap_geometry_sets` keeps only surviving node ids but does not guarantee that both sides of a symmetric pair remain present after a merge.
  - If one load node was absorbed into a support representative (or multiple load nodes collapsed into a single id on one side), the resulting angular list is no longer π-symmetric, so the validator rejects the constraint.
- Once the flag flips off, later iterations never re-enable symmetry (`self.enable_symmetry` stays False), so the SDP subproblem omits the angle-equality constraints even though the rest of the bookkeeping is refreshed.  Subsequent iterations therefore drift away from mirror geometry.

## Next Questions for Follow-up
1. Should merging rules preserve designated load/support endpoints explicitly to keep the mirror pairing intact?  e.g., forbid merging a load node with a non-load on only one side.
2. Alternatively, should `_check_node_set_symmetry` accept slightly unbalanced sets post-merge (e.g., tolerate missing pairs) but enforce symmetry on the remaining matched nodes?
3. If constraint enforcement is desired even after imbalance, how should `symmetry_pairs` be recomputed to target whatever nodes still have a viable mirror counterpart while ignoring the rest?

## TODO: Extend Symmetry to Member Areas

### Current Situation
- θ 变量的对称约束已经能够在合并之后保持几何镜像，但 `algorithm_modules.py` 中的面积变量只受上下界与总体体积约束约束，没有镜像绑定。
- 结果是最终结构在几何上对称，但同一对镜像构件的 `A_i`、`A_j` 可能不同，视觉上线宽不匹配，同时左右半跨的刚度也会轻微偏斜。

### 可行性评估
- 几何层的对称重新构建已经提供了节点对信息；通过扩展 `_prepare_symmetry_constraints` 可以同时返回“节点→镜像节点”的映射。
- 以该映射即可在每次刷新后扫描 `self.geometry.elements`：若 `(n1, n2)` 的镜像端点 `(mirror(n1), mirror(n2))` 也构成有效构件，则记录为一组对称构件；若元素本身位于对称轴（端点互为镜像或节点位于轴上），则记为单独的“自对称”构件。
- 上述映射逻辑与节点重建流程一致，节点融合后再次调用 `_prepare_symmetry_constraints` 时亦会更新，因而不会破坏现有的合并流程。
- 需要注意的边界：
  - 若某个构件在镜像一侧不存在（拓扑差异），应跳过该约束以免将问题判定为不可行；
  - 轴上构件不需要额外约束，只需保留其面积自由度。

### 预计修改步骤
1. **扩展 `_prepare_symmetry_constraints`**：
   - 生成并缓存 `self.node_mirror_map`，键为节点 id，值为镜像节点 id；轴线上节点映射到自身。
   - 基于此映射构建 `self.symmetry_member_pairs`（面积成对索引）及 `self.symmetry_member_fixed`（轴上构件索引）。
2. **在 `SubproblemSolver` 中施加面积约束**：
   - 读取 `self.opt.symmetry_member_pairs`，对每对 `(i, j)` 添加线性等式 `A[i] == A[j]`。
   - 对 `self.opt.symmetry_member_fixed` 可选择不额外约束（原约束已足够），仅记录便于调试。
3. **节点融合后刷新**：
   - 维持现有流程：合并完成 → `rebuild_from_geometry` → `_prepare_symmetry_constraints`，新字段会自动更新。
   - 如遇镜像集合不完整（例如构件数量不匹配），在 `_prepare_symmetry_constraints` 内打印诊断并禁用面积对称，以免破坏主流程的鲁棒性。
4. **验证流程**：
   - 在接受步长后输出面积对称对数目，用于观察约束是否生效。
   - 执行一次回归运行，检查最终 `area_history` 中镜像构件是否相等，以及结构图线宽是否对称。

### 后续风险 / 需要决策的点
- 若今后引入非对称载荷或支撑，需要允许“只锁角度，不锁面积”的运行模式；建议通过 CLI 参数控制新约束是否启用。
- 需要确认镜像构件列表在所有阶段都存在 1:1 配对，否则需提供自动降级逻辑（例如略去无法配对的构件）。
