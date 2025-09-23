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

## Area Symmetry Constraints (Implemented)

### Mechanism
- `_prepare_symmetry_constraints` 现在在节点配对成功后额外构建 `node_mirror_map`，用以描述所有节点（含固定节点）的镜像关系。
- 基于该映射扫描 `geometry.elements`，生成 `symmetry_member_pairs` 与 `symmetry_member_fixed`：前者用于镜像构件对，后者是位于对称轴上的自对称构件。
- `SubproblemSolver` 读取 `symmetry_member_pairs`，对每一对构件施加 `A_i == A_j` 线性约束。镜像构件缺失时自动降级并打印“面积对称已停用”提示，不影响角度对称继续生效。

### Notes & Open Items
- 轴上的自对称构件仍保持单独自由度；若未来需要，也可将其面积与相邻构件联动。
- 降级逻辑当前是“缺少镜像 → 全局禁用面积对称”，后续可考虑仅剔除缺对构件，以便最大化保持镜像约束覆盖面。
- 如需支持非对称工况，应通过 CLI 参数或配置允许关闭面积约束，以免影响求解可行性。
