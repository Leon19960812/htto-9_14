paper: Truss geometry and topology optimization with global stability constraints

# 1. Node merging (node_merge) criteria and method:
#    Merge two nodes if the distance between them is below a threshold (e.g. ≤ 0.25 units):contentReference[oaicite:0]{index=0}.
#    If either node is a support (fixed) or load (force application) point, keep that node’s position as the merged location:contentReference[oaicite:1]{index=1}.
#    Otherwise (for two free nodes), place the new merged node at the average of the two original node coordinates:contentReference[oaicite:2]{index=2}.
#    (Merging such close nodes prevents nearly coincident joints that can cause singular stiffness matrices and instability:contentReference[oaicite:3]{index=3}.)

# 2. Node spacing and move limit strategy:
#    Maintain a minimum spacing between nodes to avoid numerical instability from very short bars:contentReference[oaicite:4]{index=4}. 
#    For example, ensure no bar (distance between connected nodes) is below ~0.25 units; if it is, those nodes should be merged as above:contentReference[oaicite:5]{index=5}:contentReference[oaicite:6]{index=6}.
#    Define a move limit (movement radius) for each joint based on its connected bar lengths:contentReference[oaicite:7]{index=7}. 
#    For joint j, let r_j = min( k * min_distance_to_connected_node(j), 0.3 ):contentReference[oaicite:8]{index=8}.
#    – Here min_distance_to_connected_node(j) is the length of the shortest bar incident to node j, and k is a scaling factor (typically k = 1/3; in one case 1/4 for better convergence):contentReference[oaicite:9]{index=9}.
#    – The radius is capped at 0.3 units (maximum move in any direction):contentReference[oaicite:10]{index=10}.
#    These move limits form a spherical neighborhood (ball) around each joint’s current position, restricting how far the joint can travel in one optimization step:contentReference[oaicite:11]{index=11}.
#    (In addition, linear constraints can be applied to keep joints within the overall design domain bounds as needed:contentReference[oaicite:12]{index=12}.)

# 3. Adaptive iterative update of move limits and spacing:
#    Use an adaptive process: start with small move regions for each node, then gradually expand them in subsequent iterations:contentReference[oaicite:13]{index=13}.
#    After each optimization cycle, update each joint’s move radius r_j based on the new structure (recompute min distances to connected nodes and apply r_j = min(k * new_min_distance, 0.3)):contentReference[oaicite:14]{index=14}.
#    If the new design has any nodes that end up too close (or any bars ≤ 0.25 length), merge those nodes before the next iteration to maintain stability:contentReference[oaicite:15]{index=15}:contentReference[oaicite:16]{index=16}.
#    Continue this iterative refine-and-solve procedure until convergence – e.g. until the design’s volume change is negligible and no extremely short bars (≤ 0.25) remain in the structure:contentReference[oaicite:17]{index=17}.

### 2025-09-26 实现补充
- `Sequential_Convex_Programming/node_merger.py` 现在会为每个节点记录代表节点，所有指向被合并节点的杆件都会重定向到其代表节点。
- 仅当两端映射到同一个代表节点（形成零长度杆）时才丢弃该杆件；若多根杆共用同一端点对，则合并并累加截面积，并在累加后对截面做 `A_max` 截断，确保基线仍满足面积上限。
- 这样避免旧版本直接跳过相关杆件造成的拓扑缺失，有助于减少 MOSEK 求解失败和对称性缺口。
