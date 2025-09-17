"""
极坐标几何管理模块

基于trussopt.py的简洁设计理念，实现统一的极坐标几何系统，
替代原有的复杂分层架构。

核心设计原则：
1. 统一的极坐标表示：所有节点用(r, θ)表示
2. 简单的地面结构生成：基于几何约束
3. 清晰的接口：只使用numpy数组，无字典混用
"""

import numpy as np
import itertools
from dataclasses import dataclass
from typing import List, Tuple, Optional
from math import gcd
from shapely.geometry import LineString, Polygon
import matplotlib.pyplot as plt


@dataclass
class PolarNode:
    """极坐标节点定义
    
    统一的节点表示，消除分层概念
    """
    id: int                # 节点ID
    radius: float         # 半径（固定）
    theta: float          # 角度（优化变量）
    node_type: str        # 节点类型：'outer', 'middle', 'inner', 'support'
    is_fixed: bool        # 是否固定（如支撑点）
    
    @property 
    def x(self) -> float:
        """笛卡尔x坐标"""
        return self.radius * np.cos(self.theta)
    
    @property
    def y(self) -> float:
        """笛卡尔y坐标"""
        return self.radius * np.sin(self.theta)
    
    @property
    def coords(self) -> np.ndarray:
        """笛卡尔坐标数组"""
        return np.array([self.x, self.y])


@dataclass
class PolarConfig:
    """极坐标配置
    
    定义地面结构的生成规则，参考trussopt.py的设计
    """
    rings: List[dict]           # 环形配置：[{'radius': float, 'n_nodes': int, 'type': str}]
    support_nodes: List[int] = None       # 支撑节点ID列表
    load_nodes: List[int] = None          # 载荷节点ID列表
    
    def __post_init__(self):
        if self.support_nodes is None:
            self.support_nodes = []
        if self.load_nodes is None:
            self.load_nodes = []


class PolarGeometry:
    """极坐标几何管理器
    
    核心设计：
    - 统一的极坐标节点系统
    - 简化的地面结构生成
    - 与trussopt.py一致的连接逻辑
    """
    
    def __init__(self, config: PolarConfig):
        self.config = config
        self.nodes: List[PolarNode] = []
        self.connections: List[Tuple[int, int]] = []  # (node1_id, node2_id)
        self.elements: np.ndarray = None              # truss elements array
        
        # 生成初始几何
        self._generate_initial_geometry()
    
    def _generate_initial_geometry(self):
        """生成初始几何结构"""
        print("生成极坐标几何结构...")
        
        # 1. 生成节点
        self._generate_nodes()
        
        # 2. 生成连接（地面结构）
        self._generate_ground_structure()
        
        # 3. 转换为truss格式
        self._convert_to_truss_format()
        
        print(f"  节点数: {len(self.nodes)}")
        print(f"  连接数: {len(self.connections)}")
    
    def _generate_nodes(self):
        """生成节点环"""
        node_id = 0
        
        for ring in self.config.rings:
            radius = ring['radius']
            n_nodes = ring['n_nodes']
            node_type = ring['type']
            
            # 均匀分布角度
            angles = np.linspace(0, np.pi, n_nodes)
            
            for i, theta in enumerate(angles):
                # 支撑节点集合由外部显式指定，默认均为可移动节点
                is_fixed = (node_id in self.config.support_nodes)
                
                node = PolarNode(
                    id=node_id,
                    radius=radius,
                    theta=theta,
                    node_type=node_type,
                    is_fixed=is_fixed
                )
                
                self.nodes.append(node)
                node_id += 1
            
            print(f"  生成{node_type}环: {n_nodes}个节点，半径={radius:.2f}")
    
    def _generate_ground_structure(self):
        """生成地面结构连接（trussopt策略）
        
        使用trussopt的全连接策略：
        - 全连接 + 几何过滤 + 互质条件
        """
        print("  生成地面结构连接（trussopt策略）...")
        self._generate_trussopt_style()
    
    def _generate_trussopt_style(self):
        """采用trussopt.py的精确策略生成连接"""
        # 创建半圆形环形区域用于几何包含检查
        ring_polygon = self._create_ring_polygon()
        
        # 数值容差（相对于几何尺度）
        radius_max = max((n.radius for n in self.nodes), default=1.0)
        _EPS_COVER = 1e-7 * radius_max  # 极小缓冲，吸收贴边误差

        # 角序窗口 + 径向家族 去冗余（无层标签）
        n_total = len(self.nodes)
        order = list(range(n_total))
        order.sort(key=lambda idx: self.nodes[idx].theta)
        thetas = np.array([self.nodes[idx].theta for idx in order], dtype=float)
        # 估计平均角步长（跳过数值重复）
        diffs = np.diff(thetas)
        valid_diffs = diffs[diffs > 1e-12]
        avg_step = float(np.median(valid_diffs)) if valid_diffs.size else (np.pi / max(n_total - 1, 1))
        K_THETA_STEPS = 2
        ANG_TOL = 1e-12
        EPS_RADIAL = 0.5 * avg_step  # 判定“近似径向”的角容差

        # 候选集合：基于角差阈值 Δθ ≤ K*avg_step（与环数无关）
        conn_set = set()
        max_dtheta = K_THETA_STEPS * avg_step + ANG_TOL
        for a in range(n_total):
            ia = order[a]
            th_a = thetas[a]
            for b in range(a + 1, n_total):
                dth = thetas[b] - th_a
                if dth > max_dtheta:
                    break
                ib = order[b]
                # 域内性检查（covers + 极小 buffer）
                seg = LineString([self.nodes[ia].coords, self.nodes[ib].coords])
                if ring_polygon.buffer(_EPS_COVER).covers(seg):
                    i0, i1 = (ia, ib) if ia < ib else (ib, ia)
                    conn_set.add((i0, i1))

        # 径向家族：在近似同角度的桶中，仅保留按半径相邻的连接，去掉跨越中间半径的“长径向”
        # 1) 分桶（按角度近似相等）
        buckets = []
        start = 0
        for k in range(1, n_total + 1):
            if k == n_total or abs(thetas[k] - thetas[start]) > EPS_RADIAL:
                bucket_ids = order[start:k]
                if len(bucket_ids) >= 2:
                    buckets.append(bucket_ids)
                start = k
        # 2) 对每个桶，半径排序，仅保留相邻半径的边
        for ids in buckets:
            ids_sorted = sorted(ids, key=lambda nid: self.nodes[nid].radius)
            allowed = set()
            for t in range(len(ids_sorted) - 1):
                i, j = ids_sorted[t], ids_sorted[t + 1]
                i0, i1 = (i, j) if i < j else (j, i)
                allowed.add((i0, i1))
            # 删除该径向家族中“非相邻半径”的连接
            remove_list = []
            for i in range(len(ids_sorted)):
                for j in range(i + 1, len(ids_sorted)):
                    a, b = ids_sorted[i], ids_sorted[j]
                    p = (a, b) if a < b else (b, a)
                    if p in conn_set and p not in allowed:
                        remove_list.append(p)
            for p in remove_list:
                conn_set.discard(p)

        # 写入 connections 列表
        self.connections = list(conn_set)
    
    
    def _create_ring_polygon(self):
        """创建基于实际节点的环形区域（严格依赖 shapely，多边形无回退）。

        使用外层和内层的实际节点坐标构建边界；若多边形无效则尝试
        使用 buffer(0) 修复；仍无效则抛出错误，不再使用任意简化策略。
        """
        # 获取外层和内层节点
        outer_nodes = [node for node in self.nodes if node.node_type == 'outer']
        inner_nodes = [node for node in self.nodes if node.node_type == 'inner']
        
        # 按角度排序（确保多边形顺序正确）
        outer_nodes.sort(key=lambda n: n.theta)
        inner_nodes.sort(key=lambda n: n.theta, reverse=True)  # 内层逆序
        
        # 创建边界点列表
        boundary_points = []
        
        # 1. 外层节点（逆时针）
        for node in outer_nodes:
            boundary_points.append((node.x, node.y))
        
        # 2. 内层节点（顺时针，所以是逆序）
        for node in inner_nodes:
            boundary_points.append((node.x, node.y))
        
        # 创建多边形（无简化回退）
        ring_polygon = Polygon(boundary_points)
        if not ring_polygon.is_valid:
            ring_polygon = ring_polygon.buffer(0)
        if (ring_polygon is None) or (not ring_polygon.is_valid) or ring_polygon.is_empty:
            raise RuntimeError("Failed to construct a valid ring polygon with shapely.")
        return ring_polygon


    
    
    
    
    def _convert_to_truss_format(self):
        """转换为truss分析所需的格式"""
        # 转换连接为elements数组格式
        elements = []
        for i, (node1_id, node2_id) in enumerate(self.connections):
            node1 = self.nodes[node1_id]
            node2 = self.nodes[node2_id]
            
            # 计算单元长度
            dx = node1.x - node2.x
            dy = node1.y - node2.y
            length = np.sqrt(dx**2 + dy**2)
            
            # 格式：[起始节点, 终止节点, 长度, 激活状态]
            elements.append([node1_id, node2_id, length, False])
        
        self.elements = np.array(elements) if elements else np.empty((0, 4))

    def set_support_nodes(self, support_ids: List[int]) -> None:
        """Mark provided node ids as fixed supports and clear others.

        This keeps `config.support_nodes` in sync and updates the runtime
        PolarNode instances so downstream optimizers see the correct fixed set.
        """
        support_set = set(int(i) for i in support_ids)
        self.config.support_nodes = list(sorted(support_set))
        for node in self.nodes:
            node.is_fixed = node.id in support_set
    
    def get_optimization_variables(self) -> np.ndarray:
        """获取优化变量数组（theta值）
        
        Returns:
        --------
        np.ndarray
            所有可优化节点的theta值
        """
        theta_vars = []
        for node in self.nodes:
            if not node.is_fixed:
                theta_vars.append(node.theta)
        
        return np.array(theta_vars)
    
    def update_from_optimization(self, theta_new: np.ndarray):
        """从优化结果更新几何
        
        Parameters:
        -----------
        theta_new : np.ndarray
            新的theta值数组
        """
        var_idx = 0
        for node in self.nodes:
            if not node.is_fixed:
                if var_idx < len(theta_new):
                    node.theta = theta_new[var_idx]
                    var_idx += 1
        
        # 重新生成连接（如果需要）
        self._update_connections()

    def update_from_partial_optimization(self, theta_new: np.ndarray, node_ids: List[int]):
        """Update a subset of free nodes' angles using provided node ids.

        This is a convenience adapter to integrate with optimizers that optimize
        only a selected runtime node set (e.g., load_nodes). Fixed nodes in the
        subset are ignored. After update, element lengths are refreshed.
        """
        n = min(len(theta_new), len(node_ids))
        for i in range(n):
            nid = int(node_ids[i])
            if 0 <= nid < len(self.nodes):
                if not self.nodes[nid].is_fixed:
                    self.nodes[nid].theta = float(theta_new[i])
        self._update_connections()
    
    def _update_connections(self):
        """更新连接（重新计算长度等）"""
        for i, (node1_id, node2_id) in enumerate(self.connections):
            if i < len(self.elements):
                node1 = self.nodes[node1_id]
                node2 = self.nodes[node2_id]
                
                # 重新计算长度
                dx = node1.x - node2.x
                dy = node1.y - node2.y
                length = np.sqrt(dx**2 + dy**2)
                
                self.elements[i, 2] = length
    
    def get_cartesian_coordinates(self) -> np.ndarray:
        """获取所有节点的笛卡尔坐标
        
        Returns:
        --------
        np.ndarray
            节点坐标数组 (n_nodes, 2)
        """
        coords = []
        for node in self.nodes:
            coords.append([node.x, node.y])
        
        return np.array(coords)

    def get_free_node_ids(self) -> List[int]:
        """Return node ids for non-fixed nodes in the same order used by
        get_optimization_variables()."""
        ids: List[int] = []
        for node in self.nodes:
            if not node.is_fixed:
                ids.append(node.id)
        return ids
    
    def visualize_ground_structure(self, save_path: str = None, show_ring_boundary: bool = True):
        """可视化地面结构
        
        Args:
            save_path: 保存图片的路径，如果为None则显示图片
            show_ring_boundary: 是否显示环形边界
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # 获取坐标
        coords = self.get_cartesian_coordinates()
        
        # 绘制节点
        ax.scatter(coords[:, 0], coords[:, 1], c='red', s=50, zorder=3, label='节点')
        
        # 绘制连接
        for i, j in self.connections:
            x_coords = [coords[i, 0], coords[j, 0]]
            y_coords = [coords[i, 1], coords[j, 1]]
            ax.plot(x_coords, y_coords, 'b-', alpha=0.6, linewidth=0.8)
        
        # 绘制环形边界（手动绘制）
        radii = [ring['radius'] for ring in self.config.rings]
        max_radius = max(radii)
        min_radius = min(radii)
        
        # 设置图形属性
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title(f'Ground Structure - {len(self.nodes)}节点, {len(self.connections)}连接')
        ax.set_xlabel('X坐标')
        ax.set_ylabel('Y坐标')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图片已保存到: {save_path}")
        else:
            plt.show()
    
    
    def get_node_coordinates_by_type(self, node_type: str) -> np.ndarray:
        """获取指定类型节点的坐标"""
        coords = []
        for node in self.nodes:
            if node.node_type == node_type:
                coords.append([node.x, node.y])
        
        return np.array(coords) if coords else np.empty((0, 2))
    
    def get_free_dofs(self) -> np.ndarray:
        """获取自由度索引（非固定节点的DOF）"""
        free_dofs = []
        for i, node in enumerate(self.nodes):
            if not node.is_fixed:
                free_dofs.extend([2*i, 2*i+1])  # x和y方向
        
        return np.array(free_dofs)
    
    def get_support_dofs(self) -> np.ndarray:
        """获取支撑约束的DOF"""
        support_dofs = []
        for i, node in enumerate(self.nodes):
            if node.is_fixed:
                support_dofs.extend([2*i, 2*i+1])
        
        return np.array(support_dofs)

    # --------------------------
    # Synchronization helpers
    # --------------------------
    def rebuild_from_geometry(self, geometry) -> None:
        """Synchronize PolarGeometry from an external GeometryData.

        - Nodes: rebuilt from geometry.nodes (x,y) -> (r,θ)
        - Node types: prefer geometry.inner_nodes/outer_nodes (fallback by radius)
        - is_fixed: derived from geometry.fixed_dofs
        - Connections: from geometry.elements
        - Elements: recomputed lengths
        """
        import numpy as _np
        nodes_xy = _np.asarray(getattr(geometry, 'nodes', []), dtype=float)
        n_nodes = int(getattr(geometry, 'n_nodes', nodes_xy.shape[0]))
        if nodes_xy.shape[0] != n_nodes:
            n_nodes = nodes_xy.shape[0]
        inner = set(getattr(geometry, 'inner_nodes', []) or [])
        outer = set(getattr(geometry, 'outer_nodes', []) or [])
        fixed_dofs = getattr(geometry, 'fixed_dofs', []) or []
        fixed_nodes = set(int(d//2) for d in fixed_dofs)

        # Fallback classification by radius if sets missing
        if not inner or not outer:
            if n_nodes > 0:
                r_all = _np.hypot(nodes_xy[:, 0], nodes_xy[:, 1])
                rmin, rmax = float(_np.min(r_all)), float(_np.max(r_all))
                tol = 0.01
                inner = set(int(i) for i,r in enumerate(r_all) if r <= (1.0+tol)*rmin)
                outer = set(int(i) for i,r in enumerate(r_all) if r >= (1.0-tol)*rmax)

        new_nodes = []
        for nid in range(n_nodes):
            x, y = nodes_xy[nid]
            r = float(_np.hypot(x, y))
            th = float(_np.arctan2(y, x))
            if nid in inner:
                ntype = 'inner'
            elif nid in outer:
                ntype = 'outer'
            else:
                ntype = 'middle'
            new_nodes.append(PolarNode(id=nid, radius=r, theta=th, node_type=ntype, is_fixed=(nid in fixed_nodes)))
        self.nodes = new_nodes

        # Update support metadata to stay consistent with geometry snapshot
        support_nodes = getattr(geometry, 'support_nodes', None)
        if support_nodes is None:
            support_nodes = sorted(fixed_nodes)
        self.set_support_nodes(list(support_nodes))

        # connections from geometry.elements
        self.connections = []
        elements_list = getattr(geometry, 'elements', []) or []
        for pair in elements_list:
            i, j = int(pair[0]), int(pair[1])
            self.connections.append((i, j))

        # elements with lengths
        elements = []
        for (i, j) in self.connections:
            xi, yi = nodes_xy[i]
            xj, yj = nodes_xy[j]
            L = float(_np.hypot(xi - xj, yi - yj))
            elements.append([i, j, L, False])
        self.elements = _np.array(elements) if elements else _np.empty((0, 4))


def create_default_polar_config() -> PolarConfig:
    """创建默认的极坐标配置
    
    参考原系统的三层结构，但简化为统一管理
    """
    rings = [
        {'radius': 5.0, 'n_nodes': 13, 'type': 'outer'},
        {'radius': 3.5, 'n_nodes': 13, 'type': 'middle'}, 
        {'radius': 2.0, 'n_nodes': 13, 'type': 'inner'}
    ]
    
    config = PolarConfig(
        rings=rings,
        support_nodes=[0, 12],  # 外层两端作为支撑
        load_nodes=[19]         # 中层某节点作为载荷点
    )
    
    return config


# 便利函数：快速创建极坐标几何
def create_polar_geometry(config: Optional[PolarConfig] = None) -> PolarGeometry:
    """创建极坐标几何实例"""
    if config is None:
        config = create_default_polar_config()
    
    return PolarGeometry(config)


if __name__ == "__main__":
    # 测试代码
    print("=== 极坐标几何管理器测试 ===")
    
    # 创建几何
    geometry = create_polar_geometry()
    
    # 测试基本功能
    print(f"\n节点总数: {len(geometry.nodes)}")
    print(f"连接总数: {len(geometry.connections)}")
    
    # 测试优化变量获取
    theta_vars = geometry.get_optimization_variables()
    print(f"优化变量数: {len(theta_vars)}")
    
    # 测试坐标获取
    coords = geometry.get_cartesian_coordinates()
    print(f"坐标数组形状: {coords.shape}")
    
    print("\n✅ 极坐标几何管理器创建成功")
    
    # 可视化功能演示
    print("\n=== 可视化功能演示 ===")
    
    # 可视化当前几何的地面结构
    print("可视化地面结构...")
    geometry.visualize_ground_structure(save_path="ground_structure.png")
    
    print("\n✅ 可视化完成！")
