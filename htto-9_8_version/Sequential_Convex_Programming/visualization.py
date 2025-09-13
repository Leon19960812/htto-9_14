import numpy as np
import matplotlib.pyplot as plt
import time
import os
import warnings
from truss_system_initializer import TrussSystemInitializer

# 抑制libpng警告
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


class TrussVisualization:
    def __init__(self):
        pass

    def visualize_results(self, optimizer):
        """可视化优化结果"""
        if not hasattr(optimizer, 'final_areas'):
            print("✗ No optimization results to visualize!")
            print("Please run optimization first!")
            return
        
        print("\n" + "=" * 50)
        print("GENERATING VISUALIZATION")
        print("=" * 50)
        
        try:
            # 设置matplotlib后端
            import matplotlib
            matplotlib.use('TkAgg')  # 或者 'Qt5Agg'
            
            # 创建结果目录
            results_dir = "results"
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
                print(f"Created results directory: {results_dir}")
            
            fig = plt.figure(figsize=(18, 12))
            print("Created figure...")
            
            # 子图1：初始ground structure
            ax1 = plt.subplot(2, 4, 1)
            initial_areas = np.ones(optimizer.n_elements) * np.mean(optimizer.final_areas)
            self._plot_structure(optimizer, ax1, initial_areas, "Initial Ground Structure", 
                               linewidth_mode='uniform')  # 固定粗线宽
            print("Plotted initial structure...")
            
            # 子图2：优化前后节点位置对比
            ax2 = plt.subplot(2, 4, 2)
            if hasattr(optimizer, 'final_angles'):
                optimized_coords = optimizer._update_node_coordinates(optimizer.final_angles)
            else:
                optimized_coords = np.array(optimizer.nodes)
            
            # 绘制参考圆弧
            from matplotlib.patches import Arc
            outer_arc = Arc((0, 0), 2*optimizer.radius, 2*optimizer.radius, 
                           angle=0, theta1=0, theta2=180, 
                           fill=False, color='black', linestyle='--', alpha=0.3)
            inner_arc = Arc((0, 0), 2*optimizer.inner_radius, 2*optimizer.inner_radius,
                           angle=0, theta1=0, theta2=180,
                           fill=False, color='gray', linestyle='--', alpha=0.3)
            ax2.add_patch(outer_arc)
            ax2.add_patch(inner_arc)
            
            # 原始节点坐标
            original_coords = np.array(optimizer.nodes)
            
            # 支撑节点（内层两端固定，如果有中间层则中间层两端也固定）
            ln = getattr(getattr(optimizer, 'geometry', optimizer), 'load_nodes', getattr(optimizer, 'outer_nodes', []))
            support_nodes = [ln[0], ln[-1]] if len(ln) >= 2 else []
            
            # 非支撑节点
            non_support_nodes = [i for i in range(len(original_coords)) if i not in support_nodes]
            
            # 绘制原始非支撑节点（小点）
            if non_support_nodes:
                non_support_original = original_coords[non_support_nodes]
                ax2.scatter(non_support_original[:, 0], non_support_original[:, 1], 
                           c='lightgray', s=20, marker='o', edgecolors='gray', 
                           label='Original Nodes', alpha=0.7)
            
            # 绘制箭头连接对应节点（只对非支撑节点）
            for i in non_support_nodes:
                x1, y1 = original_coords[i]
                x2, y2 = optimized_coords[i]
                
                # 计算位移
                dx = x2 - x1
                dy = y2 - y1
                displacement = np.sqrt(dx**2 + dy**2)
                
                # 只有当位移足够大时才画箭头
                if displacement > 0.1:  # 1cm阈值
                    ax2.arrow(x1, y1, dx, dy, 
                             head_width=0.15, head_length=0.2, 
                             fc='black', ec='black', alpha=0.6, 
                             length_includes_head=True)
            
            # 特别标注外层节点（荷载点，排除支撑节点）
            load_nodes = getattr(getattr(optimizer, 'geometry', optimizer), 'load_nodes', getattr(optimizer, 'outer_nodes', []))
            outer_nodes_non_support = [node for node in load_nodes if node not in support_nodes]
            if outer_nodes_non_support:
                outer_original = original_coords[outer_nodes_non_support]
                outer_optimized = optimized_coords[outer_nodes_non_support]
                ax2.scatter(outer_original[:, 0], outer_original[:, 1], 
                           c='lightgray', s=20, marker='o', edgecolors='darkred', 
                           label='Original Outer Nodes', alpha=0.8)
                ax2.scatter(outer_optimized[:, 0], outer_optimized[:, 1], 
                           c='red', s=20, marker='o', edgecolors='darkmagenta', 
                           label='Optimized Outer Nodes', alpha=0.9)
            
            # 绘制支撑节点
            support_original = original_coords[support_nodes]
            support_optimized = optimized_coords[support_nodes]
            ax2.scatter(support_original[:, 0], support_original[:, 1], 
                       c='blue', s=40, marker='^', edgecolors='darkblue', 
                       label='Fixed Supports', alpha=0.8)
            ax2.scatter(support_optimized[:, 0], support_optimized[:, 1], 
                       c='darkblue', s=40, marker='^', edgecolors='black', 
                       label='Fixed Supports', alpha=0.9)
            
            ax2.set_xlim(-1.2 * optimizer.radius, 1.2 * optimizer.radius)
            ax2.set_ylim(-0.2 * optimizer.radius, 1.2 * optimizer.radius)
            ax2.set_aspect('equal')
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=8)
            ax2.set_title("Node Position Comparison", fontweight='bold')
            print("Plotted node position comparison...")
            
            # 子图3：拓扑清理后
            ax3 = plt.subplot(2, 4, 3)
            cleaned_areas = optimizer.final_areas.copy()
            cleaned_areas[cleaned_areas < optimizer.removal_threshold] = 0
            self._plot_structure(optimizer, ax3, cleaned_areas, "", linewidth_mode='variable', node_coords=optimized_coords)  # 用优化后坐标
            print("Plotted cleaned structure...")
            
            # 子图4：Trust Region Radius vs Iteration
            ax4 = plt.subplot(2, 4, 4)
            self._plot_trust_region_evolution(optimizer, ax4)
            print("Plotted trust region evolution...")
            
            # 子图5：截面积分布
            ax5 = plt.subplot(2, 4, 5)
            areas_mm2 = optimizer.final_areas * 1e6
            valid_areas = areas_mm2[areas_mm2 > optimizer.removal_threshold * 1e6]
            
            if len(valid_areas) > 0:
                ax5.hist(valid_areas, bins=min(25, len(valid_areas)), alpha=0.7, 
                        color='skyblue', edgecolor='black')
                ax5.axvline(x=optimizer.removal_threshold*1e6, color='red', linestyle='--', 
                           label=f'Removal Threshold')
                ax5.set_xlabel('Cross-sectional Area (mm²)')
                ax5.set_ylabel('Number of Members')
                ax5.set_title('Area Distribution')
                ax5.legend()
                ax5.grid(True, alpha=0.3)
            else:
                ax5.text(0.5, 0.5, 'No valid areas\nto display', 
                        ha='center', va='center', transform=ax5.transAxes)
                ax5.set_title('Area Distribution')
            print("Plotted area distribution...")
            
            # 子图6：荷载分布
            ax6 = plt.subplot(2, 4, 6)
            self._plot_loads(optimizer, ax6)
            print("Plotted load distribution...")
            
            # 子图7：总结统计
            ax7 = plt.subplot(2, 4, 7)
            ax7.axis('off')
            
            effective_elements = np.sum(optimizer.final_areas > optimizer.removal_threshold)
            total_volume = np.sum(optimizer.final_areas * optimizer.element_lengths)
            

    
            # 检测结构类型
            is_3layer = hasattr(optimizer, 'middle_nodes') and optimizer.middle_nodes
            structure_type = "3-layer" if is_3layer else "2-layer"
            summary_text = f"""SCP Optimization Results

Method: Sequential Convex Programming
Formulation: Joint Topology-Geometry

Structure:
• Radius: {optimizer.radius}m
• Sectors: {optimizer.n_sectors}
• Depth: {optimizer.depth}m
• Total Members: {optimizer.n_elements}

Results:
• Optimal Compliance: {optimizer.final_compliance:.3e}
• Effective Members: {effective_elements}
• Removed Members: {optimizer.n_elements - effective_elements}
• Total Volume: {total_volume*1e6:.1f} cm³
• Volume Limit: {optimizer.volume_constraint*1e6:.1f} cm³
• Volume Utilization: {total_volume/optimizer.volume_constraint:.1%}

Verification:
• Status: {'✓ Passed' if optimizer.verification_passed else '✗ Failed'}
• Method: Direct stiffness calculation
            """
            
            ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            print("Added summary statistics...")
            
            # 子图8：优化收敛历史（如果有的话）
            ax8 = plt.subplot(2, 4, 8)
            ax8.axis('off')
            ax8.text(0.05, 0.95, "Trust Region Evolution\n\nThis chart shows how the\ntrust region radius changes\nduring optimization.\n\nGreen: Expansion\nRed: Contraction\nBlue: No change", 
                    transform=ax8.transAxes, fontsize=10, verticalalignment='top', 
                    fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            print("Added trust region legend...")
            
            plt.tight_layout()
            
            # 保存图像
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            fig_path = os.path.join(results_dir, f"scp_results_{timestamp}.png")
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"✓ Results saved to: {fig_path}")
            
            # 显示图形
            print("Displaying figure...")
            plt.show(block=True)  # 阻塞显示，确保图形显示出来
            
            # 打印最终统计
            self._print_final_statistics(optimizer)
            
        except Exception as e:
            print(f"✗ Visualization failed: {e}")
            import traceback
            traceback.print_exc()
            
            # 至少打印统计信息
            try:
                self._print_final_statistics(optimizer)
            except:
                print("Could not print statistics either")
    
    def _print_final_statistics(self, optimizer):
        """打印最终统计信息 - 只添加结构类型信息"""
        if not hasattr(optimizer, 'final_areas'):
            print("No results to display")
            return
            
        print("\n" + "=" * 60)
        print("FINAL OPTIMIZATION STATISTICS")
        print("=" * 60)
        
        effective_elements = np.sum(optimizer.final_areas > optimizer.removal_threshold)
        total_volume = np.sum(optimizer.final_areas * optimizer.element_lengths)
        
        # 检测结构类型
        is_3layer = hasattr(optimizer, 'middle_nodes') and optimizer.middle_nodes
        structure_type = "3-layer" if is_3layer else "2-layer"
        
        print(f"Method: Sequential Convex Programming")
        print(f"Solver: Successfully solved")
        print(f"Verification: {'✓ Passed' if optimizer.verification_passed else '✗ Failed'}")
        print()
        
        print(f"Structure Configuration:")
        print(f"  Type: {structure_type} structure")  # 只添加这一行
        print(f"  Radius: {optimizer.radius}m")
        print(f"  Sectors: {optimizer.n_sectors}")
        print(f"  Depth: {optimizer.depth}m")
        print(f"  Total Candidate Members: {optimizer.n_elements}")
        print()
        
        # 其余统计信息保持不变
        print(f"Optimization Results:")
        print(f"  Optimal Compliance: {optimizer.final_compliance:.6e}")
        print(f"  Effective Members: {effective_elements}")
        print(f"  Removed Members: {optimizer.n_elements - effective_elements}")
        print(f"  Removal Rate: {(optimizer.n_elements - effective_elements)/optimizer.n_elements:.1%}")
        print()
        
        print(f"Volume Statistics:")
        print(f"  Total Volume: {total_volume*1e6:.1f} cm³")
        print(f"  Volume Constraint: {optimizer.volume_constraint*1e6:.1f} cm³")
        print(f"  Volume Utilization: {total_volume/optimizer.volume_constraint:.1%}")
        print()
        
        active_areas = optimizer.final_areas[optimizer.final_areas > optimizer.removal_threshold]
        if len(active_areas) > 0:
            print(f"Cross-sectional Area Statistics (Active Members):")
            print(f"  Count: {len(active_areas)}")
            print(f"  Average: {np.mean(active_areas)*1e6:.2f} mm²")
            print(f"  Range: [{np.min(active_areas)*1e6:.2f}, {np.max(active_areas)*1e6:.2f}] mm²")
            print(f"  Standard Deviation: {np.std(active_areas)*1e6:.2f} mm²")
        else:
            print("No active members found!")
        
        print("=" * 60)
    
    def _plot_structure(self, optimizer, ax, areas, title, linewidth_mode='variable', node_coords=None):
        """绘制结构"""
        if node_coords is None:
            node_coords = np.array(optimizer.nodes)
        
        # 绘制参考圆弧
        from matplotlib.patches import Arc
        outer_arc = Arc((0, 0), 2*optimizer.radius, 2*optimizer.radius, 
                    angle=0, theta1=0, theta2=180, 
                    fill=False, color='black', linestyle='--', alpha=0.3)
        inner_arc = Arc((0, 0), 2*optimizer.inner_radius, 2*optimizer.inner_radius,
                    angle=0, theta1=0, theta2=180,
                    fill=False, color='gray', linestyle='--', alpha=0.3)
        ax.add_patch(outer_arc)
        ax.add_patch(inner_arc)
        
        # 如果有中间层，绘制中间层圆弧（与内层相同样式）
        if hasattr(optimizer, 'middle_radius') and optimizer.middle_radius is not None:
            middle_arc = Arc((0, 0), 2*optimizer.middle_radius, 2*optimizer.middle_radius,
                            angle=0, theta1=0, theta2=180,
                            fill=False, color='gray', linestyle='--', alpha=0.3)
            ax.add_patch(middle_arc)
        
        # 绘制单元（保持原有逻辑不变）
        for i, ((node1, node2), area) in enumerate(zip(optimizer.elements, areas)):
            if area > optimizer.removal_threshold:
                x1, y1 = node_coords[node1]
                x2, y2 = node_coords[node2]
                
                area_ratio = area / optimizer.A_max
                alpha = 1
                
                # 根据模式设置线宽
                if linewidth_mode == 'uniform':
                    linewidth = 1
                    color = 'black'
                elif linewidth_mode == 'fine':
                    linewidth = 0.8 + 2.0 * area_ratio
                    color = 'darkblue'
                else:  # 'variable'
                    linewidth = 1+ 1 * area_ratio
                    color = 'darkblue'
                
                ax.plot([x1, x2], [y1, y2], color=color, 
                    linewidth=linewidth, alpha=alpha)
        
        # 绘制节点（保持原有配色和逻辑）
        nodes_array = node_coords
        
        # 外层节点（荷载点）
        load_nodes = getattr(getattr(optimizer, 'geometry', optimizer), 'load_nodes', getattr(optimizer, 'outer_nodes', []))
        outer_coords = nodes_array[load_nodes]
        ax.scatter(outer_coords[:, 0], outer_coords[:, 1], 
                c='red', s=60, marker='o', edgecolors='black', 
                label='Load Points', zorder=5)
        
        # 支撑节点（内层两端固定，如果有中间层则中间层两端也固定）
        ln = getattr(getattr(optimizer, 'geometry', optimizer), 'load_nodes', getattr(optimizer, 'outer_nodes', []))
        support_nodes = [ln[0], ln[-1]] if len(ln) >= 2 else []
        
        # 内层节点（绿色，排除支撑节点）
        # no inner layer in runtime
        inner_nodes_non_support = []
            ax.scatter(inner_coords[:, 0], inner_coords[:, 1], 
                    c='green', s=40, marker='o', edgecolors='black', 
                    label='Inner Nodes', zorder=5)
        
        # 中间层节点（如果存在，也用绿色，排除支撑节点）
        # no middle layer in runtime
                ax.scatter(middle_coords[:, 0], middle_coords[:, 1], 
                        c='green', s=40, marker='o', edgecolors='black', 
                        zorder=5)  # 不单独添加label，与内层节点归为一类
        
        # 绘制支撑节点
        support_coords = nodes_array[support_nodes]
        ax.scatter(support_coords[:, 0], support_coords[:, 1], 
                c='blue', s=80, marker='^', edgecolors='black', 
                label='Fixed Supports', zorder=6)
        
        ax.set_xlim(-1.2 * optimizer.radius, 1.2 * optimizer.radius)
        ax.set_ylim(-0.2 * optimizer.radius, 1.2 * optimizer.radius)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title(title, fontweight='bold')
    
    def _plot_loads(self, optimizer, ax):
        """绘制荷载分布"""
        # 使用最终角度更新后的坐标（若有），确保与当前几何一致
        if hasattr(optimizer, 'final_angles') and optimizer.final_angles is not None:
            nodes_array = optimizer._update_node_coordinates(optimizer.final_angles)
        else:
            nodes_array = np.array(optimizer.nodes)
        
        # 使用当前几何和坐标动态计算荷载向量，避免合并节点后与旧向量不一致
        try:
            current_load_vector = optimizer._compute_load_vector(nodes_array)
        except Exception:
            # 退化处理：若计算失败，回退到已存储的向量（可能不完全一致，但可避免崩溃）
            current_load_vector = optimizer.load_vector
        
        # 绘制径向荷载
        max_load = np.max(np.sqrt(current_load_vector[::2]**2 + current_load_vector[1::2]**2))
        
        load_nodes = getattr(getattr(optimizer, 'geometry', optimizer), 'load_nodes', getattr(optimizer, 'outer_nodes', []))
        for i, node_idx in enumerate(load_nodes):
            x, y = nodes_array[node_idx]
            load_x = current_load_vector[2*node_idx]
            load_y = current_load_vector[2*node_idx + 1]
            
            if abs(load_x) > 1e-6 or abs(load_y) > 1e-6:
                load_magnitude = np.sqrt(load_x**2 + load_y**2)
                arrow_scale = load_magnitude / max_load * 0.3
                
                ax.arrow(x, y, load_x/load_magnitude * arrow_scale, 
                        load_y/load_magnitude * arrow_scale,
                        head_width=0.05, head_length=0.05, 
                        fc='red', ec='red', alpha=0.8)
        
        # 绘制结构轮廓
        for i, ((node1, node2), area) in enumerate(zip(optimizer.elements, optimizer.final_areas)):
            if area > optimizer.removal_threshold:
                x1, y1 = optimizer.nodes[node1]
                x2, y2 = optimizer.nodes[node2]
                ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.3, linewidth=0.5)
        
        # 支撑节点（内层两端固定，如果有中间层则中间层两端也固定）
        ln = getattr(getattr(optimizer, 'geometry', optimizer), 'load_nodes', getattr(optimizer, 'outer_nodes', []))
        support_nodes = [ln[0], ln[-1]] if len(ln) >= 2 else []
        
        # 绘制节点
        load_nodes = getattr(getattr(optimizer, 'geometry', optimizer), 'load_nodes', getattr(optimizer, 'outer_nodes', []))
        outer_coords = nodes_array[load_nodes]
        ax.scatter(outer_coords[:, 0], outer_coords[:, 1], 
                  c='red', s=40, marker='o', edgecolors='black', alpha=0.7)
        
        # 内层节点（排除支撑节点）
        inner_nodes_non_support = []
            ax.scatter(inner_coords[:, 0], inner_coords[:, 1], 
                      c='green', s=30, marker='o', edgecolors='black', alpha=0.7)
        
        # 中间层节点（如果存在，排除支撑节点）
        # no middle layer
                ax.scatter(middle_coords[:, 0], middle_coords[:, 1], 
                          c='green', s=30, marker='o', edgecolors='black', alpha=0.7)
        
        ax.set_xlim(-1.2 * optimizer.radius, 1.2 * optimizer.radius)
        ax.set_ylim(-0.2 * optimizer.radius, 1.2 * optimizer.radius)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title('Hydrostatic Pressure Loading', fontweight='bold')


    def plot_single_figure(self, optimizer, figure_type="ground_structure", save_path="results/ground_structure.png", figsize=(8, 6)):
        """
        单独绘制并保存特定类型的图像
        
        Parameters:
        - figure_type: "ground_structure", "optimized", "cleaned", "node_comparison", "load_distribution", "area_histogram"
        - save_path: 保存路径，如果为None则只显示不保存
        - figsize: 图像大小
        """
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if figure_type == "ground_structure":
            # 绘制初始ground structure
            initial_areas = np.ones(optimizer.n_elements) * np.mean(optimizer.final_areas)
            self._plot_structure(optimizer, ax, initial_areas, title="", linewidth_mode='uniform')
        
        
        elif figure_type == "optimized":
            # 绘制优化后结果
            self._plot_structure(optimizer, ax, optimizer.final_areas, 
                            "SCP Optimized Structure", 
                            linewidth_mode='variable')
            
        elif figure_type == "cleaned":
            # 绘制清理后结构
            cleaned_areas = optimizer.final_areas.copy()
            cleaned_areas[cleaned_areas < optimizer.removal_threshold] = 0
            
            if hasattr(optimizer, 'final_angles'):
                optimized_coords = optimizer._update_node_coordinates(optimizer.final_angles)
            else:
                optimized_coords = np.array(optimizer.nodes)
                
            self._plot_structure(optimizer, ax, cleaned_areas, 
                            "", 
                            linewidth_mode='variable', node_coords=optimized_coords)
            
        elif figure_type == "node_comparison":
            # 绘制节点位置对比
            self._plot_node_comparison(optimizer, ax)
            
        elif figure_type == "load_distribution":
            # 绘制载荷分布
            self._plot_loads(optimizer, ax)
            
        elif figure_type == "area_histogram":
            # 绘制面积分布直方图
            self._plot_area_histogram(optimizer, ax)
        
        plt.tight_layout()
        
        # 保存图像
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        plt.show()
        return fig, ax
    
    def _plot_node_comparison(self, optimizer, ax):
        """绘制节点位置对比"""
        if hasattr(optimizer, 'final_angles'):
            optimized_coords = optimizer._update_node_coordinates(optimizer.final_angles)
        else:
            optimized_coords = np.array(optimizer.nodes)
        
        # 原始节点坐标
        original_coords = np.array(optimizer.nodes)
        
        # 支撑节点（内层两端固定，如果有中间层则中间层两端也固定）
        support_nodes = [optimizer.inner_nodes[0], optimizer.inner_nodes[-1]]
        if hasattr(optimizer, 'middle_nodes') and optimizer.middle_nodes:
            support_nodes.extend([optimizer.middle_nodes[0], optimizer.middle_nodes[-1]])
        
        # 非支撑节点
        non_support_nodes = [i for i in range(len(original_coords)) if i not in support_nodes]
        
        # 绘制原始非支撑节点（小点）
        if non_support_nodes:
            non_support_original = original_coords[non_support_nodes]
            ax.scatter(non_support_original[:, 0], non_support_original[:, 1], 
                       c='green', s=20, marker='o', edgecolors='gray', 
                       label='Original Nodes', alpha=0.7)
        
        # 绘制箭头连接对应节点（只对非支撑节点）
        for i in non_support_nodes:
            x1, y1 = original_coords[i]
            x2, y2 = optimized_coords[i]
            
            # 计算位移
            dx = x2 - x1
            dy = y2 - y1
            displacement = np.sqrt(dx**2 + dy**2)
            
            # 只有当位移足够大时才画箭头
            if displacement > 0.01:  # 1cm阈值
                ax.arrow(x1, y1, dx, dy, 
                         head_width=0.1, head_length=0.1, 
                         fc='blue', ec='blue', alpha=0.6, 
                         length_includes_head=True)
        
        # 特别标注外层节点（荷载点，排除支撑节点）
        load_nodes = getattr(getattr(optimizer, 'geometry', optimizer), 'load_nodes', getattr(optimizer, 'outer_nodes', []))
        outer_nodes_non_support = [node for node in load_nodes if node not in support_nodes]
        if outer_nodes_non_support:
            outer_original = original_coords[outer_nodes_non_support]
            outer_optimized = optimized_coords[outer_nodes_non_support]
            ax.scatter(outer_original[:, 0], outer_original[:, 1], 
                       c='red', s=30, marker='o', edgecolors='darkred', 
                       label='Original Outer Nodes', alpha=0.8)
            ax.scatter(outer_optimized[:, 0], outer_optimized[:, 1], 
                       c='magenta', s=30, marker='o', edgecolors='darkmagenta', 
                       label='Optimized Outer Nodes', alpha=0.9)
        
        # 绘制支撑节点
        support_original = original_coords[support_nodes]
        support_optimized = optimized_coords[support_nodes]
        ax.scatter(support_original[:, 0], support_original[:, 1], 
                   c='blue', s=40, marker='^', edgecolors='darkblue', 
                   label='Fixed Supports', alpha=0.8)
        ax.scatter(support_optimized[:, 0], support_optimized[:, 1], 
                   c='darkblue', s=40, marker='^', edgecolors='black', 
                   label='Fixed Supports', alpha=0.9)
        
        ax.set_xlim(-1.2 * optimizer.radius, 1.2 * optimizer.radius)
        ax.set_ylim(-0.2 * optimizer.radius, 1.2 * optimizer.radius)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_title("Node Position Comparison", fontweight='bold')
    
    def _plot_area_histogram(self, optimizer, ax):
        """绘制面积分布直方图"""
        areas_mm2 = optimizer.final_areas * 1e6
        valid_areas = areas_mm2[areas_mm2 > optimizer.removal_threshold * 1e6]
        
        if len(valid_areas) > 0:
            ax.hist(valid_areas, bins=min(25, len(valid_areas)), alpha=0.7, 
                    color='skyblue', edgecolor='black')
            ax.axvline(x=optimizer.removal_threshold*1e6, color='red', linestyle='--', 
                       label=f'Removal Threshold')
            ax.set_xlabel('Cross-sectional Area (mm²)',fontsize=12)
            ax.set_ylabel('Number of Members',fontsize=12)
            ax.set_title('')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No valid areas\nto display', 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Area Distribution')
    
    def plot_ground_structure_only(self, geometry_params, save_path=None):
        """只生成ground structure，不需要优化结果"""
        
        # 创建简化的初始化器
        initializer = TrussSystemInitializer(**geometry_params)
        
        # 创建图像
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 模拟optimizer对象
        class MockOptimizer:
            def __init__(self, initializer):
                self.__dict__.update(initializer.__dict__)
                self.final_areas = np.ones(self.n_elements) * 0.001
        
        mock_optimizer = MockOptimizer(initializer)
        
        # 使用现有方法绘制
        self._plot_structure(mock_optimizer, ax, mock_optimizer.final_areas, 
                            title="Ground Structure", linewidth_mode='uniform')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_trust_region_evolution_only(self, optimizer, save_path=None, figsize=(12, 8), 
                                        dpi=300, show_plot=True, format='png'):
        """单独绘制信赖域半径演化图"""
        if not hasattr(optimizer, 'trust_radius_history') or len(optimizer.trust_radius_history) < 2:
            print("✗ 无信赖域数据可显示！请先运行优化。")
            return None
        
        print("\n" + "=" * 60)
        print("GENERATING TRUST REGION EVOLUTION PLOT")
        print("=" * 60)
        
        try:
            # 设置matplotlib后端
            import matplotlib
            matplotlib.use('TkAgg')  # 或者 'Qt5Agg'
            
            # 创建图像
            fig, ax = plt.subplots(figsize=figsize)
            
            # 准备数据
            iterations = list(range(len(optimizer.trust_radius_history)))
            radii = optimizer.trust_radius_history
            
            # 绘制信赖域半径演化曲线
            ax.plot(iterations, radii, 'b-', linewidth=3, alpha=0.9, 
                   label='Trust Region Radius', zorder=1)
            
            # 标注关键事件
            if hasattr(optimizer, 'trust_radius_changes') and optimizer.trust_radius_changes:
                print(f"发现 {len(optimizer.trust_radius_changes)} 个信赖域变化事件")
                
                for i, change in enumerate(optimizer.trust_radius_changes):
                    iteration = change['iteration']
                    old_radius = change['old_radius']
                    new_radius = change['new_radius']
                    change_type = change['type']
                    rho = change['rho']
                    
                    # 根据变化类型选择颜色和标记
                    if change_type == "EXPAND":
                        color = 'green'
                        marker = '^'
                        label_suffix = 'EXP'
                        event_name = 'Expansion'
                    elif change_type == "SHRINK":
                        color = 'red'
                        marker = 'v'
                        label_suffix = 'SHR'
                        event_name = 'Shrinkage'
                    else:  # SHRINK_FAILURE
                        color = 'darkred'
                        marker = 'x'
                        label_suffix = 'FAIL'
                        event_name = 'Failure Shrink'
                    
                    # 绘制变化点
                    ax.scatter(iteration, new_radius, c=color, s=150, marker=marker, 
                              alpha=0.9, edgecolors='black', linewidth=2, zorder=3)
                    
                    # 添加详细标签
                    label_text = f'{label_suffix}\nρ={rho:.3f}'
                    ax.annotate(label_text, 
                               xy=(iteration, new_radius), 
                               xytext=(15, 15), textcoords='offset points',
                               fontsize=10, ha='center', va='bottom',
                               bbox=dict(boxstyle='round,pad=0.4', facecolor=color, 
                                        alpha=0.8, edgecolor='black'),
                               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2',
                                             color='black', lw=1.5))
                    
                    print(f"  事件 {i+1}: {event_name} at iteration {iteration}, "
                          f"ρ={rho:.3f}, {old_radius:.4f} → {new_radius:.4f}")
            
            # 设置图形属性
            ax.set_xlabel('Iteration Number', fontsize=14, fontweight='bold')
            ax.set_ylabel('Trust Region Radius Δ^(k)', fontsize=14, fontweight='bold')
            ax.set_title('Trust Region Evolution During Optimization', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.4, linestyle='--')
            ax.legend(fontsize=12, loc='upper right')
            
            # 自动选择对数刻度（如果半径变化很大）
            radius_ratio = max(radii) / min(radii)
            if radius_ratio > 100:
                ax.set_yscale('log')
                ax.set_ylabel('Trust Region Radius Δ^(k) (log scale)', fontsize=14, fontweight='bold')
                print(f"使用对数刻度 (半径变化比例: {radius_ratio:.1f})")
            
            # 添加统计信息
            total_iterations = len(iterations) - 1
            total_changes = len(optimizer.trust_radius_changes) if hasattr(optimizer, 'trust_radius_changes') else 0
            expansion_count = sum(1 for c in optimizer.trust_radius_changes if c['type'] == 'EXPAND') if optimizer.trust_radius_changes else 0
            shrink_count = sum(1 for c in optimizer.trust_radius_changes if c['type'] == 'SHRINK') if optimizer.trust_radius_changes else 0
            failure_count = sum(1 for c in optimizer.trust_radius_changes if c['type'] == 'SHRINK_FAILURE') if optimizer.trust_radius_changes else 0
            
            stats_text = f"""Statistics:
• Total Iterations: {total_iterations}
• Total Changes: {total_changes}
• Expansions: {expansion_count}
• Shrinkages: {shrink_count}
• Failures: {failure_count}
• Final Radius: {radii[-1]:.6f}
• Initial Radius: {radii[0]:.6f}"""
            
            # 在右上角添加统计信息
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
                   fontsize=10, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', 
                            alpha=0.8, edgecolor='black'),
                   fontfamily='monospace')
            
            # 优化布局
            plt.tight_layout()
            
            # 保存图像
            if save_path:
                # 确保目录存在
                import os
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                # 根据格式选择保存方式
                if format.lower() == 'pdf':
                    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', format='pdf')
                elif format.lower() == 'png':
                    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', format='png')
                elif format.lower() == 'svg':
                    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', format='svg')
                else:
                    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
                
                print(f"✓ 信赖域演化图已保存至: {save_path}")
            
            # 显示图形
            if show_plot:
                print("显示信赖域演化图...")
                plt.show(block=True)
            
            print("✓ 信赖域演化图生成完成！")
            return fig, ax
            
        except Exception as e:
            print(f"✗ 信赖域演化图生成失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def plot_trust_region_evolution_with_compliance(self, optimizer, save_path=None, 
                                                   figsize=(16, 10), dpi=300, 
                                                   show_plot=True, format='png'):
        """绘制信赖域半径演化与柔度变化的对比图"""
        if not hasattr(optimizer, 'trust_radius_history') or len(optimizer.trust_radius_history) < 2:
            print("✗ 无信赖域数据可显示！请先运行优化。")
            return None
        
        print("\n" + "=" * 60)
        print("GENERATING TRUST REGION + COMPLIANCE EVOLUTION PLOT")
        print("=" * 60)
        
        try:
            # 设置matplotlib后端
            import matplotlib
            matplotlib.use('TkAgg')
            
            # 创建子图
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                           gridspec_kw={'height_ratios': [2, 1]})
            
            # 子图1：信赖域半径演化
            iterations = list(range(len(optimizer.trust_radius_history)))
            radii = optimizer.trust_radius_history
            
            ax1.plot(iterations, radii, 'b-', linewidth=3, alpha=0.9, 
                    label='Trust Region Radius', zorder=1)
            
            # 标注关键事件
            if hasattr(optimizer, 'trust_radius_changes') and optimizer.trust_radius_changes:
                for change in optimizer.trust_radius_changes:
                    iteration = change['iteration']
                    new_radius = change['new_radius']
                    change_type = change['type']
                    rho = change['rho']
                    
                    if change_type == "EXPAND":
                        color = 'green'
                        marker = '^'
                        label_suffix = 'EXP'
                    elif change_type == "SHRINK":
                        color = 'red'
                        marker = 'v'
                        label_suffix = 'SHR'
                    else:  # SHRINK_FAILURE
                        color = 'darkred'
                        marker = 'x'
                        label_suffix = 'FAIL'
                    
                    ax1.scatter(iteration, new_radius, c=color, s=150, marker=marker, 
                               alpha=0.9, edgecolors='black', linewidth=2, zorder=3)
                    
                    ax1.annotate(f'{label_suffix}\nρ={rho:.3f}', 
                                xy=(iteration, new_radius), 
                                xytext=(15, 15), textcoords='offset points',
                                fontsize=9, ha='center', va='bottom',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, 
                                         alpha=0.8, edgecolor='black'),
                                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2',
                                              color='black', lw=1))
            
            ax1.set_ylabel('Trust Region Radius Δ^(k)', fontsize=12, fontweight='bold')
            ax1.set_title('Trust Region Evolution', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.4, linestyle='--')
            ax1.legend(fontsize=11, loc='upper right')
            
            # 自动选择对数刻度
            radius_ratio = max(radii) / min(radii)
            if radius_ratio > 100:
                ax1.set_yscale('log')
                ax1.set_ylabel('Trust Region Radius Δ^(k) (log scale)', fontsize=12, fontweight='bold')
            
            # 子图2：步长柔度值（如果有数据）
            if hasattr(optimizer, 'step_details') and optimizer.step_details:
                print(f"发现 {len(optimizer.step_details)} 个步长详细信息")
                
                # 提取数据（试探与预测）
                step_iterations = [step['iteration'] for step in optimizer.step_details]
                actual_compliances = [step['actual_compliance'] for step in optimizer.step_details]
                predicted_compliances = [step['predicted_compliance'] for step in optimizer.step_details]
                rho_values = [step['rho'] for step in optimizer.step_details]
                # 提取接受后的柔度（单调不增）
                accepted_mask = [bool(step.get('accepted', False)) for step in optimizer.step_details]
                accepted_iters = [it for it, acc in zip(step_iterations, accepted_mask) if acc]
                accepted_compliances = [step['accepted_compliance'] for step in optimizer.step_details if step.get('accepted', False)]
                
                # 绘制试探解柔度与预测柔度
                ax2.plot(step_iterations, actual_compliances, 'g-', linewidth=2, 
                        label='Trial Actual Compliance', alpha=0.7)
                ax2.plot(step_iterations, predicted_compliances, 'r--', linewidth=2, 
                        label='Predicted Compliance', alpha=0.8)
                # 叠加“按迭代对齐”的接受柔度阶梯线：每个迭代显示当前被接受的最佳柔度
                if hasattr(optimizer, 'compliance_history') and optimizer.compliance_history:
                    initial_c = optimizer.compliance_history[0]
                    accepted_series = []
                    last_c = initial_c
                    idx_accept = 1  # compliance_history 索引，0是初始
                    for acc in accepted_mask:
                        if acc and idx_accept < len(optimizer.compliance_history):
                            last_c = optimizer.compliance_history[idx_accept]
                            idx_accept += 1
                        accepted_series.append(last_c)
                    # 构造阶梯曲线：在迭代0放初始，其后为每步后的当前已接受柔度
                    x_points = [0] + step_iterations
                    y_points = [initial_c] + accepted_series
                    ax2.step(x_points, y_points, where='post', color='blue', linewidth=2.5, alpha=0.9, label='Accepted Compliance (monotone)')
                
                # 去除在绿色曲线上的 rho 标注与标记
                
                ax2.set_ylabel('Compliance', fontsize=12, fontweight='bold')
                ax2.set_title('Step Compliance: Actual vs Predicted', fontsize=14, fontweight='bold')
                ax2.grid(True, alpha=0.3, linestyle='--')
                ax2.legend(fontsize=11, loc='upper right')
                
                # 统计信息
                total_steps = len(step_iterations)
                positive_rho = sum(1 for r in rho_values if r > 0)
                negative_rho = sum(1 for r in rho_values if r < 0)
                extreme_rho = sum(1 for r in rho_values if abs(r) > 10)
                
                stats_text = f"""Step Compliance Statistics:
• Total Steps: {total_steps}
• Positive ρ: {positive_rho}
• Negative ρ: {negative_rho}
• Extreme ρ: {extreme_rho}
• Avg ρ: {np.mean(rho_values):.3f}"""
                
                ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes, 
                       fontsize=10, verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', 
                                alpha=0.8, edgecolor='black'),
                       fontfamily='monospace')
                
            else:
                ax2.text(0.5, 0.5, 'No step details available\nRun optimization to see step quality analysis', 
                        ha='center', va='center', transform=ax2.transAxes,
                        fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
                ax2.set_title('Step Quality Analysis (No Data)', fontsize=14, fontweight='bold')
            
            # 设置X轴标签
            ax2.set_xlabel('Iteration Number', fontsize=12, fontweight='bold')
            
            # 优化布局
            plt.tight_layout()
            
            # 保存图像
            if save_path:
                import os
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                if format.lower() == 'pdf':
                    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', format='pdf')
                elif format.lower() == 'png':
                    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', format='png')
                elif format.lower() == 'svg':
                    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', format='svg')
                else:
                    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
                
                print(f"✓ 信赖域+柔度演化图已保存至: {save_path}")
            
            # 显示图形
            if show_plot:
                print("显示信赖域+柔度演化图...")
                plt.show(block=True)
            
            print("✓ 信赖域+柔度演化图生成完成！")
            return fig, (ax1, ax2)
            
        except Exception as e:
            print(f"✗ 信赖域+柔度演化图生成失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _plot_trust_region_evolution(self, optimizer, ax):
        """绘制信赖域半径演化图"""
        if not hasattr(optimizer, 'trust_radius_history') or len(optimizer.trust_radius_history) < 2:
            ax.text(0.5, 0.5, 'No trust region\ndata available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Trust Region Evolution')
            return
        
        # 准备数据
        iterations = list(range(len(optimizer.trust_radius_history)))
        radii = optimizer.trust_radius_history
        
        # 绘制信赖域半径演化曲线
        ax.plot(iterations, radii, 'b-', linewidth=2, alpha=0.8, label='Trust Region Radius')
        
        # 标注关键事件
        if hasattr(optimizer, 'trust_radius_changes') and optimizer.trust_radius_changes:
            for change in optimizer.trust_radius_changes:
                iteration = change['iteration']
                old_radius = change['old_radius']
                new_radius = change['new_radius']
                change_type = change['type']
                rho = change['rho']
                
                # 根据变化类型选择颜色和标记
                if change_type == "EXPAND":
                    color = 'green'
                    marker = '^'
                    label_suffix = 'EXP'
                elif change_type == "SHRINK":
                    color = 'red'
                    marker = 'v'
                    label_suffix = 'SHR'
                else:  # SHRINK_FAILURE
                    color = 'darkred'
                    marker = 'x'
                    label_suffix = 'FAIL'
                
                # 绘制变化点
                ax.scatter(iteration, new_radius, c=color, s=100, marker=marker, 
                          alpha=0.8, edgecolors='black', linewidth=1)
                
                # 添加标签
                ax.annotate(f'{label_suffix}\nρ={rho:.3f}', 
                           xy=(iteration, new_radius), 
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=8, ha='center', va='bottom',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # 设置图形属性
        ax.set_xlabel('Iteration Number')
        ax.set_ylabel('Trust Region Radius Δ^(k)')
        ax.set_title('Trust Region Evolution', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 设置y轴为对数刻度（如果半径变化很大）
        if max(radii) / min(radii) > 100:
            ax.set_yscale('log')
            ax.set_ylabel('Trust Region Radius Δ^(k) (log scale)')

    def plot_step_quality_analysis(self, optimizer, save_path=None, figsize=(14, 10), dpi=300, format='png'):
        """专门绘制步长质量分析图，帮助诊断rho值异常"""
        if not hasattr(optimizer, 'step_details') or not optimizer.step_details:
            print("❌ 没有找到步长详细信息数据")
            print("请先运行优化算法以收集步长质量数据")
            return None
        
        print("📊 绘制步长质量分析图...")
        
        # 创建图形
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # 提取数据
        step_iterations = [step['iteration'] for step in optimizer.step_details]
        actual_reductions = [step['actual_reduction'] for step in optimizer.step_details]
        predicted_reductions = [step['predicted_reduction'] for step in optimizer.step_details]
        rho_values = [step['rho'] for step in optimizer.step_details]
        current_compliances = [step['current_compliance'] for step in optimizer.step_details]
        actual_compliances = [step['actual_compliance'] for step in optimizer.step_details]
        predicted_compliances = [step['predicted_compliance'] for step in optimizer.step_details]
        
        # 子图1：实际下降 vs 预测下降
        ax1.plot(step_iterations, actual_reductions, 'g-', linewidth=2, 
                label='Actual Reduction', alpha=0.8)
        ax1.plot(step_iterations, predicted_reductions, 'r--', linewidth=2, 
                label='Predicted Reduction', alpha=0.8)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 标记异常点
        for i, (iter_num, rho) in enumerate(zip(step_iterations, rho_values)):
            if abs(rho) > 10 or rho < -5:
                color = 'red' if rho < -5 else 'orange'
                ax1.scatter(iter_num, actual_reductions[i], c=color, s=100, 
                          marker='x', zorder=5, edgecolors='black', linewidth=2)
                ax1.annotate(f'ρ={rho:.1f}', 
                           xy=(iter_num, actual_reductions[i]),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=9, bbox=dict(boxstyle='round,pad=0.3', 
                                                facecolor=color, alpha=0.7))
        
        ax1.set_xlabel('Iteration Number', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Compliance Reduction', fontsize=12, fontweight='bold')
        ax1.set_title('Actual vs Predicted Reduction', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(fontsize=11)
        
        # 子图2：rho值演化
        ax2.plot(step_iterations, rho_values, 'b-', linewidth=2, alpha=0.8)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Perfect Match')
        ax2.axhline(y=0.25, color='orange', linestyle='--', alpha=0.5, label='Accept Threshold')
        ax2.axhline(y=0.75, color='green', linestyle='--', alpha=0.5, label='Expand Threshold')
        
        # 标记异常rho值
        for i, (iter_num, rho) in enumerate(zip(step_iterations, rho_values)):
            if abs(rho) > 10 or rho < -5:
                color = 'red' if rho < -5 else 'orange'
                ax2.scatter(iter_num, rho, c=color, s=100, 
                          marker='x', zorder=5, edgecolors='black', linewidth=2)
        
        ax2.set_xlabel('Iteration Number', fontsize=12, fontweight='bold')
        ax2.set_ylabel('ρ (Step Quality Ratio)', fontsize=12, fontweight='bold')
        ax2.set_title('Step Quality Ratio Evolution', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.legend(fontsize=11)
        
        # 子图3：柔度值对比
        ax3.plot(step_iterations, current_compliances, 'b-', linewidth=2, 
                label='Current Compliance', alpha=0.8)
        ax3.plot(step_iterations, actual_compliances, 'g-', linewidth=2, 
                label='Actual Compliance', alpha=0.8)
        ax3.plot(step_iterations, predicted_compliances, 'r--', linewidth=2, 
                label='Predicted Compliance', alpha=0.8)
        
        ax3.set_xlabel('Iteration Number', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Compliance Value', fontsize=12, fontweight='bold')
        ax3.set_title('Compliance Values Comparison', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.legend(fontsize=11)
        
        # 子图4：统计信息
        ax4.axis('off')
        
        # 计算统计信息
        total_steps = len(step_iterations)
        positive_rho = sum(1 for r in rho_values if r > 0)
        negative_rho = sum(1 for r in rho_values if r < 0)
        extreme_rho = sum(1 for r in rho_values if abs(r) > 10)
        very_negative_rho = sum(1 for r in rho_values if r < -5)
        
        # 计算改进统计
        successful_steps = sum(1 for r in actual_reductions if r > 0)
        failed_steps = sum(1 for r in actual_reductions if r <= 0)
        
        # 计算预测准确性
        prediction_errors = []
        for i in range(len(actual_reductions)):
            if abs(predicted_reductions[i]) > 1e-12:
                error = abs(actual_reductions[i] - predicted_reductions[i]) / abs(predicted_reductions[i])
                prediction_errors.append(error)
        
        avg_prediction_error = np.mean(prediction_errors) if prediction_errors else 0
        
        stats_text = f"""Step Quality Analysis Summary:

📊 Basic Statistics:
• Total Steps: {total_steps}
• Successful Steps: {successful_steps}
• Failed Steps: {failed_steps}
• Success Rate: {successful_steps/total_steps*100:.1f}%

🎯 ρ (Step Quality) Analysis:
• Positive ρ: {positive_rho} ({positive_rho/total_steps*100:.1f}%)
• Negative ρ: {negative_rho} ({negative_rho/total_steps*100:.1f}%)
• Extreme ρ (>10): {extreme_rho}
• Very Negative ρ (<-5): {very_negative_rho}

📈 Prediction Quality:
• Avg Prediction Error: {avg_prediction_error:.2f}
• Perfect Predictions (ρ≈1): {sum(1 for r in rho_values if 0.9 < r < 1.1)}
• Good Predictions (ρ>0.25): {sum(1 for r in rho_values if r > 0.25)}

⚠️ 异常情况:
• ρ < -10: {sum(1 for r in rho_values if r < -10)}
• ρ > 10: {sum(1 for r in rho_values if r > 10)}"""
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
               fontsize=10, verticalalignment='top', horizontalalignment='left',
               bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue', 
                        alpha=0.9, edgecolor='black'),
               fontfamily='monospace')
        
        # 优化布局
        plt.tight_layout()
        
        # 保存图像
        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            if format.lower() == 'pdf':
                plt.savefig(save_path, dpi=dpi, bbox_inches='tight', format='pdf')
            elif format.lower() == 'png':
                plt.savefig(save_path, dpi=dpi, bbox_inches='tight', format='png')
            elif format.lower() == 'svg':
                plt.savefig(save_path, dpi=dpi, bbox_inches='tight', format='svg')
            else:
                plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            
            print(f"✅ 步长质量分析图已保存至: {save_path}")
        
        plt.show()
        return fig
