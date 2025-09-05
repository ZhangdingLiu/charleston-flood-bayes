#!/usr/bin/env python3
"""
贝叶斯网络可视化

创建详细的网络可视化，帮助验证网络结构的合理性：
1. 有向图可视化（networkx + matplotlib）
2. 节点大小反映洪水频率（边际概率）
3. 边粗细反映条件概率强度
4. 颜色编码显示网络层次结构
5. 交互式可视化
"""

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from model import FloodBayesNetwork
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class NetworkVisualizer:
    """网络可视化器"""
    
    def __init__(self, flood_net):
        self.flood_net = flood_net
        self.network = flood_net.network
        self.marginals_dict = dict(zip(
            flood_net.marginals['link_id'], 
            flood_net.marginals['p']
        ))
        
    def create_network_visualization(self, save_path="network_structure.png", figsize=(20, 16)):
        """创建网络结构可视化"""
        print("🎨 创建贝叶斯网络可视化")
        print("=" * 50)
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Charleston洪水贝叶斯网络分析', fontsize=16, fontweight='bold')
        
        # 1. 基础网络结构
        ax1 = axes[0, 0]
        self._plot_basic_network(ax1)
        
        # 2. 按边际概率着色
        ax2 = axes[0, 1]
        self._plot_marginal_probability_network(ax2)
        
        # 3. 按度中心性着色
        ax3 = axes[1, 0]
        self._plot_centrality_network(ax3)
        
        # 4. 边权重可视化
        ax4 = axes[1, 1]
        self._plot_edge_weights_network(ax4)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 网络可视化已保存至: {save_path}")
        plt.show()
        
        return fig
    
    def _get_node_positions(self):
        """获取节点位置（使用层次布局）"""
        # 计算层次
        layers = self._compute_network_layers()
        
        pos = {}
        layer_heights = {}
        
        # 计算每层的垂直位置
        max_layer = max(layers.values()) if layers else 0
        for layer in range(max_layer + 1):
            layer_heights[layer] = 1.0 - (layer / max(1, max_layer))
        
        # 为每层分配节点位置
        for layer in range(max_layer + 1):
            nodes_in_layer = [node for node, l in layers.items() if l == layer]
            n_nodes = len(nodes_in_layer)
            
            if n_nodes == 1:
                pos[nodes_in_layer[0]] = (0.5, layer_heights[layer])
            else:
                for i, node in enumerate(nodes_in_layer):
                    x = i / max(1, n_nodes - 1)
                    pos[node] = (x, layer_heights[layer])
        
        return pos
    
    def _compute_network_layers(self):
        """计算网络层次（基于拓扑排序）"""
        try:
            # 拓扑排序
            topo_order = list(nx.topological_sort(self.network))
            
            # 计算每个节点的层次
            layers = {}
            for node in topo_order:
                predecessors = list(self.network.predecessors(node))
                if not predecessors:
                    layers[node] = 0
                else:
                    layers[node] = max(layers[pred] for pred in predecessors) + 1
            
            return layers
        except:
            # 如果不是DAG，使用度中心性作为替代
            centrality = nx.degree_centrality(self.network)
            sorted_nodes = sorted(centrality.items(), key=lambda x: x[1])
            
            layers = {}
            n_layers = min(5, len(sorted_nodes) // 8 + 1)
            layer_size = len(sorted_nodes) // n_layers
            
            for i, (node, _) in enumerate(sorted_nodes):
                layers[node] = i // max(1, layer_size)
            
            return layers
    
    def _plot_basic_network(self, ax):
        """绘制基础网络结构"""
        ax.set_title("网络拓扑结构", fontweight='bold')
        
        pos = self._get_node_positions()
        
        # 绘制边
        nx.draw_networkx_edges(
            self.network, pos, ax=ax,
            edge_color='lightgray',
            arrows=True,
            arrowsize=10,
            arrowstyle='->',
            width=0.5,
            alpha=0.6
        )
        
        # 绘制节点
        nx.draw_networkx_nodes(
            self.network, pos, ax=ax,
            node_color='lightblue',
            node_size=300,
            alpha=0.8
        )
        
        # 添加标签（缩短道路名）
        labels = {node: self._shorten_road_name(node) for node in self.network.nodes()}
        nx.draw_networkx_labels(
            self.network, pos, labels, ax=ax,
            font_size=6
        )
        
        ax.set_aspect('equal')
        ax.axis('off')
        
        # 添加统计信息
        ax.text(0.02, 0.98, f"节点: {self.network.number_of_nodes()}\n边: {self.network.number_of_edges()}", 
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    def _plot_marginal_probability_network(self, ax):
        """绘制按边际概率着色的网络"""
        ax.set_title("边际洪水概率", fontweight='bold')
        
        pos = self._get_node_positions()
        
        # 计算节点颜色和大小
        node_probs = [self.marginals_dict.get(node, 0) for node in self.network.nodes()]
        node_sizes = [300 + 1000 * prob for prob in node_probs]
        
        # 绘制边
        nx.draw_networkx_edges(
            self.network, pos, ax=ax,
            edge_color='lightgray',
            arrows=True,
            arrowsize=10,
            arrowstyle='->',
            width=0.5,
            alpha=0.4
        )
        
        # 绘制节点（按概率着色）
        nodes = nx.draw_networkx_nodes(
            self.network, pos, ax=ax,
            node_color=node_probs,
            node_size=node_sizes,
            cmap='Reds',
            vmin=0,
            vmax=max(node_probs) if node_probs else 1,
            alpha=0.8
        )
        
        # 添加颜色条
        if nodes:
            cbar = plt.colorbar(nodes, ax=ax, shrink=0.6)
            cbar.set_label('洪水概率', fontsize=8)
        
        # 添加高概率节点标签
        high_prob_nodes = {node: self._shorten_road_name(node) 
                          for node in self.network.nodes() 
                          if self.marginals_dict.get(node, 0) > 0.2}
        
        nx.draw_networkx_labels(
            self.network, pos, high_prob_nodes, ax=ax,
            font_size=6, font_color='darkred', font_weight='bold'
        )
        
        ax.set_aspect('equal')
        ax.axis('off')
        
        # 添加TOP-5高概率节点
        top_nodes = sorted([(node, prob) for node, prob in self.marginals_dict.items() 
                           if node in self.network.nodes()], key=lambda x: x[1], reverse=True)[:5]
        
        info_text = "TOP-5 高概率节点:\\n" + "\\n".join([f"{self._shorten_road_name(node)}: {prob:.3f}" 
                                                     for node, prob in top_nodes])
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=7, 
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    def _plot_centrality_network(self, ax):
        """绘制按度中心性着色的网络"""
        ax.set_title("网络中心性 (入度)", fontweight='bold')
        
        pos = self._get_node_positions()
        
        # 计算入度中心性
        in_degrees = dict(self.network.in_degree())
        max_in_degree = max(in_degrees.values()) if in_degrees else 1
        
        node_centralities = [in_degrees.get(node, 0) for node in self.network.nodes()]
        node_sizes = [300 + 20 * degree for degree in node_centralities]
        
        # 绘制边
        nx.draw_networkx_edges(
            self.network, pos, ax=ax,
            edge_color='lightgray',
            arrows=True,
            arrowsize=10,
            arrowstyle='->',
            width=0.5,
            alpha=0.4
        )
        
        # 绘制节点（按中心性着色）
        nodes = nx.draw_networkx_nodes(
            self.network, pos, ax=ax,
            node_color=node_centralities,
            node_size=node_sizes,
            cmap='Blues',
            vmin=0,
            vmax=max_in_degree,
            alpha=0.8
        )
        
        # 添加颜色条
        if nodes:
            cbar = plt.colorbar(nodes, ax=ax, shrink=0.6)
            cbar.set_label('入度', fontsize=8)
        
        # 添加高中心性节点标签
        high_centrality_nodes = {node: self._shorten_road_name(node) 
                               for node in self.network.nodes() 
                               if in_degrees.get(node, 0) > max_in_degree * 0.5}
        
        nx.draw_networkx_labels(
            self.network, pos, high_centrality_nodes, ax=ax,
            font_size=6, font_color='darkblue', font_weight='bold'
        )
        
        ax.set_aspect('equal')
        ax.axis('off')
        
        # 添加TOP-5高中心性节点
        top_central = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
        info_text = "TOP-5 高入度节点:\\n" + "\\n".join([f"{self._shorten_road_name(node)}: {degree}" 
                                                      for node, degree in top_central])
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=7, 
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    def _plot_edge_weights_network(self, ax):
        """绘制边权重网络"""
        ax.set_title("条件概率强度", fontweight='bold')
        
        pos = self._get_node_positions()
        
        # 获取边权重
        edge_weights = []
        edge_colors = []
        
        for u, v, data in self.network.edges(data=True):
            weight = data.get('weight', 0)
            edge_weights.append(weight * 5)  # 缩放宽度
            edge_colors.append(weight)
        
        # 绘制边（按权重着色和调整宽度）
        if edge_weights:
            # 创建边的集合对象以支持颜色条
            edge_collection = nx.draw_networkx_edges(
                self.network, pos, ax=ax,
                edge_color=edge_colors,
                edge_cmap=plt.cm.viridis,
                width=edge_weights,
                arrows=True,
                arrowsize=15,
                arrowstyle='->',
                alpha=0.7
            )
            
            # 添加边权重颜色条
            if edge_collection is not None:
                try:
                    cbar = plt.colorbar(edge_collection, ax=ax, shrink=0.6)
                    cbar.set_label('条件概率', fontsize=8)
                except:
                    # 如果颜色条创建失败，跳过
                    pass
        
        # 绘制节点
        nx.draw_networkx_nodes(
            self.network, pos, ax=ax,
            node_color='lightcoral',
            node_size=200,
            alpha=0.6
        )
        
        # 添加关键节点标签
        high_prob_nodes = {node: self._shorten_road_name(node) 
                          for node in self.network.nodes() 
                          if self.marginals_dict.get(node, 0) > 0.3}
        
        nx.draw_networkx_labels(
            self.network, pos, high_prob_nodes, ax=ax,
            font_size=6, font_color='darkred', font_weight='bold'
        )
        
        ax.set_aspect('equal')
        ax.axis('off')
        
        # 添加边权统计
        if edge_colors:
            info_text = f"边权重统计:\\n平均: {np.mean(edge_colors):.3f}\\n最大: {np.max(edge_colors):.3f}\\n最小: {np.min(edge_colors):.3f}"
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=7, 
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    def _shorten_road_name(self, road_name, max_length=10):
        """缩短道路名称以便显示"""
        if len(road_name) <= max_length:
            return road_name
        
        # 尝试去掉常见后缀
        suffixes = ['_ST', '_AVE', '_RD', '_DR', '_BLVD', '_PKWY']
        for suffix in suffixes:
            if road_name.endswith(suffix):
                base_name = road_name[:-len(suffix)]
                if len(base_name) <= max_length:
                    return base_name
        
        # 如果还是太长，截断
        return road_name[:max_length-2] + ".."
    
    def create_detailed_subgraph(self, central_nodes=None, save_path="network_detail.png", figsize=(12, 10)):
        """创建详细的子图，专注于核心节点"""
        if central_nodes is None:
            # 自动选择高度中心性和高概率的节点
            in_degrees = dict(self.network.in_degree())
            central_nodes = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:8]
            central_nodes = [node for node, _ in central_nodes]
        
        # 创建子图（包含这些节点及其邻居）
        subgraph_nodes = set(central_nodes)
        for node in central_nodes:
            subgraph_nodes.update(self.network.neighbors(node))
            subgraph_nodes.update(self.network.predecessors(node))
        
        subgraph = self.network.subgraph(subgraph_nodes)
        
        print(f"\\n🔍 创建详细子图可视化")
        print(f"   核心节点: {len(central_nodes)}个")
        print(f"   子图节点: {len(subgraph_nodes)}个")
        print(f"   子图边数: {subgraph.number_of_edges()}条")
        
        # 创建可视化
        fig, ax = plt.subplots(figsize=figsize)
        
        # 使用更好的布局
        pos = nx.spring_layout(subgraph, k=2, iterations=50, seed=RANDOM_SEED)
        
        # 分别绘制核心节点和普通节点
        core_nodes = [node for node in subgraph.nodes() if node in central_nodes]
        other_nodes = [node for node in subgraph.nodes() if node not in central_nodes]
        
        # 绘制边
        edge_weights = [subgraph[u][v].get('weight', 0.5) * 3 for u, v in subgraph.edges()]
        edge_colors = [subgraph[u][v].get('weight', 0.5) for u, v in subgraph.edges()]
        
        if edge_weights:
            edge_collection = nx.draw_networkx_edges(
                subgraph, pos, ax=ax,
                edge_color=edge_colors,
                edge_cmap=plt.cm.plasma,
                width=edge_weights,
                arrows=True,
                arrowsize=20,
                arrowstyle='->',
                alpha=0.6
            )
        
        # 绘制核心节点
        if core_nodes:
            core_probs = [self.marginals_dict.get(node, 0) for node in core_nodes]
            core_sizes = [800 + 1200 * prob for prob in core_probs]
            
            nx.draw_networkx_nodes(
                subgraph, pos, nodelist=core_nodes, ax=ax,
                node_color=core_probs,
                node_size=core_sizes,
                cmap='Reds',
                alpha=0.9,
                edgecolors='darkred',
                linewidths=2
            )
        
        # 绘制其他节点
        if other_nodes:
            other_probs = [self.marginals_dict.get(node, 0) for node in other_nodes]
            other_sizes = [400 + 600 * prob for prob in other_probs]
            
            nx.draw_networkx_nodes(
                subgraph, pos, nodelist=other_nodes, ax=ax,
                node_color=other_probs,
                node_size=other_sizes,
                cmap='Blues',
                alpha=0.7,
                edgecolors='darkblue',
                linewidths=1
            )
        
        # 添加标签
        labels = {node: node.replace('_', '\\n') for node in subgraph.nodes()}
        nx.draw_networkx_labels(
            subgraph, pos, labels, ax=ax,
            font_size=8, font_weight='bold'
        )
        
        ax.set_title('Charleston洪水网络 - 核心区域详图', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # 添加图例
        legend_text = "节点说明:\\n• 红色: 核心高影响节点\\n• 蓝色: 相关节点\\n• 节点大小: 洪水概率\\n• 边粗细: 条件概率"
        ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, fontsize=9, 
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 详细网络图已保存至: {save_path}")
        plt.show()
        
        return fig
    
    def analyze_network_properties(self):
        """分析网络属性"""
        print(f"\\n📊 网络属性详细分析")
        print("=" * 50)
        
        # 基础统计
        print(f"🔢 基础统计:")
        print(f"   节点数: {self.network.number_of_nodes()}")
        print(f"   边数: {self.network.number_of_edges()}")
        print(f"   是否为DAG: {nx.is_directed_acyclic_graph(self.network)}")
        print(f"   是否连通: {nx.is_weakly_connected(self.network)}")
        
        # 度分布
        in_degrees = [d for n, d in self.network.in_degree()]
        out_degrees = [d for n, d in self.network.out_degree()]
        
        print(f"\\n📈 度分布:")
        print(f"   平均入度: {np.mean(in_degrees):.2f} ± {np.std(in_degrees):.2f}")
        print(f"   平均出度: {np.mean(out_degrees):.2f} ± {np.std(out_degrees):.2f}")
        print(f"   最大入度: {max(in_degrees)}")
        print(f"   最大出度: {max(out_degrees)}")
        
        # 边权分析
        edge_weights = [d['weight'] for u, v, d in self.network.edges(data=True)]
        if edge_weights:
            print(f"\\n🔗 边权分析:")
            print(f"   平均权重: {np.mean(edge_weights):.3f} ± {np.std(edge_weights):.3f}")
            print(f"   权重范围: {min(edge_weights):.3f} - {max(edge_weights):.3f}")
            print(f"   高权重边数 (>0.8): {sum(1 for w in edge_weights if w > 0.8)}")
        
        # 路径分析
        try:
            avg_path_length = nx.average_shortest_path_length(self.network.to_undirected())
            print(f"\\n🛤️  路径分析:")
            print(f"   平均最短路径长度: {avg_path_length:.2f}")
        except:
            print(f"\\n🛤️  路径分析: 网络不连通，无法计算平均路径长度")
        
        # 中心性分析
        degree_centrality = nx.degree_centrality(self.network)
        top_central = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        
        print(f"\\n🎯 中心性分析 (TOP-5):")
        for i, (node, centrality) in enumerate(top_central, 1):
            prob = self.marginals_dict.get(node, 0)
            print(f"   {i}. {node}: 中心性={centrality:.3f}, P(洪水)={prob:.3f}")

def load_trained_network():
    """加载训练好的网络"""
    # 加载数据
    df = pd.read_csv("Road_Closures_2024.csv")
    df = df[df["REASON"].str.upper() == "FLOOD"].copy()
    
    # 预处理
    df["time_create"] = pd.to_datetime(df["START"], utc=True)
    df["link_id"] = df["STREET"].str.upper().str.replace(" ", "_")
    df["link_id"] = df["link_id"].astype(str)
    df["id"] = df["OBJECTID"].astype(str)
    
    # 时序分割
    df_sorted = df.sort_values('time_create')
    split_idx = int(len(df_sorted) * 0.7)
    train_df = df_sorted.iloc[:split_idx].copy()
    
    # 构建网络
    flood_net = FloodBayesNetwork(t_window="D")
    flood_net.fit_marginal(train_df)
    flood_net.build_network_by_co_occurrence(
        train_df, occ_thr=3, edge_thr=2, weight_thr=0.3, report=False
    )
    flood_net.fit_conditional(train_df, max_parents=2, alpha=1.0)
    flood_net.build_bayes_network()
    
    return flood_net

def main():
    """主函数"""
    print("🎨 贝叶斯网络可视化分析")
    print("=" * 60)
    
    # 加载网络
    flood_net = load_trained_network()
    
    # 创建可视化器
    visualizer = NetworkVisualizer(flood_net)
    
    # 网络属性分析
    visualizer.analyze_network_properties()
    
    # 创建综合可视化
    fig1 = visualizer.create_network_visualization()
    
    # 创建详细子图
    fig2 = visualizer.create_detailed_subgraph()
    
    return visualizer

if __name__ == "__main__":
    visualizer = main()