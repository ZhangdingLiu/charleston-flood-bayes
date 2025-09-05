# -*- coding: utf-8 -*-
"""
Created on 6/10/2025 4:02 PM

@author: zliu952
"""

# visualization.py

import matplotlib.pyplot as plt
import networkx as nx
import os


def visualize_flood_network(graph: nx.DiGraph, save_path: str = None, figsize=(15, 12)):
    """
    可视化洪水贝叶斯网络结构。
    :param graph: nx.DiGraph, 包含节点和边的加权图
    :param save_path: 保存图像的路径，默认不保存
    :param figsize: 图像尺寸
    """
    plt.figure(figsize=figsize)

    # 使用 spring layout （可选改为其他布局）
    pos = nx.spring_layout(graph, k=1, seed=42)

    # 绘制节点
    nx.draw_networkx_nodes(graph, pos, node_size=300, node_color='lightblue', edgecolors='black')

    # 绘制边
    nx.draw_networkx_edges(graph, pos, arrowstyle='->', arrowsize=15, edge_color='gray')

    # 绘制节点标签
    nx.draw_networkx_labels(graph, pos, font_size=5)

    # 绘制边权重（概率）
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    edge_labels = {k: f"{v:.2f}" for k, v in edge_labels.items()}  # 保留两位小数
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=7)

    plt.axis('off')
    plt.title("Bayesian Flood Network", fontsize=14)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Network saved to {save_path}")
    else:
        plt.show()


def plot_edge_weight_distribution(graph, save_path=None):
    """
    Plot histogram of edge weights in the network.
    """
    weights = [d["weight"] for _, _, d in graph.edges(data=True)]
    plt.figure(figsize=(8, 5))
    plt.hist(weights, bins=30, color='orange', edgecolor='black', alpha=0.8)
    plt.title("Distribution of Edge Weights", fontsize=14)
    plt.xlabel("Edge Weight", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Edge weight histogram saved to {save_path}")
    else:
        plt.show()

import matplotlib.pyplot as plt

def visualize_edge_stats(edge_stats, save_dir="figs/edge_stats"):
    """
    Visualizes the distribution of edge weights, co-occurrence counts, and occurrence counts.

    Parameters:
    - edge_stats: List of (source, target, weight, count, occurrence_a)
    - save_dir: Directory to save the figures
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    weights = [w for _, _, w, _, _ in edge_stats]
    counts = [c for _, _, _, c, _ in edge_stats]
    occurrences = [o for _, _, _, _, o in edge_stats]

    # Plot 1: Edge weight distribution
    plt.figure(figsize=(6, 4))
    plt.hist(weights, bins=30, color='skyblue', edgecolor='black')
    plt.title("Distribution of Edge Weights")
    plt.xlabel("Weight")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/weight_distribution.png")
    plt.close()

    # Plot 2: Co-occurrence count distribution
    plt.figure(figsize=(6, 4))
    plt.hist(counts, bins=30, color='salmon', edgecolor='black')
    plt.title("Distribution of Co-occurrence Counts")
    plt.xlabel("Count")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/co_occurrence_count_distribution.png")
    plt.close()

    # Plot 3: Source road occurrence distribution
    plt.figure(figsize=(6, 4))
    plt.hist(occurrences, bins=30, color='lightgreen', edgecolor='black')
    plt.title("Distribution of Source Road Occurrences")
    plt.xlabel("Occurrence Count")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/source_occurrence_distribution.png")
    plt.close()

    print(f"✅ Edge statistics visualizations saved to: {save_dir}")
