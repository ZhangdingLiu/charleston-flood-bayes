#!/usr/bin/env python3
"""
构建极简洪水贝叶斯网络

功能：
1. 严格筛选高频核心道路节点 (频次≥15)
2. 使用互信息和共现次数筛选边
3. 应用Chow-Liu算法构建最大生成树
4. 限制节点度数≤2，确保网络简洁
5. 生成ultra_core_network.csv

用法：
    python build_ultra_core.py

输出：
    - ultra_core_network.csv - 极简网络结构
    - 终端输出网络统计信息
"""

import pandas as pd
import numpy as np
import networkx as nx
from collections import Counter, defaultdict
from sklearn.metrics import mutual_info_score
from sklearn.model_selection import train_test_split
import itertools
import random

# 设置随机种子
RANDOM_SEED = 42
random.seed(0)
np.random.seed(0)

class UltraCoreNetworkBuilder:
    """极简核心网络构建器"""
    
    def __init__(self, data_csv_path="Road_Closures_2024.csv"):
        self.data_csv_path = data_csv_path
        self.train_df = None
        self.core_nodes = []
        self.edge_weights = {}
        self.final_network = None
        
    def load_and_split_data(self):
        """加载数据并分割（与main_clean.py保持一致）"""
        print("1. 加载和分割数据...")
        
        # 加载数据
        df = pd.read_csv(self.data_csv_path)
        df = df[df["REASON"].str.upper() == "FLOOD"].copy()
        
        # 数据预处理
        df["time_create"] = pd.to_datetime(df["START"], utc=True)
        df["link_id"] = df["STREET"].str.upper().str.replace(" ", "_")
        df["link_id"] = df["link_id"].astype(str)
        df["id"] = df["OBJECTID"].astype(str)
        
        # 使用相同的随机种子分割
        train_df, test_df = train_test_split(
            df, test_size=0.3, random_state=RANDOM_SEED
        )
        
        self.train_df = train_df
        
        print(f"   训练集: {len(train_df)}条记录")
        print(f"   涉及道路: {train_df['link_id'].nunique()}条")
        
        return train_df
        
    def select_core_nodes(self, min_frequency=15):
        """选择核心节点：频次≥15"""
        print("2. 选择核心节点...")
        
        # 统计道路频次
        road_counts = Counter(self.train_df['link_id'])
        
        # 筛选高频节点
        self.core_nodes = [road for road, count in road_counts.items() 
                          if count >= min_frequency]
        
        print(f"   频次阈值: ≥{min_frequency}")
        print(f"   核心节点数: {len(self.core_nodes)}")
        
        if len(self.core_nodes) == 0:
            raise ValueError(f"没有道路满足频次≥{min_frequency}的条件")
            
        # 显示核心节点及其频次
        print("   核心节点列表:")
        for road in sorted(self.core_nodes):
            count = road_counts[road]
            print(f"     {road.replace('_', ' '):<25} (频次: {count})")
            
        return self.core_nodes
        
    def compute_mutual_information_edges(self, min_cooccurrence=5, min_mi=0.02):
        """计算边：基于共现次数和互信息"""
        print("3. 计算边权重...")
        
        # 创建日期-道路矩阵
        df_filtered = self.train_df[self.train_df['link_id'].isin(self.core_nodes)].copy()
        df_filtered['date'] = df_filtered['time_create'].dt.floor('D')
        
        # 构建道路-日期二元矩阵
        pivot_table = df_filtered.pivot_table(
            index='date', 
            columns='link_id', 
            values='id', 
            aggfunc='count', 
            fill_value=0
        )
        
        # 转换为二元矩阵 (0/1)
        binary_matrix = (pivot_table > 0).astype(int)
        
        # 确保所有核心节点都在矩阵中
        for node in self.core_nodes:
            if node not in binary_matrix.columns:
                binary_matrix[node] = 0
        
        print(f"   时间窗口: {len(binary_matrix)}天")
        print(f"   道路节点: {len(binary_matrix.columns)}个")
        
        # 计算所有道路对的互信息和共现次数
        valid_edges = []
        mi_scores = []
        cooccurrence_counts = []
        
        for road1, road2 in itertools.combinations(self.core_nodes, 2):
            if road1 not in binary_matrix.columns or road2 not in binary_matrix.columns:
                continue
                
            # 获取二元序列
            seq1 = binary_matrix[road1].values
            seq2 = binary_matrix[road2].values
            
            # 计算共现次数
            cooccurrence = np.sum((seq1 == 1) & (seq2 == 1))
            
            # 计算互信息
            if len(set(seq1)) > 1 and len(set(seq2)) > 1:  # 确保不是常数序列
                mi = mutual_info_score(seq1, seq2)
            else:
                mi = 0.0
            
            # 应用筛选条件
            if cooccurrence >= min_cooccurrence and mi >= min_mi:
                valid_edges.append((road1, road2))
                self.edge_weights[(road1, road2)] = {
                    'mutual_info': mi,
                    'cooccurrence': cooccurrence,
                    'weight': mi  # 使用MI作为边权重
                }
                mi_scores.append(mi)
                cooccurrence_counts.append(cooccurrence)
        
        print(f"   共现阈值: ≥{min_cooccurrence}")
        print(f"   互信息阈值: ≥{min_mi}")
        print(f"   有效边数: {len(valid_edges)}")
        
        if len(valid_edges) > 0:
            print(f"   互信息范围: [{min(mi_scores):.4f}, {max(mi_scores):.4f}]")
            print(f"   共现次数范围: [{min(cooccurrence_counts)}, {max(cooccurrence_counts)}]")
        else:
            print("   ⚠️ 没有找到满足条件的边")
            
        return valid_edges
        
    def build_chow_liu_tree(self, valid_edges):
        """使用Chow-Liu算法构建最大生成树"""
        print("4. 构建Chow-Liu最大生成树...")
        
        if len(valid_edges) == 0:
            print("   ⚠️ 没有有效边，创建空网络")
            self.final_network = nx.Graph()
            self.final_network.add_nodes_from(self.core_nodes)
            return self.final_network
        
        # 创建完全图
        G = nx.Graph()
        G.add_nodes_from(self.core_nodes)
        
        # 添加所有有效边（权重为负互信息，因为MST算法寻找最小权重）
        for edge in valid_edges:
            road1, road2 = edge
            mi_weight = self.edge_weights[edge]['weight']
            G.add_edge(road1, road2, weight=-mi_weight, mutual_info=mi_weight,
                      cooccurrence=self.edge_weights[edge]['cooccurrence'])
        
        # 使用Kruskal算法找最大生成树（通过负权重转换为最小生成树）
        if G.number_of_edges() > 0:
            mst = nx.minimum_spanning_tree(G, weight='weight')
            
            # 恢复正权重
            for u, v, data in mst.edges(data=True):
                mst[u][v]['weight'] = -data['weight']
                
            print(f"   生成树节点: {mst.number_of_nodes()}")
            print(f"   生成树边数: {mst.number_of_edges()}")
        else:
            mst = nx.Graph()
            mst.add_nodes_from(self.core_nodes)
            print("   生成空生成树（没有有效边）")
        
        self.final_network = mst
        return mst
        
    def enforce_degree_constraint(self, max_degree=2):
        """限制节点度数≤2"""
        print("5. 限制节点度数...")
        
        if self.final_network.number_of_edges() == 0:
            print("   网络没有边，跳过度数限制")
            return self.final_network
        
        # 检查当前度数分布
        degrees = dict(self.final_network.degree())
        high_degree_nodes = [node for node, degree in degrees.items() if degree > max_degree]
        
        print(f"   最大度数限制: ≤{max_degree}")
        print(f"   超限节点数: {len(high_degree_nodes)}")
        
        if len(high_degree_nodes) == 0:
            print("   所有节点已满足度数限制")
            return self.final_network
        
        # 对每个超限节点，删除互信息最小的边
        edges_removed = 0
        for node in high_degree_nodes:
            while self.final_network.degree(node) > max_degree:
                # 获取该节点的所有邻接边
                incident_edges = list(self.final_network.edges(node, data=True))
                
                if len(incident_edges) <= max_degree:
                    break
                
                # 找到互信息最小的边
                min_edge = min(incident_edges, key=lambda x: x[2]['mutual_info'])
                
                # 删除边
                self.final_network.remove_edge(min_edge[0], min_edge[1])
                edges_removed += 1
                
                print(f"     删除边: {min_edge[0]} - {min_edge[1]} (MI: {min_edge[2]['mutual_info']:.4f})")
        
        print(f"   删除边数: {edges_removed}")
        print(f"   最终边数: {self.final_network.number_of_edges()}")
        
        return self.final_network
        
    def save_network(self, output_path="ultra_core_network.csv"):
        """保存网络到CSV"""
        print("6. 保存网络...")
        
        if self.final_network.number_of_edges() == 0:
            # 如果没有边，创建空CSV
            empty_df = pd.DataFrame(columns=['source', 'target', 'mutual_info', 'cooccurrence', 'weight'])
            empty_df.to_csv(output_path, index=False)
            print(f"   ✅ 空网络保存到: {output_path}")
        else:
            # 保存边信息
            edges_data = []
            for u, v, data in self.final_network.edges(data=True):
                edges_data.append({
                    'source': u,
                    'target': v,
                    'mutual_info': data['mutual_info'],
                    'cooccurrence': data['cooccurrence'],
                    'weight': data['weight']
                })
            
            edges_df = pd.DataFrame(edges_data)
            edges_df.to_csv(output_path, index=False, float_format='%.6f')
            print(f"   ✅ 网络保存到: {output_path}")
        
        return output_path
        
    def print_network_statistics(self):
        """打印网络统计信息"""
        print("\n" + "="*60)
        print("📊 极简核心网络统计")
        print("="*60)
        
        n_nodes = self.final_network.number_of_nodes()
        n_edges = self.final_network.number_of_edges()
        
        print(f"节点数: {n_nodes}")
        print(f"边数: {n_edges}")
        
        if n_edges > 0:
            # 计算平均度
            degrees = [d for n, d in self.final_network.degree()]
            avg_degree = np.mean(degrees)
            max_degree = max(degrees)
            
            print(f"平均度: {avg_degree:.2f}")
            print(f"最大度: {max_degree}")
            
            # 显示度分布
            degree_dist = Counter(degrees)
            print("度分布:")
            for degree in sorted(degree_dist.keys()):
                count = degree_dist[degree]
                print(f"  度{degree}: {count}个节点")
            
            # 显示边信息
            print(f"\n边列表 (按互信息排序):")
            edges_with_data = [(u, v, data) for u, v, data in self.final_network.edges(data=True)]
            edges_with_data.sort(key=lambda x: x[2]['mutual_info'], reverse=True)
            
            for u, v, data in edges_with_data:
                print(f"  {u.replace('_', ' '):<25} - {v.replace('_', ' '):<25} "
                      f"(MI: {data['mutual_info']:.4f}, 共现: {data['cooccurrence']})")
        else:
            print("平均度: 0.00")
            print("最大度: 0")
            print("网络无连接")
        
        # 连通性分析
        if n_edges > 0:
            n_components = nx.number_connected_components(self.final_network)
            largest_cc_size = len(max(nx.connected_components(self.final_network), key=len))
            print(f"\n连通组件数: {n_components}")
            print(f"最大连通组件大小: {largest_cc_size}")
        else:
            print(f"\n连通组件数: {n_nodes} (所有节点孤立)")
            
        print("="*60)
        
    def build_ultra_core_network(self):
        """构建完整的极简核心网络"""
        print("🚀 构建极简洪水贝叶斯网络...")
        print("="*60)
        
        try:
            # 1. 数据加载
            self.load_and_split_data()
            
            # 2. 选择核心节点
            self.select_core_nodes(min_frequency=15)
            
            # 3. 计算边权重
            valid_edges = self.compute_mutual_information_edges(
                min_cooccurrence=5, 
                min_mi=0.02
            )
            
            # 4. 构建Chow-Liu树
            self.build_chow_liu_tree(valid_edges)
            
            # 5. 限制度数
            self.enforce_degree_constraint(max_degree=2)
            
            # 6. 保存网络
            self.save_network("ultra_core_network.csv")
            
            # 7. 打印统计
            self.print_network_statistics()
            
            print(f"\n✅ 极简核心网络构建完成！")
            print(f"📁 网络文件: ultra_core_network.csv")
            
            return self.final_network
            
        except Exception as e:
            print(f"\n❌ 构建过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """主函数"""
    builder = UltraCoreNetworkBuilder()
    network = builder.build_ultra_core_network()
    
    if network is not None:
        print(f"\n🎯 极简网络特点:")
        print(f"   - 高频核心道路 (频次≥15)")
        print(f"   - 强共现关系 (共现≥5, MI≥0.02)")
        print(f"   - 树状结构 (度数≤2)")
        print(f"   - 最优信息量 (Chow-Liu算法)")

if __name__ == "__main__":
    main()