#!/usr/bin/env python3
"""
基于训练集统计构建贝叶斯网络

功能：
1. 读取训练集共现分析结果(train_cooccurrence_stats.csv)
2. 自动选择条件概率最高的前3对道路
3. 构建有向贝叶斯网络，确保无环
4. 基于训练集数据拟合条件概率表(CPT)
5. 保存训练好的模型用于测试集评估

网络构建策略：
- 使用confA2B和confB2A确定边的方向
- 优先选择条件概率高的方向
- 检测并避免循环依赖
- 支持星型、链型、树型等拓扑结构

用法：
    python build_train_based_bn.py

输出：
    - train_based_bn.pkl - 基于训练集的贝叶斯网络模型
    - 终端输出网络结构和条件概率表分析
"""

import pandas as pd
import numpy as np
import pickle
import os
from collections import defaultdict
from sklearn.model_selection import train_test_split

# 贝叶斯网络相关
try:
    try:
        from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
    except ImportError:
        from pgmpy.models import BayesianNetwork
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.estimators import MaximumLikelihoodEstimator
except ImportError:
    print("请安装pgmpy: pip install pgmpy")
    exit(1)

# 设置随机种子
RANDOM_SEED = 42

class TrainBasedBayesianNetworkBuilder:
    """基于训练集统计的贝叶斯网络构建器"""
    
    def __init__(self, 
                 cooccurrence_file="results/train_cooccurrence_stats.csv",
                 data_csv_path="Road_Closures_2024.csv",
                 top_k=3):
        self.cooccurrence_file = cooccurrence_file
        self.data_csv_path = data_csv_path
        self.top_k = top_k
        
        # 数据存储
        self.train_df = None
        self.test_df = None
        self.cooccurrence_stats = None
        self.selected_pairs = []
        self.selected_roads = []
        self.network_edges = []
        self.bayesian_network = None
        self.node_mapping = {}
        
    def load_data(self):
        """加载数据并分割（保持与分析脚本一致）"""
        print("1. 加载和分割数据...")
        
        # 加载原始数据
        df = pd.read_csv(self.data_csv_path)
        df = df[df["REASON"].str.upper() == "FLOOD"].copy()
        
        # 数据预处理
        df["time_create"] = pd.to_datetime(df["START"], utc=True)
        df["road"] = df["STREET"].str.upper().str.strip()
        df["date"] = df["time_create"].dt.floor("D")
        df["id"] = df["OBJECTID"].astype(str)
        
        # 使用相同的随机种子分割
        self.train_df, self.test_df = train_test_split(
            df, test_size=0.3, random_state=RANDOM_SEED
        )
        
        print(f"   训练集: {len(self.train_df)}条")
        print(f"   测试集: {len(self.test_df)}条")
        
        return self.train_df, self.test_df
        
    def load_cooccurrence_stats(self):
        """加载训练集共现统计"""
        print("2. 加载共现统计...")
        
        try:
            self.cooccurrence_stats = pd.read_csv(self.cooccurrence_file)
            print(f"   ✅ 加载{len(self.cooccurrence_stats)}对道路共现统计")
            
            # 显示前几行
            print(f"   前5对道路统计:")
            for i, row in self.cooccurrence_stats.head().iterrows():
                print(f"     {row['road_a']} → {row['road_b']}: "
                      f"confA2B={row['conf_a2b']:.4f}, confB2A={row['conf_b2a']:.4f}")
            
            return True
            
        except FileNotFoundError:
            print(f"   ❌ 找不到共现统计文件: {self.cooccurrence_file}")
            print("   请先运行 train_cooccurrence_analysis.py")
            return False
        except Exception as e:
            print(f"   ❌ 加载统计文件时出错: {e}")
            return False
            
    def select_network_structure(self):
        """选择网络结构并检测环路"""
        print("3. 选择网络结构...")
        
        if self.cooccurrence_stats is None or len(self.cooccurrence_stats) == 0:
            print("   ❌ 没有可用的共现统计")
            return False
        
        # 按最大条件概率排序
        stats_sorted = self.cooccurrence_stats.sort_values('max_conf', ascending=False)
        
        # 贪婪算法选择不产生环路的边
        selected_edges = []
        used_roads = set()
        
        print(f"   候选道路对 (top-{min(10, len(stats_sorted))}):")
        print(f"   {'序号':<4} {'道路A':<15} {'道路B':<15} {'confA2B':<8} {'confB2A':<8} {'状态'}")
        print("-" * 70)
        
        for i, (idx, row) in enumerate(stats_sorted.head(10).iterrows()):
            road_a, road_b = row['road_a'], row['road_b']
            conf_a2b, conf_b2a = row['conf_a2b'], row['conf_b2a']
            
            # 确定边的方向
            if conf_a2b >= conf_b2a:
                parent, child = road_a, road_b
                conf = conf_a2b
            else:
                parent, child = road_b, road_a
                conf = conf_b2a
            
            # 检查是否会产生环路（简单检查：避免互相指向）
            reverse_edge = (child, parent)
            has_cycle = any(edge[:2] == reverse_edge for edge in selected_edges)
            
            status = "选中" if len(selected_edges) < self.top_k and not has_cycle else "跳过"
            print(f"   {i+1:<4} {road_a:<15} {road_b:<15} {conf_a2b:<8.4f} {conf_b2a:<8.4f} {status}")
            
            # 选择前top_k个不产生环路的边
            if len(selected_edges) < self.top_k and not has_cycle:
                selected_edges.append((parent, child, conf))
                used_roads.add(parent)
                used_roads.add(child)
        
        if len(selected_edges) == 0:
            print("   ❌ 没有找到有效的边")
            return False
        
        self.network_edges = selected_edges
        self.selected_roads = list(used_roads)
        
        print(f"\n   选定的网络结构:")
        print(f"   节点数: {len(self.selected_roads)}")
        print(f"   边数: {len(self.network_edges)}")
        for parent, child, conf in self.network_edges:
            print(f"     {parent} → {child} (条件概率: {conf:.4f})")
        
        return True
        
    def create_binary_matrix(self):
        """创建选定道路的日期-道路二元矩阵"""
        print("4. 创建二元矩阵...")
        
        train_data = self.train_df[self.train_df['road'].isin(self.selected_roads)].copy()
        
        # 按日期分组，创建道路出现的二元矩阵
        daily_roads = defaultdict(set)
        for _, row in train_data.iterrows():
            daily_roads[str(row['date'].date())].add(row['road'])
        
        # 创建DataFrame
        dates = sorted(daily_roads.keys())
        matrix_data = []
        
        for date in dates:
            row_data = {'date': date}
            roads_today = daily_roads[date]
            
            for road in self.selected_roads:
                # 将道路名转换为合法的列名
                col_name = road.replace(' ', '_').replace('-', '_')
                row_data[col_name] = 1 if road in roads_today else 0
            
            matrix_data.append(row_data)
        
        binary_df = pd.DataFrame(matrix_data)
        
        print(f"   二元矩阵大小: {len(binary_df)} 天 × {len(self.selected_roads)} 道路")
        print(f"   日期范围: {min(dates)} 至 {max(dates)}")
        
        # 显示各道路的洪水频率
        print(f"   各道路洪水频率:")
        for road in self.selected_roads:
            col_name = road.replace(' ', '_').replace('-', '_')
            freq = binary_df[col_name].mean()
            total = binary_df[col_name].sum()
            print(f"     {road}: {freq:.3f} ({total}/{len(binary_df)})")
        
        return binary_df
        
    def build_bayesian_network(self, binary_df):
        """构建贝叶斯网络"""
        print("5. 构建贝叶斯网络...")
        
        # 创建节点名映射（合法的变量名）
        for road in self.selected_roads:
            node_name = road.replace(' ', '_').replace('-', '_')
            self.node_mapping[road] = node_name
        
        reverse_mapping = {v: k for k, v in self.node_mapping.items()}
        
        # 转换边定义
        edges = []
        for parent_road, child_road, conf in self.network_edges:
            parent_node = self.node_mapping[parent_road]
            child_node = self.node_mapping[child_road]
            edges.append((parent_node, child_node))
        
        print(f"   网络结构:")
        for parent_road, child_road, conf in self.network_edges:
            print(f"     {parent_road} → {child_road} (概率: {conf:.4f})")
        
        # 创建贝叶斯网络
        self.bayesian_network = BayesianNetwork(edges)
        
        # 准备数据（只保留节点列）
        node_columns = [self.node_mapping[road] for road in self.selected_roads]
        training_data = binary_df[node_columns].copy()
        
        print(f"   节点数: {len(self.bayesian_network.nodes())}")
        print(f"   边数: {len(self.bayesian_network.edges())}")
        
        # 手动计算CPD参数（包含拉普拉斯平滑）
        cpds = []
        for node in self.bayesian_network.nodes():
            parents = list(self.bayesian_network.predecessors(node))
            
            if len(parents) == 0:
                # 根节点 - 计算先验概率
                prob_1 = (training_data[node].sum() + 1) / (len(training_data) + 2)
                prob_0 = 1 - prob_1
                
                cpd = TabularCPD(
                    variable=node,
                    variable_card=2,
                    values=[[prob_0], [prob_1]],
                    state_names={node: [0, 1]}
                )
                
            else:
                # 有父节点的节点
                parent = parents[0]  # 假设最多1个父节点
                
                # 计算条件概率 P(child|parent)
                parent_0_child_0 = len(training_data[(training_data[parent] == 0) & (training_data[node] == 0)])
                parent_0_child_1 = len(training_data[(training_data[parent] == 0) & (training_data[node] == 1)])
                parent_1_child_0 = len(training_data[(training_data[parent] == 1) & (training_data[node] == 0)])
                parent_1_child_1 = len(training_data[(training_data[parent] == 1) & (training_data[node] == 1)])
                
                # 拉普拉斯平滑
                total_parent_0 = parent_0_child_0 + parent_0_child_1 + 2
                total_parent_1 = parent_1_child_0 + parent_1_child_1 + 2
                
                prob_child_0_given_parent_0 = (parent_0_child_0 + 1) / total_parent_0
                prob_child_1_given_parent_0 = (parent_0_child_1 + 1) / total_parent_0
                prob_child_0_given_parent_1 = (parent_1_child_0 + 1) / total_parent_1
                prob_child_1_given_parent_1 = (parent_1_child_1 + 1) / total_parent_1
                
                cpd = TabularCPD(
                    variable=node,
                    variable_card=2,
                    values=[
                        [prob_child_0_given_parent_0, prob_child_0_given_parent_1],
                        [prob_child_1_given_parent_0, prob_child_1_given_parent_1]
                    ],
                    evidence=[parent],
                    evidence_card=[2],
                    state_names={node: [0, 1], parent: [0, 1]}
                )
            
            cpds.append(cpd)
        
        # 将CPD添加到网络
        self.bayesian_network.add_cpds(*cpds)
        
        # 验证网络
        assert self.bayesian_network.check_model()
        print("   ✅ 贝叶斯网络构建并验证成功")
        
        return self.bayesian_network, reverse_mapping
        
    def print_network_statistics(self, reverse_mapping):
        """打印网络统计和条件概率表"""
        print("\n" + "="*60)
        print("📊 基于训练集的贝叶斯网络统计")
        print("="*60)
        
        print(f"节点数: {len(self.bayesian_network.nodes())}")
        print(f"边数: {len(self.bayesian_network.edges())}")
        
        print(f"\n网络结构:")
        for parent_road, child_road, conf in self.network_edges:
            print(f"  {parent_road} → {child_road} (训练集条件概率: {conf:.4f})")
        
        print(f"\n贝叶斯网络条件概率表 (CPTs):")
        print("-" * 60)
        
        for cpd in self.bayesian_network.get_cpds():
            node_name = cpd.variable
            road_name = reverse_mapping[node_name]
            
            print(f"\n🔸 {road_name} ({node_name}):")
            
            parents = list(self.bayesian_network.predecessors(node_name))
            if len(parents) == 0:
                # 根节点
                values = cpd.values
                if values.ndim == 1:
                    prob_no_flood, prob_flood = values[0], values[1]
                else:
                    prob_flood = values[1, 0]  # P(flood=1)
                    prob_no_flood = values[0, 0]  # P(flood=0)
                print(f"   P({road_name}=洪水) = {prob_flood:.4f}")
                print(f"   P({road_name}=无洪水) = {prob_no_flood:.4f}")
            else:
                # 有父节点
                parent_node = parents[0]
                parent_road = reverse_mapping[parent_node]
                
                values = cpd.values
                if values.ndim == 1:
                    prob_flood_given_no_parent = values[1] if len(values) > 1 else 0.5
                    prob_flood_given_parent = values[1] if len(values) > 1 else 0.5
                else:
                    # P(child=1|parent=0)
                    prob_flood_given_no_parent = values[1, 0]
                    # P(child=1|parent=1) 
                    prob_flood_given_parent = values[1, 1]
                
                print(f"   P({road_name}=洪水 | {parent_road}=无洪水) = {prob_flood_given_no_parent:.4f}")
                print(f"   P({road_name}=洪水 | {parent_road}=洪水) = {prob_flood_given_parent:.4f}")
                
                # 计算提升效果
                lift = prob_flood_given_parent / prob_flood_given_no_parent if prob_flood_given_no_parent > 0 else float('inf')
                print(f"   条件概率提升: {lift:.2f}x")
        
        print("="*60)
        
    def save_model(self, model_path="train_based_bn.pkl"):
        """保存贝叶斯网络模型"""
        print("6. 保存模型...")
        
        # 保存模型和元数据
        model_data = {
            'bayesian_network': self.bayesian_network,
            'selected_roads': self.selected_roads,
            'network_edges': self.network_edges,
            'node_mapping': self.node_mapping,
            'cooccurrence_source': self.cooccurrence_file
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"   ✅ 模型保存到: {model_path}")
        return model_path
        
    def build_complete_network(self):
        """构建完整的贝叶斯网络"""
        print("🚀 构建基于训练集统计的贝叶斯网络...")
        print("="*60)
        
        try:
            # 1. 数据加载
            self.load_data()
            
            # 2. 加载共现统计
            if not self.load_cooccurrence_stats():
                return None
            
            # 3. 选择网络结构
            if not self.select_network_structure():
                return None
            
            # 4. 创建二元矩阵
            binary_df = self.create_binary_matrix()
            
            # 5. 构建贝叶斯网络
            network, reverse_mapping = self.build_bayesian_network(binary_df)
            
            # 6. 打印统计
            self.print_network_statistics(reverse_mapping)
            
            # 7. 保存模型
            self.save_model()
            
            print(f"\n✅ 基于训练集的贝叶斯网络构建完成！")
            print(f"📁 模型文件: train_based_bn.pkl")
            
            return network
            
        except Exception as e:
            print(f"\n❌ 构建过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """主函数"""
    builder = TrainBasedBayesianNetworkBuilder()
    network = builder.build_complete_network()
    
    if network is not None:
        print(f"\n🎯 网络特点:")
        print(f"   - 基于训练集真实共现模式构建")
        print(f"   - 避免了数据泄露问题")
        print(f"   - 选择条件概率最高的道路对")
        print(f"   - 网络结构简洁且可解释")
        print(f"   - 为测试集评估提供可靠基础")

if __name__ == "__main__":
    main()