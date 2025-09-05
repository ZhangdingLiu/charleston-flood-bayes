#!/usr/bin/env python3
"""
构建手动选定4条道路的极小贝叶斯网络

功能：
1. 基于共现分析结果，选择4条最强关联道路
2. 手动定义网络结构：BEE ST → SMITH ST, E BAY ST → VANDERHORST ST
3. 拟合条件概率表(CPT)并保存模型
4. 输出网络统计和条件概率

选定道路：
- BEE ST (父节点)
- SMITH ST (BEE ST的子节点) 
- E BAY ST (父节点)
- VANDERHORST ST (E BAY ST的子节点)

网络结构：
   BEE ST → SMITH ST
   E BAY ST → VANDERHORST ST

用法：
    python build_manual_bn.py

输出：
    - manual_bn.pkl - 训练好的贝叶斯网络模型
    - 终端输出网络结构和条件概率表
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

class ManualBayesianNetworkBuilder:
    """手动贝叶斯网络构建器"""
    
    def __init__(self, data_csv_path="Road_Closures_2024.csv"):
        self.data_csv_path = data_csv_path
        
        # 手动选定的4条道路（基于共现分析结果）
        self.selected_roads = ['BEE ST', 'SMITH ST', 'E BAY ST', 'VANDERHORST ST']
        
        # 手动定义网络结构
        self.network_edges = [
            ('BEE ST', 'SMITH ST'),        # 提升度7.30的最强关联
            ('E BAY ST', 'VANDERHORST ST') # 提升度7.15的次强关联
        ]
        
        # 数据存储
        self.train_df = None
        self.test_df = None
        self.filtered_data = None
        self.bayesian_network = None
        
    def load_and_split_data(self):
        """加载数据并分割（与main_clean.py保持一致）"""
        print("1. 加载和分割数据...")
        
        # 加载数据
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
        
        print(f"   总洪水记录: {len(df)}")
        print(f"   训练集: {len(self.train_df)}条")
        print(f"   测试集: {len(self.test_df)}条")
        
        return self.train_df, self.test_df
        
    def filter_selected_roads(self):
        """过滤出选定道路的数据"""
        print("2. 过滤选定道路数据...")
        
        # 过滤训练集
        train_filtered = self.train_df[self.train_df['road'].isin(self.selected_roads)].copy()
        test_filtered = self.test_df[self.test_df['road'].isin(self.selected_roads)].copy()
        
        print(f"   选定道路: {self.selected_roads}")
        print(f"   训练集过滤后: {len(train_filtered)}条记录")
        print(f"   测试集过滤后: {len(test_filtered)}条记录")
        
        # 统计每条道路的出现次数
        print("   各道路在训练集中的出现次数:")
        for road in self.selected_roads:
            count = len(train_filtered[train_filtered['road'] == road])
            print(f"     {road}: {count}次")
        
        self.filtered_data = {
            'train': train_filtered,
            'test': test_filtered
        }
        
        return train_filtered
        
    def create_binary_matrix(self):
        """创建日期-道路二元矩阵用于拟合"""
        print("3. 创建二元矩阵...")
        
        train_data = self.filtered_data['train']
        
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
        print("   各道路洪水频率:")
        for road in self.selected_roads:
            col_name = road.replace(' ', '_').replace('-', '_')
            freq = binary_df[col_name].mean()
            total = binary_df[col_name].sum()
            print(f"     {road}: {freq:.3f} ({total}/{len(binary_df)})")
        
        return binary_df
        
    def build_bayesian_network(self, binary_df):
        """构建贝叶斯网络"""
        print("4. 构建贝叶斯网络...")
        
        # 创建节点名（合法的变量名）
        node_mapping = {}
        reverse_mapping = {}
        for road in self.selected_roads:
            node_name = road.replace(' ', '_').replace('-', '_')
            node_mapping[road] = node_name
            reverse_mapping[node_name] = road
        
        # 转换边定义
        edges = []
        for parent_road, child_road in self.network_edges:
            parent_node = node_mapping[parent_road]
            child_node = node_mapping[child_road]
            edges.append((parent_node, child_node))
        
        print(f"   网络结构:")
        for parent_road, child_road in self.network_edges:
            print(f"     {parent_road} → {child_road}")
        
        # 创建贝叶斯网络
        self.bayesian_network = BayesianNetwork(edges)
        
        # 准备数据（只保留节点列）
        node_columns = [node_mapping[road] for road in self.selected_roads]
        training_data = binary_df[node_columns].copy()
        
        print(f"   节点数: {len(self.bayesian_network.nodes())}")
        print(f"   边数: {len(self.bayesian_network.edges())}")
        
        # 拟合参数
        estimator = MaximumLikelihoodEstimator(self.bayesian_network, training_data)
        
        # 添加拉普拉斯平滑
        cpds = []
        for node in self.bayesian_network.nodes():
            parents = list(self.bayesian_network.predecessors(node))
            
            if len(parents) == 0:
                # 根节点 - 计算先验概率
                prob_1 = (training_data[node].sum() + 1) / (len(training_data) + 2)  # 拉普拉斯平滑
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
                total_parent_0 = parent_0_child_0 + parent_0_child_1 + 2  # +2 for smoothing
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
        print("📊 手动贝叶斯网络统计")
        print("="*60)
        
        print(f"节点数: {len(self.bayesian_network.nodes())}")
        print(f"边数: {len(self.bayesian_network.edges())}")
        
        print(f"\n网络结构:")
        for parent_road, child_road in self.network_edges:
            print(f"  {parent_road} → {child_road}")
        
        print(f"\n条件概率表 (CPTs):")
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
                    # 处理一维数组的情况
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
        
    def save_model(self, model_path="manual_bn.pkl"):
        """保存贝叶斯网络模型"""
        print("5. 保存模型...")
        
        # 保存模型和元数据
        model_data = {
            'bayesian_network': self.bayesian_network,
            'selected_roads': self.selected_roads,
            'network_edges': self.network_edges,
            'node_mapping': {road: road.replace(' ', '_').replace('-', '_') for road in self.selected_roads}
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"   ✅ 模型保存到: {model_path}")
        return model_path
        
    def build_complete_network(self):
        """构建完整的贝叶斯网络"""
        print("🚀 构建手动选定道路的极小贝叶斯网络...")
        print("="*60)
        
        try:
            # 1. 数据加载和分割
            self.load_and_split_data()
            
            # 2. 过滤选定道路
            self.filter_selected_roads()
            
            # 3. 创建二元矩阵
            binary_df = self.create_binary_matrix()
            
            # 4. 构建贝叶斯网络
            network, reverse_mapping = self.build_bayesian_network(binary_df)
            
            # 5. 打印统计
            self.print_network_statistics(reverse_mapping)
            
            # 6. 保存模型
            self.save_model()
            
            print(f"\n✅ 极小贝叶斯网络构建完成！")
            print(f"📁 模型文件: manual_bn.pkl")
            
            return network
            
        except Exception as e:
            print(f"\n❌ 构建过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """主函数"""
    builder = ManualBayesianNetworkBuilder()
    network = builder.build_complete_network()
    
    if network is not None:
        print(f"\n🎯 网络特点:")
        print(f"   - 基于最强共现关联的4条道路")
        print(f"   - 简单的链式结构，易于解释")
        print(f"   - 快速推理，适合实时应用")
        print(f"   - 体现了BEE ST→SMITH ST和E BAY ST→VANDERHORST ST的因果关系")

if __name__ == "__main__":
    main()