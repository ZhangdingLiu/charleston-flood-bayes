#!/usr/bin/env python3
"""
详细分析：透明化贝叶斯网络构建和评估过程

输出所有中间步骤，让用户能够验证结果的真实性和可靠性：
1. 网络构建过程详细分析
2. 训练数据深度统计  
3. 边际概率和条件概率表展示
4. 数据分布和质量验证
"""

import random
import numpy as np
import pandas as pd
from model import FloodBayesNetwork
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class DetailedNetworkAnalyzer:
    """详细的网络分析器"""
    
    def __init__(self):
        self.train_df = None
        self.test_df = None
        self.flood_net = None
        
    def load_and_analyze_data(self):
        """加载数据并进行详细分析"""
        print("🔍 数据加载和预处理详细分析")
        print("=" * 60)
        
        # 加载原始数据
        df = pd.read_csv("Road_Closures_2024.csv")
        print(f"原始数据: {len(df)}条记录")
        
        # 过滤洪水记录
        flood_df = df[df["REASON"].str.upper() == "FLOOD"].copy()
        print(f"洪水记录: {len(flood_df)}条 ({len(flood_df)/len(df)*100:.1f}%)")
        
        # 数据预处理
        flood_df["time_create"] = pd.to_datetime(flood_df["START"], utc=True)
        flood_df["link_id"] = flood_df["STREET"].str.upper().str.replace(" ", "_")
        flood_df["link_id"] = flood_df["link_id"].astype(str)
        flood_df["id"] = flood_df["OBJECTID"].astype(str)
        
        # 时间分析
        print(f"\n📅 时间范围分析:")
        print(f"   开始时间: {flood_df['time_create'].min()}")
        print(f"   结束时间: {flood_df['time_create'].max()}")
        print(f"   时间跨度: {(flood_df['time_create'].max() - flood_df['time_create'].min()).days}天")
        
        # 按月统计
        monthly_counts = flood_df.groupby(flood_df['time_create'].dt.to_period('M')).size()
        print(f"\n📊 月度洪水记录分布:")
        for month, count in monthly_counts.items():
            print(f"   {month}: {count}条")
        
        # 道路分析
        road_counts = flood_df['link_id'].value_counts()
        print(f"\n🛣️  道路分析:")
        print(f"   独特道路数: {len(road_counts)}条")
        print(f"   平均每条道路洪水次数: {road_counts.mean():.1f}")
        print(f"   最高洪水次数: {road_counts.max()}次 (道路: {road_counts.index[0]})")
        
        print(f"\n🔝 洪水频次TOP-10道路:")
        for i, (road, count) in enumerate(road_counts.head(10).items(), 1):
            print(f"   {i:2d}. {road:<25} {count:3d}次")
        
        # 时序分割详细分析
        df_sorted = flood_df.sort_values('time_create')
        split_idx = int(len(df_sorted) * 0.7)
        self.train_df = df_sorted.iloc[:split_idx].copy()
        self.test_df = df_sorted.iloc[split_idx:].copy()
        
        print(f"\n✂️  时序分割详情:")
        print(f"   训练集: {len(self.train_df)}条 ({len(self.train_df)/len(flood_df)*100:.1f}%)")
        print(f"   测试集: {len(self.test_df)}条 ({len(self.test_df)/len(flood_df)*100:.1f}%)")
        print(f"   训练时间段: {self.train_df['time_create'].min().date()} 至 {self.train_df['time_create'].max().date()}")
        print(f"   测试时间段: {self.test_df['time_create'].min().date()} 至 {self.test_df['time_create'].max().date()}")
        
        # 训练集vs测试集道路重叠分析
        train_roads = set(self.train_df['link_id'].unique())
        test_roads = set(self.test_df['link_id'].unique())
        overlap_roads = train_roads & test_roads
        
        print(f"\n🔄 训练集vs测试集道路分析:")
        print(f"   训练集独特道路: {len(train_roads)}条")
        print(f"   测试集独特道路: {len(test_roads)}条")
        print(f"   重叠道路: {len(overlap_roads)}条 ({len(overlap_roads)/len(train_roads)*100:.1f}% of 训练集)")
        print(f"   测试集新道路: {len(test_roads - train_roads)}条")
        
        return flood_df
        
    def analyze_network_construction(self):
        """详细分析网络构建过程"""
        print(f"\n\n🏗️  贝叶斯网络构建详细过程")
        print("=" * 60)
        
        # 1. 创建网络并计算边际概率
        self.flood_net = FloodBayesNetwork(t_window="D")
        self.flood_net.fit_marginal(self.train_df)
        
        print(f"\n1️⃣  边际概率计算:")
        marginals_sorted = self.flood_net.marginals.sort_values('p', ascending=False)
        print(f"   基于训练集的{len(self.train_df)}条记录")
        print(f"   计算了{len(marginals_sorted)}条道路的边际概率")
        
        print(f"\n📊 边际概率分布:")
        prob_ranges = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 1.0)]
        for low, high in prob_ranges:
            count = len(marginals_sorted[(marginals_sorted['p'] >= low) & (marginals_sorted['p'] < high)])
            print(f"   {low:.1f}-{high:.1f}: {count:2d}条道路")
        
        print(f"\n🔝 边际概率TOP-15道路:")
        for i, (_, row) in enumerate(marginals_sorted.head(15).iterrows(), 1):
            print(f"   {i:2d}. {row['link_id']:<25} P={row['p']:.3f}")
        
        print(f"\n🔻 边际概率BOTTOM-10道路:")
        for i, (_, row) in enumerate(marginals_sorted.tail(10).iterrows(), 1):
            print(f"   {i:2d}. {row['link_id']:<25} P={row['p']:.3f}")
        
        # 2. 共现分析
        print(f"\n2️⃣  共现网络构建 (参数: occ_thr=3, edge_thr=2, weight_thr=0.3):")
        
        # 显示构建前的统计
        time_groups, occurrence, co_occurrence = self.flood_net.process_raw_flood_data(self.train_df.copy())
        
        print(f"   训练数据包含{len(time_groups)}个洪水日")
        print(f"   道路出现统计: {len(occurrence)}条道路")
        print(f"   共现对统计: {len(co_occurrence)}个道路对")
        
        # 显示出现频次分布
        occ_counts = Counter(occurrence.values())
        print(f"\n📈 道路出现频次分布:")
        for freq in sorted(occ_counts.keys(), reverse=True)[:10]:
            print(f"   出现{freq}次: {occ_counts[freq]}条道路")
        
        # 构建网络
        self.flood_net.build_network_by_co_occurrence(
            self.train_df,
            occ_thr=3,
            edge_thr=2,
            weight_thr=0.3,
            report=False
        )
        
        print(f"\n✅ 网络构建完成:")
        print(f"   节点数: {self.flood_net.network.number_of_nodes()}")
        print(f"   边数: {self.flood_net.network.number_of_edges()}")
        
        # 分析被过滤掉的道路
        all_roads = set(occurrence.keys())
        network_roads = set(self.flood_net.network.nodes())
        filtered_roads = all_roads - network_roads
        
        print(f"\n🚫 过滤分析:")
        print(f"   原始道路数: {len(all_roads)}")
        print(f"   保留道路数: {len(network_roads)}")
        print(f"   过滤道路数: {len(filtered_roads)} ({len(filtered_roads)/len(all_roads)*100:.1f}%)")
        
        if len(filtered_roads) > 0:
            print(f"\n🔍 部分被过滤道路 (出现次数<3):")
            filtered_with_count = [(road, occurrence[road]) for road in filtered_roads]
            filtered_with_count.sort(key=lambda x: x[1], reverse=True)
            for road, count in filtered_with_count[:10]:
                print(f"   {road:<25} 出现{count}次")
        
        # 网络拓扑分析
        print(f"\n3️⃣  网络拓扑特征:")
        
        # 度分布
        in_degrees = dict(self.flood_net.network.in_degree())
        out_degrees = dict(self.flood_net.network.out_degree())
        
        print(f"   平均入度: {np.mean(list(in_degrees.values())):.2f}")
        print(f"   平均出度: {np.mean(list(out_degrees.values())):.2f}")
        print(f"   最大入度: {max(in_degrees.values())} (节点: {max(in_degrees, key=in_degrees.get)})")
        print(f"   最大出度: {max(out_degrees.values())} (节点: {max(out_degrees, key=out_degrees.get)})")
        
        # 显示高度节点
        print(f"\n🌟 高入度节点 (容易被其他道路影响):")
        high_in_degree = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:8]
        for road, degree in high_in_degree:
            marginal_p = self.flood_net.marginals[self.flood_net.marginals['link_id'] == road]['p'].iloc[0]
            print(f"   {road:<25} 入度={degree}, P(洪水)={marginal_p:.3f}")
        
        print(f"\n🌟 高出度节点 (容易影响其他道路):")
        high_out_degree = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)[:8]
        for road, degree in high_out_degree:
            marginal_p = self.flood_net.marginals[self.flood_net.marginals['link_id'] == road]['p'].iloc[0]
            print(f"   {road:<25} 出度={degree}, P(洪水)={marginal_p:.3f}")
        
        # 边权分析
        edge_weights = [d['weight'] for u, v, d in self.flood_net.network.edges(data=True)]
        print(f"\n🔗 边权分析:")
        print(f"   边权范围: {min(edge_weights):.3f} - {max(edge_weights):.3f}")
        print(f"   平均边权: {np.mean(edge_weights):.3f}")
        print(f"   边权中位数: {np.median(edge_weights):.3f}")
        
        # 显示最强连接
        edge_list = [(u, v, d['weight']) for u, v, d in self.flood_net.network.edges(data=True)]
        edge_list.sort(key=lambda x: x[2], reverse=True)
        
        print(f"\n💪 最强依赖关系TOP-10:")
        for i, (u, v, weight) in enumerate(edge_list[:10], 1):
            print(f"   {i:2d}. {u} → {v} (权重={weight:.3f})")
        
        return network_roads
    
    def analyze_conditional_probabilities(self):
        """分析条件概率表"""
        print(f"\n\n4️⃣  条件概率表 (CPT) 构建详细过程")
        print("=" * 60)
        
        # 拟合条件概率
        self.flood_net.fit_conditional(self.train_df, max_parents=2, alpha=1.0)
        
        print(f"参数: max_parents=2, alpha=1.0 (拉普拉斯平滑)")
        
        # 分析CPT统计
        nodes_with_parents = 0
        nodes_without_parents = 0
        total_cpt_entries = 0
        
        for node in self.flood_net.network.nodes():
            parents = list(self.flood_net.network.predecessors(node))
            if len(parents) > 0 and node in self.flood_net.conditionals:
                nodes_with_parents += 1
                cpt_size = 2 ** len(self.flood_net.conditionals[node]['parents'])
                total_cpt_entries += cpt_size
            else:
                nodes_without_parents += 1
        
        print(f"\n📊 CPT统计:")
        print(f"   有父节点的节点: {nodes_with_parents}个")
        print(f"   无父节点的节点: {nodes_without_parents}个")
        print(f"   总CPT条目数: {total_cpt_entries}个")
        
        # 显示几个具体的CPT例子
        print(f"\n📋 条件概率表示例:")
        
        cpt_examples = 0
        for node in self.flood_net.network.nodes():
            if node in self.flood_net.conditionals and cpt_examples < 3:
                cpt_examples += 1
                cfg = self.flood_net.conditionals[node]
                parents = cfg['parents']
                conditionals = cfg['conditionals']
                
                marginal_p = self.flood_net.marginals[self.flood_net.marginals['link_id'] == node]['p'].iloc[0]
                print(f"\n   节点: {node} (边际概率={marginal_p:.3f})")
                print(f"   父节点: {parents}")
                print(f"   条件概率:")
                
                for state, prob in conditionals.items():
                    parent_state_str = ", ".join([f"{p}={s}" for p, s in zip(parents, state)])
                    print(f"     P({node}=1 | {parent_state_str}) = {prob:.3f}")
        
        # 构建最终的贝叶斯网络
        self.flood_net.build_bayes_network()
        print(f"\n✅ 贝叶斯网络构建完成，准备进行推理")
        
    def analyze_test_data_structure(self):
        """分析测试数据结构"""
        print(f"\n\n🧪 测试数据结构详细分析")
        print("=" * 60)
        
        # 按日期分组测试数据
        test_by_date = self.test_df.groupby(self.test_df["time_create"].dt.floor("D"))
        
        print(f"测试数据覆盖{len(test_by_date)}个洪水日")
        
        # 统计每日洪水道路数量
        daily_road_counts = []
        daily_network_road_counts = []
        network_roads = set(self.flood_net.network.nodes())
        
        print(f"\n📅 测试日期详细分解:")
        for i, (date, day_group) in enumerate(test_by_date):
            flooded_roads = list(day_group["link_id"].unique())
            flooded_in_network = [road for road in flooded_roads if road in network_roads]
            
            daily_road_counts.append(len(flooded_roads))
            daily_network_road_counts.append(len(flooded_in_network))
            
            if i < 10:  # 显示前10天的详情
                print(f"   {date.date()}: {len(flooded_roads)}条道路洪水, {len(flooded_in_network)}条在网络中")
                if len(flooded_in_network) > 0:
                    print(f"      网络道路: {', '.join(flooded_in_network[:5])}{'...' if len(flooded_in_network) > 5 else ''}")
        
        if len(test_by_date) > 10:
            print(f"   ... (还有{len(test_by_date)-10}天)")
        
        print(f"\n📊 测试数据统计:")
        print(f"   平均每日洪水道路: {np.mean(daily_road_counts):.1f}条")
        print(f"   平均每日网络道路: {np.mean(daily_network_road_counts):.1f}条")
        print(f"   可评估日数 (≥2条网络道路): {sum(1 for x in daily_network_road_counts if x >= 2)}天")
        
        # 测试道路频次分析
        test_road_counts = self.test_df['link_id'].value_counts()
        test_network_roads = test_road_counts[test_road_counts.index.isin(network_roads)]
        
        print(f"\n🛣️  测试集道路分析:")
        print(f"   测试集独特道路: {len(test_road_counts)}条")
        print(f"   测试集网络道路: {len(test_network_roads)}条")
        print(f"   覆盖率: {len(test_network_roads)/len(network_roads)*100:.1f}% (网络道路在测试集中出现)")
        
        print(f"\n🔝 测试集高频洪水道路TOP-10:")
        for i, (road, count) in enumerate(test_network_roads.head(10).items(), 1):
            marginal_p = self.flood_net.marginals[self.flood_net.marginals['link_id'] == road]['p'].iloc[0]
            print(f"   {i:2d}. {road:<25} {count:2d}次 (训练P={marginal_p:.3f})")
    
    def run_complete_analysis(self):
        """运行完整的详细分析"""
        # 1. 数据分析
        original_df = self.load_and_analyze_data()
        
        # 2. 网络构建分析
        network_roads = self.analyze_network_construction()
        
        # 3. 条件概率分析
        self.analyze_conditional_probabilities()
        
        # 4. 测试数据分析
        self.analyze_test_data_structure()
        
        print(f"\n\n✅ 详细分析完成！")
        print(f"📊 关键统计总结:")
        print(f"   训练数据: {len(self.train_df)}条记录")
        print(f"   测试数据: {len(self.test_df)}条记录")
        print(f"   网络节点: {len(network_roads)}个")
        print(f"   网络边数: {self.flood_net.network.number_of_edges()}条")
        
        return {
            'train_df': self.train_df,
            'test_df': self.test_df,
            'flood_net': self.flood_net,
            'network_roads': network_roads,
            'original_df': original_df
        }

def main():
    """主函数"""
    analyzer = DetailedNetworkAnalyzer()
    results = analyzer.run_complete_analysis()
    return analyzer, results

if __name__ == "__main__":
    analyzer, results = main()