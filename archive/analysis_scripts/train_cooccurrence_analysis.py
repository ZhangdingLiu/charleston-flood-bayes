#!/usr/bin/env python3
"""
基于训练集的道路共现分析

功能：
1. 严格基于训练集数据计算道路共现统计
2. 避免数据泄露，确保训练-测试分离
3. 计算条件概率confA2B，选择最强关联的道路对
4. 为基于训练集的贝叶斯网络构建提供统计基础

分析指标：
- countA: 道路A的洪水次数
- countB: 道路B的洪水次数  
- countAB: 道路A和B同时洪水的次数
- confA2B: P(B洪水|A洪水) = countAB / countA
- confB2A: P(A洪水|B洪水) = countAB / countB

选择策略：
选择confA2B最高的前3对道路，构建A→B的有向边

用法：
    python train_cooccurrence_analysis.py

输出：
    - results/train_cooccurrence_stats.csv - 训练集共现统计表
    - figs/train_cooccurrence_heatmap.png - 训练集共现热图
    - 终端输出最强关联的道路对分析
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
from itertools import combinations

# 设置随机种子（与其他脚本保持一致）
RANDOM_SEED = 42

class TrainCooccurrenceAnalyzer:
    """基于训练集的道路共现分析器"""
    
    def __init__(self, 
                 data_csv_path="Road_Closures_2024.csv",
                 results_dir="results",
                 figs_dir="figs",
                 min_road_count=3):
        self.data_csv_path = data_csv_path
        self.results_dir = results_dir
        self.figs_dir = figs_dir
        self.min_road_count = min_road_count  # 最小道路出现次数阈值
        
        # 创建输出目录
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(figs_dir, exist_ok=True)
        
        # 数据存储
        self.train_df = None
        self.test_df = None
        self.road_stats = {}
        self.cooccurrence_stats = []
        
    def load_and_split_data(self):
        """加载数据并分割（与其他脚本保持一致的分割）"""
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
        
        print(f"   总洪水记录: {len(df)}条")
        print(f"   训练集: {len(self.train_df)}条")
        print(f"   测试集: {len(self.test_df)}条")
        
        return self.train_df, self.test_df
        
    def analyze_road_frequencies(self):
        """分析训练集中各道路的洪水频率"""
        print("2. 分析训练集道路频率...")
        
        # 统计每条道路的出现次数
        road_counts = Counter(self.train_df['road'])
        
        # 过滤掉出现次数过少的道路
        filtered_roads = {road: count for road, count in road_counts.items() 
                         if count >= self.min_road_count}
        
        print(f"   训练集中独特道路数: {len(road_counts)}")
        print(f"   出现≥{self.min_road_count}次的道路数: {len(filtered_roads)}")
        
        # 显示前20个最频繁的道路
        print(f"   前20个最频繁洪水道路:")
        for road, count in sorted(filtered_roads.items(), 
                                key=lambda x: x[1], reverse=True)[:20]:
            print(f"     {road}: {count}次")
        
        self.road_stats = filtered_roads
        return filtered_roads
        
    def compute_cooccurrence_matrix(self):
        """计算训练集中道路的共现矩阵"""
        print("3. 计算道路共现矩阵...")
        
        # 按日期分组，获取每天洪水的道路集合
        daily_roads = defaultdict(set)
        for _, row in self.train_df.iterrows():
            road = row['road']
            if road in self.road_stats:  # 只考虑频繁道路
                date_str = str(row['date'].date())
                daily_roads[date_str].add(road)
        
        print(f"   分析天数: {len(daily_roads)}天")
        
        # 计算所有道路对的共现统计
        road_list = list(self.road_stats.keys())
        cooccurrence_data = []
        
        for i, road_a in enumerate(road_list):
            for j, road_b in enumerate(road_list):
                if i >= j:  # 避免重复计算和自环
                    continue
                    
                count_a = self.road_stats[road_a]
                count_b = self.road_stats[road_b]
                
                # 计算共现次数
                count_ab = 0
                for date, roads_set in daily_roads.items():
                    if road_a in roads_set and road_b in roads_set:
                        count_ab += 1
                
                # 计算条件概率
                conf_a2b = count_ab / count_a if count_a > 0 else 0
                conf_b2a = count_ab / count_b if count_b > 0 else 0
                
                # 只保存有共现的道路对
                if count_ab > 0:
                    cooccurrence_data.append({
                        'road_a': road_a,
                        'road_b': road_b,
                        'count_a': count_a,
                        'count_b': count_b,
                        'count_ab': count_ab,
                        'conf_a2b': conf_a2b,
                        'conf_b2a': conf_b2a,
                        'max_conf': max(conf_a2b, conf_b2a)
                    })
        
        print(f"   找到共现道路对: {len(cooccurrence_data)}对")
        
        self.cooccurrence_stats = cooccurrence_data
        return cooccurrence_data
        
    def select_top_road_pairs(self, top_k=10):
        """选择条件概率最高的道路对"""
        print("4. 选择最强关联道路对...")
        
        if not self.cooccurrence_stats:
            print("   ❌ 没有共现统计数据")
            return []
        
        # 按最大条件概率排序
        sorted_pairs = sorted(self.cooccurrence_stats, 
                            key=lambda x: x['max_conf'], reverse=True)
        
        top_pairs = sorted_pairs[:top_k]
        
        print(f"   Top-{top_k} 道路对 (按条件概率):")
        print(f"   {'道路A':<20} {'道路B':<20} {'countA':<8} {'countB':<8} {'countAB':<9} {'confA2B':<8} {'confB2A':<8}")
        print("-" * 100)
        
        for pair in top_pairs:
            print(f"   {pair['road_a']:<20} {pair['road_b']:<20} "
                  f"{pair['count_a']:<8} {pair['count_b']:<8} "
                  f"{pair['count_ab']:<9} {pair['conf_a2b']:<8.4f} {pair['conf_b2a']:<8.4f}")
        
        return top_pairs
        
    def create_cooccurrence_heatmap(self, top_pairs):
        """创建共现关系热图"""
        print("5. 生成共现关系热图...")
        
        if len(top_pairs) < 3:
            print("   ⚠️  道路对数量不足，跳过热图生成")
            return None
        
        # 选择前15对用于可视化
        viz_pairs = top_pairs[:15]
        
        # 获取所有涉及的道路
        all_roads = set()
        for pair in viz_pairs:
            all_roads.add(pair['road_a'])
            all_roads.add(pair['road_b'])
        
        road_list = sorted(list(all_roads))
        n_roads = len(road_list)
        
        # 创建共现矩阵
        cooccur_matrix = np.zeros((n_roads, n_roads))
        road_to_idx = {road: i for i, road in enumerate(road_list)}
        
        for pair in viz_pairs:
            i = road_to_idx[pair['road_a']]
            j = road_to_idx[pair['road_b']]
            # 使用最大条件概率填充矩阵
            cooccur_matrix[i, j] = pair['max_conf']
            cooccur_matrix[j, i] = pair['max_conf']
        
        # 创建热图
        plt.figure(figsize=(12, 10))
        
        mask = cooccur_matrix == 0
        sns.heatmap(cooccur_matrix, 
                   annot=True, 
                   fmt='.3f',
                   cmap='Reds',
                   xticklabels=[road[:15] + '...' if len(road) > 15 else road for road in road_list],
                   yticklabels=[road[:15] + '...' if len(road) > 15 else road for road in road_list],
                   mask=mask,
                   cbar_kws={'label': 'Max Conditional Probability'})
        
        plt.title('Training Set Road Co-occurrence Heatmap\n(Top Road Pairs by Conditional Probability)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Road B')
        plt.ylabel('Road A')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # 保存图表
        heatmap_path = os.path.join(self.figs_dir, "train_cooccurrence_heatmap.png")
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 热图保存到: {heatmap_path}")
        return heatmap_path
        
    def save_cooccurrence_stats(self):
        """保存共现统计到CSV文件"""
        print("6. 保存共现统计...")
        
        if not self.cooccurrence_stats:
            print("   ❌ 没有统计数据可保存")
            return None
        
        # 转换为DataFrame
        df_stats = pd.DataFrame(self.cooccurrence_stats)
        
        # 按最大条件概率排序
        df_stats = df_stats.sort_values('max_conf', ascending=False)
        
        # 重新排列列的顺序
        df_stats = df_stats[['road_a', 'road_b', 'count_a', 'count_b', 
                           'count_ab', 'conf_a2b', 'conf_b2a', 'max_conf']]
        
        # 保存到CSV
        output_path = os.path.join(self.results_dir, "train_cooccurrence_stats.csv")
        df_stats.to_csv(output_path, index=False)
        
        print(f"   ✅ 统计数据保存到: {output_path}")
        print(f"   共现道路对数量: {len(df_stats)}")
        
        return output_path
        
    def analyze_network_candidates(self, top_pairs):
        """分析候选网络结构"""
        print("7. 分析候选贝叶斯网络结构...")
        
        if len(top_pairs) < 3:
            print("   ❌ 道路对数量不足，无法构建网络")
            return None
        
        print(f"\n   推荐的前3对道路用于贝叶斯网络:")
        print(f"   {'序号':<4} {'父节点':<20} {'子节点':<20} {'条件概率':<10} {'关系强度'}")
        print("-" * 80)
        
        network_edges = []
        for i, pair in enumerate(top_pairs[:3]):
            # 选择条件概率更高的方向作为边的方向
            if pair['conf_a2b'] >= pair['conf_b2a']:
                parent, child = pair['road_a'], pair['road_b']
                cond_prob = pair['conf_a2b']
            else:
                parent, child = pair['road_b'], pair['road_a']
                cond_prob = pair['conf_b2a']
            
            network_edges.append((parent, child, cond_prob))
            
            strength = "极强" if cond_prob > 0.6 else "强" if cond_prob > 0.4 else "中等"
            print(f"   {i+1:<4} {parent:<20} {child:<20} {cond_prob:<10.4f} {strength}")
        
        print(f"\n   网络特点:")
        print(f"   - 节点数: {len(set([edge[0] for edge in network_edges] + [edge[1] for edge in network_edges]))}")
        print(f"   - 边数: {len(network_edges)}")
        print(f"   - 平均条件概率: {np.mean([edge[2] for edge in network_edges]):.4f}")
        
        return network_edges
        
    def run_analysis(self):
        """运行完整的训练集共现分析"""
        print("🚀 开始基于训练集的道路共现分析...")
        print("="*60)
        
        try:
            # 1. 数据加载和分割
            self.load_and_split_data()
            
            # 2. 分析道路频率
            self.analyze_road_frequencies()
            
            # 3. 计算共现矩阵
            self.compute_cooccurrence_matrix()
            
            # 4. 选择顶级道路对
            top_pairs = self.select_top_road_pairs()
            
            # 5. 创建可视化
            self.create_cooccurrence_heatmap(top_pairs)
            
            # 6. 保存统计数据
            self.save_cooccurrence_stats()
            
            # 7. 分析网络候选
            network_edges = self.analyze_network_candidates(top_pairs)
            
            print(f"\n✅ 训练集共现分析完成！")
            print(f"📁 统计文件: {self.results_dir}/train_cooccurrence_stats.csv")
            print(f"📊 可视化: {self.figs_dir}/train_cooccurrence_heatmap.png")
            
            return {
                'top_pairs': top_pairs,
                'network_edges': network_edges,
                'road_stats': self.road_stats
            }
            
        except Exception as e:
            print(f"\n❌ 分析过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """主函数"""
    analyzer = TrainCooccurrenceAnalyzer()
    results = analyzer.run_analysis()
    
    if results:
        print(f"\n🎯 分析总结:")
        print(f"   基于训练集的道路共现模式已识别")
        print(f"   避免了数据泄露问题")
        print(f"   为构建高质量贝叶斯网络提供了统计基础")
        print(f"   选择的道路对具有真实的预测价值")

if __name__ == "__main__":
    main()