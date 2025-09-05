#!/usr/bin/env python3
"""
Charleston洪水数据道路共现分析脚本

功能：
1. 分析同一天同时淹没的道路对的强共现模式
2. 使用关联规则挖掘方法计算支持度、置信度和提升度
3. 识别具有强关联性的道路对组合
4. 生成统计报告和可视化图表

用法：
    python cooccurrence_analysis.py

参数调整：
    可在main()函数中修改以下参数：
    - min_count: 单路最小出现天数阈值 (默认: 3)
    - min_support: 道路对最小共现天数阈值 (默认: 5)
    - min_lift: 最小提升度阈值 (默认: 2.0)

输出：
    - results/road_pair_stats.csv - 详细的道路对统计
    - figs/top_pairs_lift.png - Top-10道路对提升度条形图
    - 终端输出按提升度排序的前15个道路对

关联规则指标说明：
    - Support: count(A,B)/N - 道路对共现的相对频率
    - Confidence A→B: count(A,B)/count(A) - A出现时B也出现的概率
    - Lift: conf(A→B)/(count(B)/N) - 相对于随机的提升倍数
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from itertools import combinations
import os

class CooccurrenceAnalyzer:
    """道路共现分析器"""
    
    def __init__(self, data_csv_path="Road_Closures_2024.csv", results_dir="results", figs_dir="figs"):
        self.data_csv_path = data_csv_path
        self.results_dir = results_dir
        self.figs_dir = figs_dir
        
        # 创建输出目录
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(figs_dir, exist_ok=True)
        
        # 数据存储
        self.flood_by_date = {}  # {date: set(roads)}
        self.road_counts = {}    # {road: count}
        self.pair_counts = {}    # {(road1, road2): count}
        self.total_flood_days = 0
        
        # 结果存储
        self.road_pair_stats = []
        
    def load_flood_data(self):
        """加载洪水数据并按日期分组"""
        print("1. 加载洪水数据...")
        
        # 读取数据
        df = pd.read_csv(self.data_csv_path)
        print(f"   总记录数: {len(df)}")
        
        # 筛选洪水记录
        flood_df = df[df["REASON"].str.upper() == "FLOOD"].copy()
        print(f"   洪水记录: {len(flood_df)}")
        
        # 处理时间和道路名称
        flood_df["time_create"] = pd.to_datetime(flood_df["START"], utc=True)
        flood_df["date"] = flood_df["time_create"].dt.floor("D")
        flood_df["road"] = flood_df["STREET"].str.upper().str.strip()
        
        # 按日期分组，生成每日洪水道路集合
        for date, group in flood_df.groupby("date"):
            flooded_roads = set(group["road"].dropna().unique())
            if len(flooded_roads) > 0:  # 只保留有道路数据的日期
                self.flood_by_date[str(date.date())] = flooded_roads
        
        self.total_flood_days = len(self.flood_by_date)
        
        print(f"   洪水日期数: {self.total_flood_days}")
        print(f"   日期范围: {min(self.flood_by_date.keys())} 至 {max(self.flood_by_date.keys())}")
        
        # 显示每日道路数分布
        daily_road_counts = [len(roads) for roads in self.flood_by_date.values()]
        print(f"   每日平均道路数: {np.mean(daily_road_counts):.1f}")
        print(f"   每日最多道路数: {max(daily_road_counts)}")
        print(f"   每日最少道路数: {min(daily_road_counts)}")
        
        return self.flood_by_date
        
    def compute_statistics(self, min_count=3):
        """计算道路单独出现和共现统计"""
        print("2. 计算道路出现和共现统计...")
        
        # 统计每条道路出现天数
        all_roads = set()
        for roads in self.flood_by_date.values():
            all_roads.update(roads)
        
        for road in all_roads:
            count = sum(1 for roads in self.flood_by_date.values() if road in roads)
            self.road_counts[road] = count
        
        # 过滤低频道路
        frequent_roads = {road: count for road, count in self.road_counts.items() 
                         if count >= min_count}
        
        print(f"   总道路数: {len(all_roads)}")
        print(f"   频繁道路数 (≥{min_count}天): {len(frequent_roads)}")
        
        # 计算道路对共现次数
        pair_count = 0
        for date, roads in self.flood_by_date.items():
            # 只考虑频繁道路
            frequent_roads_today = [road for road in roads if road in frequent_roads]
            
            # 计算所有道路对组合
            for road1, road2 in combinations(frequent_roads_today, 2):
                pair = tuple(sorted([road1, road2]))  # 确保顺序一致
                self.pair_counts[pair] = self.pair_counts.get(pair, 0) + 1
                pair_count += 1
        
        print(f"   道路对组合总数: {len(self.pair_counts)}")
        print(f"   共现事件总数: {pair_count}")
        
        # 更新road_counts只保留频繁道路
        self.road_counts = frequent_roads
        
        return self.road_counts, self.pair_counts
        
    def compute_association_metrics(self, min_support=5, min_lift=2.0):
        """计算关联规则指标"""
        print("3. 计算关联规则指标...")
        
        # 过滤满足最小支持度的道路对
        frequent_pairs = {pair: count for pair, count in self.pair_counts.items() 
                         if count >= min_support}
        
        print(f"   满足支持度阈值 (≥{min_support}) 的道路对: {len(frequent_pairs)}")
        
        # 计算关联指标
        strong_associations = []
        
        for (road_a, road_b), count_ab in frequent_pairs.items():
            count_a = self.road_counts[road_a]
            count_b = self.road_counts[road_b]
            
            # 置信度
            conf_a_to_b = count_ab / count_a
            conf_b_to_a = count_ab / count_b
            
            # 提升度
            lift_a_to_b = conf_a_to_b / (count_b / self.total_flood_days)
            lift_b_to_a = conf_b_to_a / (count_a / self.total_flood_days)
            
            # 使用更高的提升度
            max_lift = max(lift_a_to_b, lift_b_to_a)
            
            # 应用提升度过滤
            if max_lift >= min_lift:
                strong_associations.append({
                    'A': road_a,
                    'B': road_b,
                    'countA': count_a,
                    'countB': count_b,
                    'countAB': count_ab,
                    'confA2B': conf_a_to_b,
                    'confB2A': conf_b_to_a,
                    'lift': max_lift,
                    'lift_A2B': lift_a_to_b,
                    'lift_B2A': lift_b_to_a
                })
        
        # 按提升度排序
        strong_associations.sort(key=lambda x: x['lift'], reverse=True)
        
        print(f"   强关联道路对 (lift≥{min_lift}): {len(strong_associations)}")
        
        if len(strong_associations) > 0:
            print(f"   提升度范围: [{min([x['lift'] for x in strong_associations]):.2f}, "
                  f"{max([x['lift'] for x in strong_associations]):.2f}]")
        
        self.road_pair_stats = strong_associations
        return strong_associations
        
    def save_results(self):
        """保存结果到CSV"""
        print("4. 保存统计结果...")
        
        if len(self.road_pair_stats) == 0:
            print("   ⚠️ 没有满足条件的道路对，创建空文件")
            empty_df = pd.DataFrame(columns=['A', 'B', 'countA', 'countB', 'countAB', 
                                           'confA2B', 'confB2A', 'lift'])
            csv_path = os.path.join(self.results_dir, "road_pair_stats.csv")
            empty_df.to_csv(csv_path, index=False)
            return csv_path
        
        # 转换为DataFrame
        df = pd.DataFrame(self.road_pair_stats)
        
        # 选择输出列
        output_columns = ['A', 'B', 'countA', 'countB', 'countAB', 
                         'confA2B', 'confB2A', 'lift']
        df_output = df[output_columns].copy()
        
        # 保存CSV
        csv_path = os.path.join(self.results_dir, "road_pair_stats.csv")
        df_output.to_csv(csv_path, index=False, float_format='%.4f')
        
        print(f"   ✅ 结果保存到: {csv_path}")
        print(f"   共保存 {len(df_output)} 个强关联道路对")
        
        return csv_path
        
    def print_top_results(self, top_n=15):
        """打印前N个最强关联的道路对"""
        print(f"5. 显示前{top_n}个强关联道路对...")
        
        if len(self.road_pair_stats) == 0:
            print("   没有满足条件的道路对")
            return
        
        print(f"\n{'='*90}")
        print(f"🏆 按提升度(Lift)排序的前{min(top_n, len(self.road_pair_stats))}个强关联道路对")
        print(f"{'='*90}")
        
        # 表头
        print(f"{'排名':<4} {'道路A':<25} {'道路B':<25} {'Lift':<6} {'置信度A→B':<10} {'置信度B→A':<10}")
        print("-" * 90)
        
        # 显示结果
        for i, pair in enumerate(self.road_pair_stats[:top_n]):
            rank = f"{i+1}."
            road_a = pair['A'][:24]  # 截断长名称
            road_b = pair['B'][:24]
            lift = f"{pair['lift']:.2f}"
            conf_ab = f"{pair['confA2B']:.3f}"
            conf_ba = f"{pair['confB2A']:.3f}"
            
            print(f"{rank:<4} {road_a:<25} {road_b:<25} {lift:<6} {conf_ab:<10} {conf_ba:<10}")
        
        # 显示统计摘要
        if len(self.road_pair_stats) > 0:
            avg_lift = np.mean([x['lift'] for x in self.road_pair_stats])
            max_lift = max([x['lift'] for x in self.road_pair_stats])
            
            print(f"\n📊 统计摘要:")
            print(f"   强关联对总数: {len(self.road_pair_stats)}")
            print(f"   平均提升度: {avg_lift:.2f}")
            print(f"   最高提升度: {max_lift:.2f}")
            
            # 最强关联对解释
            top_pair = self.road_pair_stats[0]
            print(f"\n🎯 最强关联: {top_pair['A']} ↔ {top_pair['B']}")
            print(f"   共现天数: {top_pair['countAB']}")
            print(f"   {top_pair['A']} 出现 {top_pair['countA']} 天")
            print(f"   {top_pair['B']} 出现 {top_pair['countB']} 天")
            print(f"   提升度: {top_pair['lift']:.2f} (比随机高 {top_pair['lift']:.1f} 倍)")
        
    def create_visualization(self, top_n=10):
        """创建Top-N道路对提升度条形图"""
        print("6. 生成可视化图表...")
        
        if len(self.road_pair_stats) == 0:
            print("   没有数据可视化")
            return None
        
        # 准备数据
        top_pairs = self.road_pair_stats[:min(top_n, len(self.road_pair_stats))]
        
        # 创建道路对标签
        pair_labels = []
        lifts = []
        
        for pair in top_pairs:
            # 缩短道路名称
            road_a = pair['A'].replace('STREET', 'ST').replace('AVENUE', 'AVE')[:15]
            road_b = pair['B'].replace('STREET', 'ST').replace('AVENUE', 'AVE')[:15]
            label = f"{road_a}\n↔\n{road_b}"
            pair_labels.append(label)
            lifts.append(pair['lift'])
        
        # 创建图表
        plt.figure(figsize=(14, 8))
        
        # 创建条形图
        bars = plt.bar(range(len(lifts)), lifts, 
                      color='steelblue', alpha=0.8, edgecolor='black', linewidth=1)
        
        # 添加数值标签
        for bar, lift in zip(bars, lifts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{lift:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # 设置图表属性
        plt.title(f'Top-{len(top_pairs)} 道路对关联强度 (提升度)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('道路对', fontsize=12)
        plt.ylabel('提升度 (Lift)', fontsize=12)
        
        # 设置x轴标签
        plt.xticks(range(len(pair_labels)), pair_labels, rotation=45, ha='right', fontsize=9)
        
        # 添加网格
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 添加基准线
        plt.axhline(y=2.0, color='red', linestyle='--', alpha=0.7, 
                   label='最小提升度阈值 (2.0)')
        plt.legend()
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        fig_path = os.path.join(self.figs_dir, "top_pairs_lift.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 可视化图表保存到: {fig_path}")
        
        return fig_path
        
    def run_analysis(self, min_count=3, min_support=5, min_lift=2.0):
        """运行完整的共现分析"""
        print("🚀 开始Charleston洪水道路共现分析...")
        print("="*70)
        
        try:
            # 1. 加载数据
            self.load_flood_data()
            
            # 2. 计算统计
            self.compute_statistics(min_count=min_count)
            
            # 3. 计算关联指标
            self.compute_association_metrics(min_support=min_support, min_lift=min_lift)
            
            # 4. 保存结果
            self.save_results()
            
            # 5. 显示结果
            self.print_top_results(top_n=15)
            
            # 6. 创建可视化
            self.create_visualization(top_n=10)
            
            print(f"\n✅ 道路共现分析完成！")
            print(f"📁 详细结果: {self.results_dir}/road_pair_stats.csv")
            print(f"📊 可视化图表: {self.figs_dir}/top_pairs_lift.png")
            
            return self.road_pair_stats
            
        except Exception as e:
            print(f"\n❌ 分析过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """主函数 - 可在此调整分析参数"""
    
    # 分析参数 (可调整)
    MIN_COUNT = 3      # 单路最小出现天数
    MIN_SUPPORT = 5    # 道路对最小共现天数  
    MIN_LIFT = 2.0     # 最小提升度阈值
    
    print(f"分析参数:")
    print(f"  最小单路出现天数: {MIN_COUNT}")
    print(f"  最小道路对共现天数: {MIN_SUPPORT}")
    print(f"  最小提升度: {MIN_LIFT}")
    print()
    
    # 创建分析器
    analyzer = CooccurrenceAnalyzer()
    
    # 运行分析
    results = analyzer.run_analysis(
        min_count=MIN_COUNT,
        min_support=MIN_SUPPORT, 
        min_lift=MIN_LIFT
    )
    
    if results:
        print(f"\n🎯 关键发现:")
        print(f"   发现 {len(results)} 个强关联道路对")
        print(f"   这些道路对在洪水期间具有显著的共现模式")
        print(f"   可用于洪水预警和应急响应规划")

if __name__ == "__main__":
    main()