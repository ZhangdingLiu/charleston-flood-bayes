#!/usr/bin/env python3
"""
N MARKET ST ↔ S MARKET ST 两条道路贝叶斯网络实验

目的：
通过最简单的两条道路案例，深入分析为什么贝叶斯网络在测试集上表现差
选择这两条道路是因为它们在共现分析中显示了强关联性

实验设计：
1. 仅使用 N MARKET ST 和 S MARKET ST 两条道路
2. 在训练集上构建贝叶斯网络 
3. 在测试集上进行预测
4. 详细分析训练集和测试集中这两条道路的洪水出现模式
5. 显示具体的条件概率计算和推理过程

用法：
    python two_road_experiment.py

输出：
    - 训练集和测试集中两条道路的详细统计
    - 贝叶斯网络结构和条件概率表
    - 测试集预测结果的详细分析
    - 性能不佳的根本原因分析
"""

import pandas as pd
import numpy as np
import pickle
import os
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    accuracy_score
)

# 贝叶斯网络相关
try:
    try:
        from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
    except ImportError:
        from pgmpy.models import BayesianNetwork
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination
except ImportError:
    print("请安装pgmpy: pip install pgmpy")
    exit(1)

# 设置随机种子
RANDOM_SEED = 42

class TwoRoadBayesianExperiment:
    """两条道路贝叶斯网络实验类"""
    
    def __init__(self, data_csv_path="Road_Closures_2024.csv"):
        self.data_csv_path = data_csv_path
        self.selected_roads = ['N MARKET ST', 'S MARKET ST']
        
        # 数据存储
        self.train_df = None
        self.test_df = None
        self.train_filtered = None
        self.test_filtered = None
        self.bayesian_network = None
        self.inference_engine = None
        
        # 统计数据
        self.train_stats = {}
        self.test_stats = {}
        
    def load_and_split_data(self):
        """加载数据并分割"""
        print("🚀 开始两条道路贝叶斯网络实验")
        print("="*60)
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
        
        # 过滤出两条目标道路
        self.train_filtered = self.train_df[self.train_df['road'].isin(self.selected_roads)].copy()
        self.test_filtered = self.test_df[self.test_df['road'].isin(self.selected_roads)].copy()
        
        print(f"   训练集目标道路记录: {len(self.train_filtered)}条")
        print(f"   测试集目标道路记录: {len(self.test_filtered)}条")
        
        return True
        
    def analyze_road_patterns(self):
        """分析两条道路在训练集和测试集中的模式"""
        print("\n2. 分析道路洪水模式...")
        
        def analyze_dataset(filtered_df, dataset_name):
            print(f"\n   📊 {dataset_name}统计:")
            
            # 各道路出现次数
            road_counts = filtered_df['road'].value_counts()
            for road in self.selected_roads:
                count = road_counts.get(road, 0)
                print(f"     {road}: {count}次")
            
            # 按日期分组，计算共现
            daily_roads = defaultdict(set)
            for _, row in filtered_df.iterrows():
                date_str = str(row['date'].date())
                daily_roads[date_str].add(row['road'])
            
            # 统计共现模式
            patterns = {
                'only_n_market': 0,
                'only_s_market': 0, 
                'both_roads': 0,
                'total_days': len(daily_roads)
            }
            
            cooccur_dates = []
            for date, roads in daily_roads.items():
                if 'N MARKET ST' in roads and 'S MARKET ST' in roads:
                    patterns['both_roads'] += 1
                    cooccur_dates.append(date)
                elif 'N MARKET ST' in roads:
                    patterns['only_n_market'] += 1
                elif 'S MARKET ST' in roads:
                    patterns['only_s_market'] += 1
            
            print(f"     有洪水的天数: {patterns['total_days']}天")
            print(f"     仅N MARKET ST: {patterns['only_n_market']}天")
            print(f"     仅S MARKET ST: {patterns['only_s_market']}天")
            print(f"     两条路同时: {patterns['both_roads']}天")
            
            if patterns['both_roads'] > 0:
                print(f"     共现日期示例: {cooccur_dates[:3]}")
            
            return patterns, daily_roads
        
        # 分析训练集
        self.train_stats, self.train_daily = analyze_dataset(self.train_filtered, "训练集")
        
        # 分析测试集
        self.test_stats, self.test_daily = analyze_dataset(self.test_filtered, "测试集")
        
        # 计算条件概率统计
        print(f"\n   📈 训练集条件概率计算:")
        train_n_count = sum(1 for roads in self.train_daily.values() if 'N MARKET ST' in roads)
        train_s_count = sum(1 for roads in self.train_daily.values() if 'S MARKET ST' in roads)
        train_both_count = self.train_stats['both_roads']
        
        if train_n_count > 0:
            conf_s_given_n = train_both_count / train_n_count
            print(f"     P(S MARKET ST=洪水 | N MARKET ST=洪水) = {train_both_count}/{train_n_count} = {conf_s_given_n:.4f}")
        
        if train_s_count > 0:
            conf_n_given_s = train_both_count / train_s_count
            print(f"     P(N MARKET ST=洪水 | S MARKET ST=洪水) = {train_both_count}/{train_s_count} = {conf_n_given_s:.4f}")
        
        return True
        
    def build_bayesian_network(self):
        """构建两条道路的贝叶斯网络"""
        print("\n3. 构建贝叶斯网络...")
        
        # 创建日期-道路二元矩阵
        all_dates = sorted(set(self.train_daily.keys()))
        matrix_data = []
        
        for date in all_dates:
            roads_today = self.train_daily[date]
            row_data = {
                'date': date,
                'N_MARKET_ST': 1 if 'N MARKET ST' in roads_today else 0,
                'S_MARKET_ST': 1 if 'S MARKET ST' in roads_today else 0
            }
            matrix_data.append(row_data)
        
        binary_df = pd.DataFrame(matrix_data)
        
        print(f"   训练矩阵大小: {len(binary_df)} 天 × 2 道路")
        print(f"   N MARKET ST洪水频率: {binary_df['N_MARKET_ST'].mean():.3f}")
        print(f"   S MARKET ST洪水频率: {binary_df['S_MARKET_ST'].mean():.3f}")
        
        # 决定网络结构（选择条件概率更高的方向）
        n_count = binary_df['N_MARKET_ST'].sum()
        s_count = binary_df['S_MARKET_ST'].sum()
        both_count = ((binary_df['N_MARKET_ST'] == 1) & (binary_df['S_MARKET_ST'] == 1)).sum()
        
        if n_count > 0 and s_count > 0:
            conf_s_given_n = both_count / n_count
            conf_n_given_s = both_count / s_count
            
            print(f"   条件概率比较:")
            print(f"     P(S|N) = {conf_s_given_n:.4f}")
            print(f"     P(N|S) = {conf_n_given_s:.4f}")
            
            # 选择条件概率更高的方向
            if conf_s_given_n >= conf_n_given_s:
                edges = [('N_MARKET_ST', 'S_MARKET_ST')]
                print(f"   选择结构: N MARKET ST → S MARKET ST")
            else:
                edges = [('S_MARKET_ST', 'N_MARKET_ST')]
                print(f"   选择结构: S MARKET ST → N MARKET ST")
        else:
            # 如果一条路从未出现，创建无边网络
            edges = []
            print(f"   无足够数据，创建无边网络")
        
        # 创建贝叶斯网络
        self.bayesian_network = BayesianNetwork(edges)
        
        # 计算CPD参数
        cpds = []
        
        for node in ['N_MARKET_ST', 'S_MARKET_ST']:
            parents = list(self.bayesian_network.predecessors(node))
            
            if len(parents) == 0:
                # 根节点 - 先验概率
                prob_1 = (binary_df[node].sum() + 1) / (len(binary_df) + 2)  # 拉普拉斯平滑
                prob_0 = 1 - prob_1
                
                cpd = TabularCPD(
                    variable=node,
                    variable_card=2,
                    values=[[prob_0], [prob_1]],
                    state_names={node: [0, 1]}
                )
                
                print(f"   {node} (根节点):")
                print(f"     P({node}=1) = {prob_1:.4f}")
                
            else:
                # 有父节点
                parent = parents[0]
                
                # 计算条件概率表
                parent_0_child_0 = len(binary_df[(binary_df[parent] == 0) & (binary_df[node] == 0)])
                parent_0_child_1 = len(binary_df[(binary_df[parent] == 0) & (binary_df[node] == 1)])
                parent_1_child_0 = len(binary_df[(binary_df[parent] == 1) & (binary_df[node] == 0)])
                parent_1_child_1 = len(binary_df[(binary_df[parent] == 1) & (binary_df[node] == 1)])
                
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
                
                print(f"   {node} (子节点, 父节点: {parent}):")
                print(f"     P({node}=1|{parent}=0) = {prob_child_1_given_parent_0:.4f}")
                print(f"     P({node}=1|{parent}=1) = {prob_child_1_given_parent_1:.4f}")
                
                # 计算提升效果
                if prob_child_1_given_parent_0 > 0:
                    lift = prob_child_1_given_parent_1 / prob_child_1_given_parent_0
                    print(f"     条件概率提升: {lift:.2f}x")
            
            cpds.append(cpd)
        
        # 添加CPD到网络
        self.bayesian_network.add_cpds(*cpds)
        
        # 验证网络
        if self.bayesian_network.check_model():
            print("   ✅ 贝叶斯网络构建成功")
        else:
            print("   ❌ 贝叶斯网络验证失败")
            return False
        
        # 创建推理引擎
        self.inference_engine = VariableElimination(self.bayesian_network)
        
        return True
        
    def evaluate_on_test_set(self):
        """在测试集上进行预测和评估"""
        print("\n4. 测试集预测评估...")
        
        if len(self.test_daily) == 0:
            print("   ❌ 测试集中没有目标道路的洪水记录")
            return False
        
        predictions = []
        
        print(f"   测试集中有洪水的天数: {len(self.test_daily)}天")
        print(f"   逐日预测分析:")
        
        for date, roads_today in self.test_daily.items():
            print(f"\n     📅 {date}:")
            print(f"       实际洪水道路: {list(roads_today)}")
            
            # 为每条道路进行预测
            for target_road in self.selected_roads:
                target_node = target_road.replace(' ', '_').replace('-', '_')
                
                # 构建evidence（除目标道路外的其他道路）
                evidence = {}
                for other_road in self.selected_roads:
                    if other_road != target_road:
                        other_node = other_road.replace(' ', '_').replace('-', '_')
                        evidence[other_node] = 1 if other_road in roads_today else 0
                
                try:
                    # 贝叶斯推理
                    if len(evidence) > 0:
                        query_result = self.inference_engine.query(
                            variables=[target_node], 
                            evidence=evidence
                        )
                        prob_flood = query_result.values[1]
                    else:
                        # 没有evidence，使用先验概率
                        query_result = self.inference_engine.query(variables=[target_node])
                        prob_flood = query_result.values[1]
                    
                    # 真实标签
                    true_flood = 1 if target_road in roads_today else 0
                    
                    predictions.append({
                        'date': date,
                        'road': target_road,
                        'evidence': evidence,
                        'prob_flood': prob_flood,
                        'true_flood': true_flood
                    })
                    
                    print(f"       {target_road}:")
                    print(f"         Evidence: {evidence}")
                    print(f"         预测概率: {prob_flood:.4f}")
                    print(f"         真实标签: {true_flood}")
                    
                except Exception as e:
                    print(f"       {target_road}: 推理失败 - {e}")
        
        if len(predictions) == 0:
            print("   ❌ 没有有效的预测结果")
            return False
        
        # 计算性能指标
        print(f"\n   📈 性能评估:")
        
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        print(f"   {'阈值':<6} {'准确率':<8} {'精确率':<8} {'召回率':<8} {'F1':<8} {'TP':<3} {'TN':<3} {'FP':<3} {'FN':<3}")
        print(f"   {'-'*60}")
        
        for threshold in thresholds:
            y_true = [p['true_flood'] for p in predictions]
            y_pred = [1 if p['prob_flood'] >= threshold else 0 for p in predictions]
            
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (len(y_true), 0, 0, 0)
            
            print(f"   {threshold:<6.1f} {acc:<8.4f} {prec:<8.4f} {rec:<8.4f} {f1:<8.4f} "
                  f"{tp:<3} {tn:<3} {fp:<3} {fn:<3}")
        
        return predictions
        
    def analyze_performance_issues(self, predictions):
        """分析性能问题的根本原因"""
        print(f"\n5. 性能问题根本原因分析...")
        
        print(f"   🔍 关键发现:")
        
        # 1. 样本量分析
        total_predictions = len(predictions)
        positive_samples = sum(p['true_flood'] for p in predictions)
        print(f"     总预测样本: {total_predictions}个")
        print(f"     正样本(真实洪水): {positive_samples}个 ({positive_samples/total_predictions*100:.1f}%)")
        print(f"     负样本(无洪水): {total_predictions-positive_samples}个 ({(total_predictions-positive_samples)/total_predictions*100:.1f}%)")
        
        # 2. 训练-测试分布差异
        print(f"\n     训练集 vs 测试集对比:")
        print(f"     训练集洪水天数: {self.train_stats['total_days']}天")
        print(f"     测试集洪水天数: {self.test_stats['total_days']}天")
        print(f"     训练集共现天数: {self.train_stats['both_roads']}天")
        print(f"     测试集共现天数: {self.test_stats['both_roads']}天")
        
        # 3. 预测概率分布
        probs = [p['prob_flood'] for p in predictions]
        print(f"\n     预测概率分布:")
        print(f"     最小概率: {min(probs):.4f}")
        print(f"     最大概率: {max(probs):.4f}")
        print(f"     平均概率: {np.mean(probs):.4f}")
        print(f"     中位数概率: {np.median(probs):.4f}")
        
        # 4. 具体案例分析
        print(f"\n     典型预测案例:")
        for i, pred in enumerate(predictions[:5]):
            print(f"     案例{i+1}: {pred['date']} {pred['road']}")
            print(f"       Evidence: {pred['evidence']}")
            print(f"       预测: {pred['prob_flood']:.4f}, 真实: {pred['true_flood']}")
        
        # 5. 根本原因总结
        print(f"\n   💡 根本原因总结:")
        
        reasons = []
        if positive_samples == 0:
            reasons.append("测试集中无正样本，无法评估召回率")
        
        if self.test_stats['both_roads'] == 0:
            reasons.append("测试集中两条道路从未同时洪水，训练的条件概率无法验证")
        
        if self.train_stats['total_days'] < 10:
            reasons.append("训练数据量不足，统计不可靠")
        
        if max(probs) < 0.5:
            reasons.append("预测概率普遍偏低，阈值设置可能不当")
        
        for i, reason in enumerate(reasons, 1):
            print(f"     {i}. {reason}")
        
        if not reasons:
            print(f"     网络构建正常，可能是数据本身的预测难度较高")
        
        return True
        
    def run_experiment(self):
        """运行完整实验"""
        try:
            # 1. 数据加载和分割
            if not self.load_and_split_data():
                return False
            
            # 2. 分析道路模式
            if not self.analyze_road_patterns():
                return False
            
            # 3. 构建贝叶斯网络
            if not self.build_bayesian_network():
                return False
            
            # 4. 测试集评估
            predictions = self.evaluate_on_test_set()
            if not predictions:
                return False
            
            # 5. 性能问题分析
            self.analyze_performance_issues(predictions)
            
            print(f"\n✅ 两条道路贝叶斯网络实验完成！")
            print(f"🎯 通过这个简单案例，我们深入理解了贝叶斯网络性能问题的根源")
            
            return True
            
        except Exception as e:
            print(f"\n❌ 实验过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """主函数"""
    experiment = TwoRoadBayesianExperiment()
    experiment.run_experiment()

if __name__ == "__main__":
    main()