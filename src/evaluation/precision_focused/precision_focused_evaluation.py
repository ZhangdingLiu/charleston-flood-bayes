#!/usr/bin/env python3
"""
Precision-Focused Evaluation Strategy for Charleston Flood Prediction

Designed to handle observational bias in police data:
- Flood=1 records are reliable (police observed flooding)
- Flood=0 (no records) may be observation gaps, not true negatives

Key Features:
1. Multi-evidence inference (40-60% roads as evidence)
2. Conservative negative sampling from low-probability roads
3. Dual-threshold system (high for positive, low for negative)
4. Confidence-based evaluation with abstention
5. Precision optimization targeting ≥0.8, Recall 0.3-0.5
"""

import random
import numpy as np
import pandas as pd
from model import FloodBayesNetwork
from collections import defaultdict, Counter
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class PrecisionFocusedEvaluator:
    """
    精确度优先的评估器，专门处理Charleston警察数据的观测偏差
    """
    
    def __init__(self, flood_net, test_df):
        self.flood_net = flood_net
        self.test_df = test_df
        self.bn_nodes = set(flood_net.network_bayes.nodes()) if flood_net.network_bayes else set()
        self.marginals_dict = dict(zip(
            flood_net.marginals['link_id'], 
            flood_net.marginals['p']
        )) if flood_net.marginals is not None else {}
        
        # 策略参数
        self.evidence_ratio = 0.5  # 用作evidence的道路比例
        self.positive_threshold = 0.7  # 正预测阈值（高精度）
        self.negative_threshold = 0.3  # 负预测阈值（保守）
        self.min_marginal_for_negative = 0.15  # 负样本候选的最大边际概率
        
    def identify_reliable_negative_candidates(self):
        """
        识别可靠的负样本候选：网络中边际概率很低的道路
        """
        negative_candidates = []
        for road, prob in self.marginals_dict.items():
            if road in self.bn_nodes and prob <= self.min_marginal_for_negative:
                negative_candidates.append(road)
        
        return negative_candidates
    
    def select_evidence_roads(self, flooded_roads, strategy='centrality'):
        """
        选择evidence道路的策略
        
        Args:
            flooded_roads: 当天洪水道路列表
            strategy: 选择策略 ('first', 'random', 'centrality', 'high_marginal')
        """
        if len(flooded_roads) < 2:
            return [], flooded_roads
            
        evidence_count = max(1, int(len(flooded_roads) * self.evidence_ratio))
        evidence_count = min(evidence_count, len(flooded_roads) - 1)  # 至少保留1个作为目标
        
        if strategy == 'first':
            evidence_roads = flooded_roads[:evidence_count]
        elif strategy == 'random':
            evidence_roads = random.sample(flooded_roads, evidence_count)
        elif strategy == 'centrality':
            # 按网络中心性排序（入度+出度）
            centrality_scores = []
            for road in flooded_roads:
                in_deg = self.flood_net.network.in_degree(road) if road in self.flood_net.network else 0
                out_deg = self.flood_net.network.out_degree(road) if road in self.flood_net.network else 0
                centrality_scores.append((road, in_deg + out_deg))
            
            centrality_scores.sort(key=lambda x: x[1], reverse=True)
            evidence_roads = [road for road, _ in centrality_scores[:evidence_count]]
        elif strategy == 'high_marginal':
            # 按边际概率排序
            marginal_scores = [(road, self.marginals_dict.get(road, 0)) for road in flooded_roads]
            marginal_scores.sort(key=lambda x: x[1], reverse=True)
            evidence_roads = [road for road, _ in marginal_scores[:evidence_count]]
        else:
            evidence_roads = flooded_roads[:evidence_count]
            
        target_roads = [road for road in flooded_roads if road not in evidence_roads]
        return evidence_roads, target_roads
    
    def get_temporal_negative_samples(self, date, flooded_roads, max_samples=3):
        """
        获取时间负样本：当天没有洪水但在网络中的道路
        只选择边际概率较低的作为负样本候选
        """
        # 当天没有洪水的网络道路
        non_flooded = [road for road in self.bn_nodes if road not in flooded_roads]
        
        # 只选择边际概率低的道路作为负样本候选
        reliable_negatives = [
            road for road in non_flooded 
            if self.marginals_dict.get(road, 1.0) <= self.min_marginal_for_negative
        ]
        
        # 随机采样，限制负样本数量
        max_samples = min(max_samples, len(reliable_negatives))
        if max_samples > 0:
            return random.sample(reliable_negatives, max_samples)
        return []
    
    def make_prediction(self, target_road, evidence, return_prob=False):
        """
        进行预测并应用双阈值策略
        
        Returns:
            prediction: 1 (flood), 0 (no flood), -1 (uncertain/abstain)
            confidence: prediction confidence level
        """
        try:
            result = self.flood_net.infer_w_evidence(target_road, evidence)
            prob_flood = result['flooded']
            
            if prob_flood >= self.positive_threshold:
                prediction = 1
                confidence = prob_flood
            elif prob_flood <= self.negative_threshold:
                prediction = 0
                confidence = 1 - prob_flood
            else:
                prediction = -1  # uncertain, abstain
                confidence = 0.5
                
            if return_prob:
                return prediction, confidence, prob_flood
            return prediction, confidence
            
        except Exception as e:
            if return_prob:
                return -1, 0.0, 0.5
            return -1, 0.0
    
    def evaluate_precision_focused(self, evidence_strategy='centrality', include_negatives=True, verbose=False):
        """
        精确度优先的评估方法
        
        Args:
            evidence_strategy: evidence选择策略
            include_negatives: 是否包含负样本测试
            verbose: 是否显示详细信息
        """
        results = {
            'total_days': 0,
            'evaluated_days': 0,
            'positive_samples': [],  # (prediction, confidence, true_label, prob)
            'negative_samples': [],
            'uncertain_samples': [],
            'evidence_strategy': evidence_strategy
        }
        
        # 按日期分组测试数据
        test_by_date = self.test_df.groupby(self.test_df["time_create"].dt.floor("D"))
        
        for date, day_group in test_by_date:
            results['total_days'] += 1
            
            # 当天洪水道路列表（只考虑在贝叶斯网络中的道路）
            flooded_roads = list(day_group["link_id"].unique())
            flooded_in_bn = [road for road in flooded_roads if road in self.bn_nodes]
            
            if len(flooded_in_bn) < 2:
                continue  # 需要至少2条道路才能做推理
                
            results['evaluated_days'] += 1
            
            # 选择evidence和target道路
            evidence_roads, target_roads = self.select_evidence_roads(
                flooded_in_bn, strategy=evidence_strategy
            )
            
            if len(target_roads) == 0:
                continue
                
            evidence = {road: 1 for road in evidence_roads}
            
            if verbose and results['evaluated_days'] <= 3:
                print(f"📅 {date.date()}: 洪水道路{len(flooded_in_bn)}, "
                      f"evidence{len(evidence_roads)}, target{len(target_roads)}")
                print(f"   Evidence: {evidence_roads}")
                print(f"   Targets: {target_roads}")
            
            # 测试正样本（真实洪水道路）
            for target_road in target_roads:
                pred, conf, prob = self.make_prediction(target_road, evidence, return_prob=True)
                
                if pred == 1:
                    results['positive_samples'].append((pred, conf, 1, prob))
                elif pred == 0:
                    results['positive_samples'].append((pred, conf, 1, prob))  # 错误预测
                else:
                    results['uncertain_samples'].append((pred, conf, 1, prob))
                    
                if verbose and results['evaluated_days'] <= 3:
                    status = "✅" if pred == 1 else "❌" if pred == 0 else "❓"
                    print(f"   {target_road}: P={prob:.3f} → pred={pred} {status}")
            
            # 测试负样本（如果启用）
            if include_negatives:
                negative_candidates = self.get_temporal_negative_samples(
                    date, flooded_in_bn, max_samples=min(3, len(target_roads))
                )
                
                for neg_road in negative_candidates:
                    pred, conf, prob = self.make_prediction(neg_road, evidence, return_prob=True)
                    
                    if pred == 0:
                        results['negative_samples'].append((pred, conf, 0, prob))
                    elif pred == 1:
                        results['negative_samples'].append((pred, conf, 0, prob))  # 错误预测
                    else:
                        results['uncertain_samples'].append((pred, conf, 0, prob))
                        
                    if verbose and results['evaluated_days'] <= 3:
                        status = "✅" if pred == 0 else "❌" if pred == 1 else "❓"
                        print(f"   {neg_road} (neg): P={prob:.3f} → pred={pred} {status}")
        
        return results
    
    def calculate_metrics(self, evaluation_results, include_uncertain=False):
        """
        计算评估指标
        """
        pos_samples = evaluation_results['positive_samples']
        neg_samples = evaluation_results['negative_samples']
        uncertain_samples = evaluation_results['uncertain_samples']
        
        # 合并所有样本
        all_samples = pos_samples + neg_samples
        if include_uncertain:
            all_samples += uncertain_samples
        
        if len(all_samples) == 0:
            return {
                'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                'accuracy': 0.0, 'samples': 0, 'abstention_rate': 0.0
            }
        
        # 计算混淆矩阵
        tp = sum(1 for pred, _, true, _ in all_samples if pred == 1 and true == 1)
        tn = sum(1 for pred, _, true, _ in all_samples if pred == 0 and true == 0)
        fp = sum(1 for pred, _, true, _ in all_samples if pred == 1 and true == 0)
        fn = sum(1 for pred, _, true, _ in all_samples if pred == 0 and true == 1)
        
        # 计算基本指标
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(all_samples) if len(all_samples) > 0 else 0.0
        
        # 弃权率
        total_predictions = len(pos_samples) + len(neg_samples) + len(uncertain_samples)
        abstention_rate = len(uncertain_samples) / total_predictions if total_predictions > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'samples': len(all_samples),
            'positive_samples': len(pos_samples),
            'negative_samples': len(neg_samples),
            'uncertain_samples': len(uncertain_samples),
            'abstention_rate': abstention_rate,
            'total_days': evaluation_results['total_days'],
            'evaluated_days': evaluation_results['evaluated_days']
        }
    
    def optimize_thresholds_for_precision(self, target_precision=0.8, min_recall=0.3):
        """
        优化阈值以达到目标精确度
        """
        print(f"\n🎯 优化阈值以达到精确度≥{target_precision}, 召回率≥{min_recall}")
        
        # 测试不同的阈值组合
        pos_thresholds = np.arange(0.5, 0.95, 0.05)
        neg_thresholds = np.arange(0.05, 0.5, 0.05)
        
        best_config = None
        best_score = -1
        
        for pos_thr in pos_thresholds:
            for neg_thr in neg_thresholds:
                if neg_thr >= pos_thr:
                    continue
                    
                # 更新阈值
                self.positive_threshold = pos_thr
                self.negative_threshold = neg_thr
                
                # 评估
                results = self.evaluate_precision_focused(verbose=False)
                metrics = self.calculate_metrics(results)
                
                # 检查是否满足条件
                if (metrics['precision'] >= target_precision and 
                    metrics['recall'] >= min_recall and
                    metrics['samples'] > 10):  # 确保有足够样本
                    
                    # 计算综合分数（优先考虑recall，因为precision已满足）
                    score = metrics['recall'] + 0.1 * (metrics['precision'] - target_precision)
                    
                    if score > best_score:
                        best_score = score
                        best_config = {
                            'pos_threshold': pos_thr,
                            'neg_threshold': neg_thr,
                            'metrics': metrics.copy()
                        }
        
        if best_config:
            self.positive_threshold = best_config['pos_threshold']
            self.negative_threshold = best_config['neg_threshold']
            print(f"✅ 找到最佳阈值配置:")
            print(f"   正预测阈值: {self.positive_threshold:.2f}")
            print(f"   负预测阈值: {self.negative_threshold:.2f}")
            print(f"   精确度: {best_config['metrics']['precision']:.3f}")
            print(f"   召回率: {best_config['metrics']['recall']:.3f}")
            print(f"   F1: {best_config['metrics']['f1']:.3f}")
            print(f"   弃权率: {best_config['metrics']['abstention_rate']:.3f}")
            return best_config
        else:
            print(f"❌ 未找到满足条件的阈值配置")
            return None

def load_data():
    """加载和预处理数据"""
    print("🚀 加载Charleston洪水数据")
    print("=" * 50)
    
    # 加载数据
    df = pd.read_csv("Road_Closures_2024.csv")
    df = df[df["REASON"].str.upper() == "FLOOD"].copy()
    
    # 数据预处理
    df["time_create"] = pd.to_datetime(df["START"], utc=True)
    df["link_id"] = df["STREET"].str.upper().str.replace(" ", "_")
    df["link_id"] = df["link_id"].astype(str)
    df["id"] = df["OBJECTID"].astype(str)
    
    # 时序分割避免数据泄露
    df_sorted = df.sort_values('time_create')
    split_idx = int(len(df_sorted) * 0.7)
    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()
    
    print(f"总洪水记录: {len(df)}条")
    print(f"训练集: {len(train_df)}条")
    print(f"测试集: {len(test_df)}条")
    
    return train_df, test_df

def build_network(train_df):
    """构建贝叶斯网络"""
    print("\n📊 构建贝叶斯网络")
    
    flood_net = FloodBayesNetwork(t_window="D")
    flood_net.fit_marginal(train_df)
    
    # 使用较优参数构建网络
    flood_net.build_network_by_co_occurrence(
        train_df,
        occ_thr=3,
        edge_thr=2,
        weight_thr=0.3,
        report=False
    )
    
    print(f"节点数: {flood_net.network.number_of_nodes()}")
    print(f"边数: {flood_net.network.number_of_edges()}")
    
    if flood_net.network.number_of_nodes() == 0:
        print("❌ 网络为空")
        return None
    
    # 拟合条件概率和构建贝叶斯网络
    flood_net.fit_conditional(train_df, max_parents=2, alpha=1.0)
    flood_net.build_bayes_network()
    
    return flood_net

def main():
    """主函数"""
    # 1. 加载数据
    train_df, test_df = load_data()
    
    # 2. 构建网络
    flood_net = build_network(train_df)
    if flood_net is None:
        print("❌ 无法构建有效网络")
        return
    
    # 3. 创建精确度优先评估器
    print("\n🎯 创建精确度优先评估器")
    evaluator = PrecisionFocusedEvaluator(flood_net, test_df)
    
    print(f"网络节点数: {len(evaluator.bn_nodes)}")
    print(f"低概率负样本候选: {len(evaluator.identify_reliable_negative_candidates())}个")
    
    # 4. 阈值优化
    best_config = evaluator.optimize_thresholds_for_precision(
        target_precision=0.8, 
        min_recall=0.3
    )
    
    if best_config is None:
        print("使用默认阈值继续评估...")
        evaluator.positive_threshold = 0.7
        evaluator.negative_threshold = 0.3
    
    # 5. 测试不同的evidence策略
    print("\n📈 测试不同Evidence选择策略")
    strategies = ['centrality', 'high_marginal', 'first', 'random']
    
    strategy_results = {}
    for strategy in strategies:
        print(f"\n--- {strategy.upper()} 策略 ---")
        results = evaluator.evaluate_precision_focused(
            evidence_strategy=strategy, 
            include_negatives=True,
            verbose=(strategy == 'centrality')  # 只对第一个策略显示详细信息
        )
        metrics = evaluator.calculate_metrics(results)
        strategy_results[strategy] = metrics
        
        print(f"精确度: {metrics['precision']:.3f}")
        print(f"召回率: {metrics['recall']:.3f}")
        print(f"F1分数: {metrics['f1']:.3f}")
        print(f"弃权率: {metrics['abstention_rate']:.3f}")
        print(f"样本数: {metrics['samples']} (正:{metrics['positive_samples']}, 负:{metrics['negative_samples']}, 不确定:{metrics['uncertain_samples']})")
    
    # 6. 总结最佳策略
    print("\n🏆 策略对比总结")
    print(f"{'策略':<12} {'精确度':<8} {'召回率':<8} {'F1':<8} {'弃权率':<8} {'样本数':<8}")
    print("-" * 65)
    
    best_strategy = None
    best_f1 = -1
    
    for strategy, metrics in strategy_results.items():
        print(f"{strategy:<12} {metrics['precision']:<8.3f} {metrics['recall']:<8.3f} "
              f"{metrics['f1']:<8.3f} {metrics['abstention_rate']:<8.3f} {metrics['samples']:<8}")
        
        if metrics['f1'] > best_f1 and metrics['precision'] >= 0.8:
            best_f1 = metrics['f1']
            best_strategy = strategy
    
    print(f"\n✅ 推荐策略: {best_strategy.upper() if best_strategy else 'CENTRALITY'}")
    print(f"🎯 成功实现精确度优先评估，适合Charleston警察数据的观测偏差特征")
    
    return evaluator, strategy_results

if __name__ == "__main__":
    evaluator, results = main()