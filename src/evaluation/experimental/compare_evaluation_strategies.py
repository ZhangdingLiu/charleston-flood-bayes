#!/usr/bin/env python3
"""
比较评估策略：精确度优先 vs 传统洪水推理

对比两种评估方法：
1. 传统flood-only策略 (test_improved_eval.py)
2. 新的precision-focused策略 (precision_focused_evaluation.py)
"""

import random
import numpy as np
import pandas as pd
from model import FloodBayesNetwork
from precision_focused_evaluation import PrecisionFocusedEvaluator

# 设置随机种子确保可重现性
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def traditional_flood_only_evaluation(flood_net, test_df, prob_thr=0.5):
    """
    传统的flood-only评估策略（来自test_improved_eval.py）
    """
    bn_nodes = set(flood_net.network_bayes.nodes())
    
    all_predictions = []
    all_true_labels = []
    evaluated_days = 0
    
    # 按日期分组测试数据
    test_by_date = test_df.groupby(test_df["time_create"].dt.floor("D"))
    
    for date, day_group in test_by_date:
        # 当天洪水道路列表
        flooded_roads = list(day_group["link_id"].unique())
        
        # 只考虑在贝叶斯网络中的道路
        flooded_in_bn = [road for road in flooded_roads if road in bn_nodes]
        
        if len(flooded_in_bn) < 2:
            continue  # 需要至少2条道路才能做推理
        
        evaluated_days += 1
        
        # 选择第一条道路作为evidence
        evidence_road = flooded_in_bn[0]
        target_roads = flooded_in_bn[1:]
        
        evidence = {evidence_road: 1}
        
        # 对每个目标道路进行推理
        for target_road in target_roads:
            try:
                # 贝叶斯推理
                result = flood_net.infer_w_evidence(target_road, evidence)
                prob_flood = result['flooded']
                
                # 预测标签（根据概率阈值）
                pred_label = 1 if prob_flood >= prob_thr else 0
                true_label = 1  # 目标道路确实发生了洪水
                
                all_predictions.append(pred_label)
                all_true_labels.append(true_label)
                    
            except Exception as e:
                continue
    
    # 计算性能指标
    if len(all_predictions) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'samples': 0}
    
    # 在这种特殊设定下，所有true_label都是1
    tp = sum(all_predictions)  # 预测为正的数量
    fn = len(all_predictions) - tp  # 预测为负的数量
    
    precision = tp / tp if tp > 0 else 0.0  # 在这种设定下precision总是1.0或0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'samples': len(all_predictions),
        'evaluated_days': evaluated_days,
        'tp': tp,
        'fn': fn
    }

def load_data():
    """加载和预处理数据"""
    df = pd.read_csv("Road_Closures_2024.csv")
    df = df[df["REASON"].str.upper() == "FLOOD"].copy()
    
    # 数据预处理
    df["time_create"] = pd.to_datetime(df["START"], utc=True)
    df["link_id"] = df["STREET"].str.upper().str.replace(" ", "_")
    df["link_id"] = df["link_id"].astype(str)
    df["id"] = df["OBJECTID"].astype(str)
    
    # 时序分割，避免数据泄露
    df_sorted = df.sort_values('time_create')
    split_idx = int(len(df_sorted) * 0.7)
    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()
    
    return train_df, test_df

def build_network(train_df):
    """构建贝叶斯网络"""
    flood_net = FloodBayesNetwork(t_window="D")
    flood_net.fit_marginal(train_df)
    
    # 构建共现网络
    flood_net.build_network_by_co_occurrence(
        train_df,
        occ_thr=3,
        edge_thr=2,
        weight_thr=0.3,
        report=False
    )
    
    if flood_net.network.number_of_nodes() == 0:
        return None
    
    # 拟合条件概率
    flood_net.fit_conditional(train_df, max_parents=2, alpha=1.0)
    
    # 构建贝叶斯网络
    flood_net.build_bayes_network()
    
    return flood_net

def compare_strategies():
    """对比两种评估策略"""
    print("🔍 评估策略对比分析")
    print("=" * 60)
    
    # 1. 加载数据和构建网络
    train_df, test_df = load_data()
    flood_net = build_network(train_df)
    
    if flood_net is None:
        print("❌ 无法构建有效网络")
        return
    
    print(f"数据规模: 训练集{len(train_df)}条, 测试集{len(test_df)}条")
    print(f"网络规模: {flood_net.network.number_of_nodes()}个节点, {flood_net.network.number_of_edges()}条边")
    
    # 2. 传统flood-only评估
    print("\n📊 传统Flood-Only评估策略")
    print("-" * 40)
    
    traditional_results = {}
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    for thr in thresholds:
        result = traditional_flood_only_evaluation(flood_net, test_df, prob_thr=thr)
        traditional_results[thr] = result
        print(f"阈值{thr:.1f}: F1={result['f1']:.3f}, P={result['precision']:.3f}, R={result['recall']:.3f}, 样本={result['samples']}")
    
    # 找到传统方法的最佳结果
    best_traditional = max(traditional_results.values(), key=lambda x: x['f1'])
    best_thr = [k for k, v in traditional_results.items() if v == best_traditional][0]
    
    # 3. 精确度优先评估
    print("\n🎯 精确度优先评估策略")
    print("-" * 40)
    
    evaluator = PrecisionFocusedEvaluator(flood_net, test_df)
    
    # 优化阈值
    best_config = evaluator.optimize_thresholds_for_precision(
        target_precision=0.8, 
        min_recall=0.3
    )
    
    # 测试不同策略
    strategies = ['centrality', 'random', 'first']
    precision_results = {}
    
    for strategy in strategies:
        results = evaluator.evaluate_precision_focused(
            evidence_strategy=strategy, 
            include_negatives=True,
            verbose=False
        )
        metrics = evaluator.calculate_metrics(results)
        precision_results[strategy] = metrics
        print(f"{strategy}: F1={metrics['f1']:.3f}, P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, "
              f"样本={metrics['samples']}, 弃权率={metrics['abstention_rate']:.3f}")
    
    # 找到精确度方法的最佳结果
    best_precision = max(precision_results.values(), key=lambda x: x['f1'])
    best_strategy = [k for k, v in precision_results.items() if v == best_precision][0]
    
    # 4. 详细对比分析
    print("\n📈 详细对比分析")
    print("=" * 60)
    
    print(f"\n🔸 传统Flood-Only策略 (最佳阈值={best_thr})")
    print(f"  ✓ 精确度: {best_traditional['precision']:.3f}")
    print(f"  ✓ 召回率: {best_traditional['recall']:.3f}")
    print(f"  ✓ F1分数: {best_traditional['f1']:.3f}")
    print(f"  ✓ 评估样本: {best_traditional['samples']}")
    print(f"  ✓ 评估天数: {best_traditional['evaluated_days']}")
    print(f"  ⚠️  只测试正样本，精确度可能虚高")
    print(f"  ⚠️  单一evidence策略，信息利用不充分")
    
    print(f"\n🔸 精确度优先策略 (最佳策略={best_strategy})")
    print(f"  ✓ 精确度: {best_precision['precision']:.3f}")
    print(f"  ✓ 召回率: {best_precision['recall']:.3f}")
    print(f"  ✓ F1分数: {best_precision['f1']:.3f}")
    print(f"  ✓ 评估样本: {best_precision['samples']}")
    print(f"  ✓ 弃权率: {best_precision['abstention_rate']:.3f}")
    print(f"  ✓ 包含负样本测试，精确度更可靠")
    print(f"  ✓ 多evidence策略，信息利用更充分")
    print(f"  ✓ 双阈值系统，避免不确定预测")
    
    # 5. 关键优势分析
    print(f"\n🎯 精确度优先策略的关键优势")
    print("-" * 40)
    print(f"1. 📊 真实精确度测试: 包含{best_precision['negative_samples']}个负样本")
    print(f"2. 🎚️  保守预测策略: {best_precision['abstention_rate']:.1%}的预测被标记为不确定")
    print(f"3. 🔄 多样化evidence: 不同策略适应不同场景")
    print(f"4. 🎯 目标导向优化: 专门优化精确度≥0.8, 召回率≥0.3")
    print(f"5. 🚨 适应观测偏差: 专门处理Charleston警察数据特征")
    
    # 6. 适用场景建议
    print(f"\n💡 应用建议")
    print("-" * 40)
    print(f"🔸 传统策略适用于: 快速评估、概念验证")
    print(f"🔸 精确度策略适用于: 实际部署、高精度要求、观测偏差数据")
    print(f"🔸 推荐配置: {best_strategy}策略 + 阈值(正:{evaluator.positive_threshold:.2f}, 负:{evaluator.negative_threshold:.2f})")
    
    return {
        'traditional_best': best_traditional,
        'precision_best': best_precision,
        'recommendation': f"{best_strategy}策略"
    }

if __name__ == "__main__":
    results = compare_strategies()