#!/usr/bin/env python3
"""
快速改进评估 - 不含可视化
"""

import random
import numpy as np
import pandas as pd
from model import FloodBayesNetwork
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def print_detailed_results(all_samples):
    """打印每个样本的详细预测过程"""
    print("\n" + "=" * 80)
    print("📋 详细样本预测结果")
    print("=" * 80)
    
    for i, sample in enumerate(all_samples, 1):
        sample_type = sample['type']
        evidence_roads = ', '.join(sample['evidence_roads'])
        target_road = sample['target_road']
        prob_flood = sample['prob_flood']
        prediction = sample['prediction']
        true_label = sample['true_label']
        is_correct = sample['is_correct']
        date = sample['date']
        
        # 预测结果文字描述
        if prediction == 1:
            pred_text = "洪水 (1)"
        elif prediction == 0:
            pred_text = "无洪水 (0)"
        else:
            pred_text = "不确定 (-1)"
        
        # 真实标签文字描述
        true_text = "洪水 (1)" if true_label == 1 else "无洪水 (0)"
        
        # 正确性标记
        if prediction == -1:
            correctness = "⚠️ 不确定"
        elif is_correct:
            correctness = "✅ 正确"
        else:
            correctness = "❌ 错误"
        
        print(f"\n样本 #{i} [{sample_type}] 日期: {date}")
        print(f"  Evidence道路: [{evidence_roads}]")
        print(f"  目标道路: {target_road}")
        print(f"  预测概率: {prob_flood:.3f} → 预测: {pred_text}")
        print(f"  真实标签: {true_text} → {correctness}")

def quick_evaluation():
    """快速评估"""
    print("🎯 快速改进评估")
    print("=" * 60)
    
    # 加载数据
    df = pd.read_csv("Road_Closures_2024.csv")
    df = df[df["REASON"].str.upper() == "FLOOD"].copy()
    
    # 预处理
    df["time_create"] = pd.to_datetime(df["START"], utc=True)
    df["link_id"] = df["STREET"].str.upper().str.replace(" ", "_")
    df["link_id"] = df["link_id"].astype(str)
    df["id"] = df["OBJECTID"].astype(str)
    
    # 时序分割
    df_sorted = df.sort_values('time_create')
    split_idx = int(len(df_sorted) * 0.7)
    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()
    
    # 构建网络
    flood_net = FloodBayesNetwork(t_window="D")
    flood_net.fit_marginal(train_df)
    flood_net.build_network_by_co_occurrence(
        train_df, occ_thr=3, edge_thr=2, weight_thr=0.3, report=False
    )
    flood_net.fit_conditional(train_df, max_parents=2, alpha=1.0)
    flood_net.build_bayes_network()
    
    print("✅ 网络构建完成")
    
    # 获取网络信息
    bn_nodes = set(flood_net.network_bayes.nodes())
    marginals_dict = dict(zip(flood_net.marginals['link_id'], flood_net.marginals['p']))
    
    # 使用优化后的阈值
    positive_threshold = 0.20
    negative_threshold = 0.05
    
    print(f"🎯 使用优化阈值: 正预测={positive_threshold:.3f}, 负预测={negative_threshold:.3f}")
    
    # 评估所有样本
    all_samples = []
    negative_candidates = [road for road, prob in marginals_dict.items() if road in bn_nodes and prob <= 0.15]
    
    test_by_date = test_df.groupby(test_df["time_create"].dt.floor("D"))
    evaluated_days = 0
    
    for date, day_group in test_by_date:
        flooded_roads = list(day_group["link_id"].unique())
        flooded_in_bn = [road for road in flooded_roads if road in bn_nodes]
        
        if len(flooded_in_bn) < 2:
            continue
            
        evaluated_days += 1
        
        # Evidence选择
        evidence_count = max(1, int(len(flooded_in_bn) * 0.5))
        evidence_roads = flooded_in_bn[:evidence_count]
        target_roads = flooded_in_bn[evidence_count:]
        evidence = {road: 1 for road in evidence_roads}
        
        # 处理正样本
        for target_road in target_roads:
            try:
                result = flood_net.infer_w_evidence(target_road, evidence)
                prob_flood = result['flooded']
                
                if prob_flood >= positive_threshold:
                    prediction = 1
                elif prob_flood <= negative_threshold:
                    prediction = 0
                else:
                    prediction = -1
                
                # 判断预测是否正确
                is_correct = (prediction == 1)  # 正样本真实标签是1
                
                all_samples.append({
                    'type': 'Positive',
                    'true_label': 1,
                    'prob_flood': prob_flood,
                    'prediction': prediction,
                    'evidence_roads': list(evidence.keys()),
                    'target_road': target_road,
                    'date': date.strftime('%Y-%m-%d'),
                    'is_correct': is_correct
                })
            except:
                continue
        
        # 处理负样本
        available_negatives = [road for road in negative_candidates if road not in flooded_roads]
        selected_negatives = available_negatives[:min(3, len(target_roads))]
        
        for neg_road in selected_negatives:
            try:
                result = flood_net.infer_w_evidence(neg_road, evidence)
                prob_flood = result['flooded']
                
                if prob_flood >= positive_threshold:
                    prediction = 1
                elif prob_flood <= negative_threshold:
                    prediction = 0
                else:
                    prediction = -1
                
                # 判断预测是否正确
                is_correct = (prediction == 0)  # 负样本真实标签是0
                
                all_samples.append({
                    'type': 'Negative',
                    'true_label': 0,
                    'prob_flood': prob_flood,
                    'prediction': prediction,
                    'evidence_roads': list(evidence.keys()),
                    'target_road': neg_road,
                    'date': date.strftime('%Y-%m-%d'),
                    'is_correct': is_correct
                })
            except:
                continue
    
    # 计算混淆矩阵
    tp = fp = tn = fn = uncertain = 0
    
    for sample in all_samples:
        pred = sample['prediction']
        true = sample['true_label']
        
        if pred == -1:
            uncertain += 1
        elif pred == 1 and true == 1:
            tp += 1
        elif pred == 1 and true == 0:
            fp += 1
        elif pred == 0 and true == 1:
            fn += 1
        elif pred == 0 and true == 0:
            tn += 1
    
    # 计算指标
    total_samples = len(all_samples)
    valid_predictions = tp + fp + tn + fn
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / valid_predictions if valid_predictions > 0 else 0.0
    
    # 输出结果
    print(f"\n📊 评估结果:")
    print(f"   评估天数: {evaluated_days}")
    print(f"   总样本数: {total_samples}")
    print(f"   有效预测: {valid_predictions}")
    print(f"   不确定预测: {uncertain}")
    
    print(f"\n📈 混淆矩阵:")
    print(f"                  预测")
    print(f"              正类    负类")
    print(f"    真实 正类  {tp:4d}   {fn:4d}")
    print(f"         负类  {fp:4d}   {tn:4d}")
    
    print(f"\n📋 详细分类:")
    print(f"   TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}, 不确定: {uncertain}")
    
    print(f"\n📈 性能指标:")
    print(f"   精确度 (Precision): {precision:.6f}")
    print(f"   召回率 (Recall):    {recall:.6f}")
    print(f"   F1分数 (F1-Score): {f1_score:.6f}")
    print(f"   准确率 (Accuracy):  {accuracy:.6f}")
    
    # 打印详细结果
    print_detailed_results(all_samples)
    
    # 保存详细的结果
    df_detailed = pd.DataFrame(all_samples)
    df_detailed.to_csv("detailed_evaluation_results.csv", index=False)
    print(f"\n✅ 详细结果已保存到 detailed_evaluation_results.csv")
    
    return {
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn, 'uncertain': uncertain,
        'precision': precision, 'recall': recall, 'f1_score': f1_score
    }

if __name__ == "__main__":
    results = quick_evaluation()