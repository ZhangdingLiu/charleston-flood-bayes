#!/usr/bin/env python3
"""
分析激进策略的预测概率分布
找出为什么高阈值时precision和recall都是0
"""

import json

def analyze_aggressive_probabilities():
    # 加载详细结果
    with open('enhanced_coverage_validation_results_20250820_212943.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 找激进策略的实验
    aggressive_experiments = [exp for exp in data['detailed_results'] 
                            if exp['strategy'] == 'aggressive']
    
    print(f"🔍 分析激进策略实验数量: {len(aggressive_experiments)}")
    
    # 分析不同阈值下的情况
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    for threshold in thresholds:
        threshold_experiments = [exp for exp in aggressive_experiments 
                               if exp['pred_threshold'] == threshold]
        
        print(f"\n📊 阈值 {threshold} 分析:")
        print(f"   实验数量: {len(threshold_experiments)}")
        
        if threshold_experiments:
            # 取第一个实验分析
            exp = threshold_experiments[0]
            predictions = exp['detailed_predictions']
            
            # 统计概率分布
            probabilities = [p['predicted_probability'] for p in predictions if p['predicted_probability'] is not None]
            positive_labels = [p for p in predictions if p['true_label'] == 1]
            negative_labels = [p for p in predictions if p['true_label'] == 0]
            
            print(f"   总预测数: {len(predictions)}")
            print(f"   实际正样本数: {len(positive_labels)}")
            print(f"   实际负样本数: {len(negative_labels)}")
            
            if probabilities:
                print(f"   概率范围: {min(probabilities):.4f} - {max(probabilities):.4f}")
                print(f"   平均概率: {sum(probabilities)/len(probabilities):.4f}")
                
                # 统计超过阈值的预测
                above_threshold = [p for p in probabilities if p >= threshold]
                print(f"   超过阈值{threshold}的预测数: {len(above_threshold)}")
                
                # 混淆矩阵
                tp = exp['tp']
                fp = exp['fp'] 
                tn = exp['tn']
                fn = exp['fn']
                
                print(f"   混淆矩阵: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
                print(f"   Precision: {exp['precision']:.4f}")
                print(f"   Recall: {exp['recall']:.4f}")
                
                # 检查预测概率分布
                positive_probs = [p['predicted_probability'] for p in positive_labels if p['predicted_probability'] is not None]
                negative_probs = [p['predicted_probability'] for p in negative_labels if p['predicted_probability'] is not None]
                
                if positive_probs:
                    print(f"   正样本概率范围: {min(positive_probs):.4f} - {max(positive_probs):.4f}")
                if negative_probs:
                    print(f"   负样本概率范围: {min(negative_probs):.4f} - {max(negative_probs):.4f}")

if __name__ == "__main__":
    analyze_aggressive_probabilities()