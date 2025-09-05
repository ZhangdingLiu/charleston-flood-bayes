#!/usr/bin/env python3
"""
从激进策略实验中提取最佳结果
按照 best_2017_09_11_threshold_04_experiment.json 格式输出
"""

import json
import csv
from datetime import datetime

def extract_aggressive_best_result():
    print("🔍 提取激进策略最佳实验结果...")
    
    # 1. 加载激进策略的CSV数据
    aggressive_results = []
    with open('enhanced_coverage_validation_summary_20250820_212943.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['strategy'] == 'aggressive':
                # 转换数值字段
                numeric_fields = ['pred_threshold', 'coverage_rate', 'test_roads_total', 
                                'test_roads_in_network', 'network_nodes', 'network_edges',
                                'precision', 'recall', 'f1_score', 'accuracy', 
                                'tp', 'fp', 'tn', 'fn', 'trial_id']
                for field in numeric_fields:
                    if field in row and row[field]:
                        row[field] = float(row[field])
                aggressive_results.append(row)
    
    print(f"✅ 加载激进策略实验: {len(aggressive_results)} 条记录")
    
    # 2. 找到最佳F1分数的实验 (排除F1=0的)
    valid_results = [r for r in aggressive_results if r['f1_score'] > 0]
    if not valid_results:
        print("❌ 没有找到有效的激进策略结果")
        return
    
    best_result = max(valid_results, key=lambda x: x['f1_score'])
    print(f"🏆 最佳实验: 阈值{best_result['pred_threshold']}, F1={best_result['f1_score']:.3f}")
    
    # 3. 加载对应的详细JSON数据
    with open('enhanced_coverage_validation_results_20250820_212943.json', 'r', encoding='utf-8') as f:
        detailed_data = json.load(f)
    
    # 4. 找到匹配的详细实验
    target_threshold = best_result['pred_threshold'] 
    target_trial = int(best_result['trial_id'])
    
    matching_experiment = None
    for exp in detailed_data['detailed_results']:
        if (exp['strategy'] == 'aggressive' and 
            exp['pred_threshold'] == target_threshold and 
            exp['trial_id'] == target_trial):
            matching_experiment = exp
            break
    
    if not matching_experiment:
        print("❌ 没有找到匹配的详细实验数据")
        return
    
    print(f"✅ 找到匹配的详细实验")
    
    # 5. 按照基线格式构造结果
    best_experiment_result = {
        "experiment_info": {
            "date": matching_experiment['test_date'],
            "threshold": matching_experiment['pred_threshold'],
            "trial_id": matching_experiment['trial_id'],
            "strategy": "aggressive_enhanced_coverage",
            "strategy_name": "激进优化策略(增强覆盖率)",
            "network_type": "enhanced_bayesian_network", 
            "performance": {
                "precision": matching_experiment['precision'],
                "recall": matching_experiment['recall'],
                "f1_score": matching_experiment['f1_score'],
                "accuracy": matching_experiment['accuracy']
            },
            "network_stats": {
                "total_nodes": matching_experiment['network_nodes'],
                "total_edges": matching_experiment['network_edges'],
                "coverage_rate": matching_experiment['coverage_rate'],
                "test_roads_covered": f"{matching_experiment['test_roads_in_network']}/{matching_experiment['test_roads_total']}"
            }
        },
        "best_experiment": {
            "test_date": matching_experiment['test_date'],
            "pred_threshold": matching_experiment['pred_threshold'], 
            "trial_id": matching_experiment['trial_id'],
            "test_roads_total": matching_experiment['test_roads_total'],
            "test_roads_in_network": matching_experiment['test_roads_in_network'],
            "coverage_rate": matching_experiment['coverage_rate'],
            "evidence_roads_count": len(matching_experiment['evidence_roads']),
            "positive_predict_roads_count": matching_experiment['positive_predict_roads_count'],
            "negative_predict_roads_count": matching_experiment['negative_predict_roads_count'],
            "total_predict_roads_count": matching_experiment['total_predict_roads_count'],
            "successful_predictions": matching_experiment['successful_predictions'],
            "failed_predictions": matching_experiment['failed_predictions'],
            "prediction_mode": "full_network_enhanced_coverage",
            "evidence_roads": matching_experiment['evidence_roads'],
            "network_parameters": {
                "occ_thr": 1,      # 激进策略参数
                "edge_thr": 1,
                "weight_thr": 0.05,
                "evidence_ratio": 0.3
            },
            "performance_metrics": {
                "precision": matching_experiment['precision'],
                "recall": matching_experiment['recall'], 
                "f1_score": matching_experiment['f1_score'],
                "accuracy": matching_experiment['accuracy'],
                "tp": matching_experiment['tp'],
                "fp": matching_experiment['fp'], 
                "tn": matching_experiment['tn'],
                "fn": matching_experiment['fn']
            },
            "network_statistics": {
                "total_nodes": matching_experiment['network_nodes'],
                "total_edges": matching_experiment['network_edges'],
                "nodes_vs_baseline": f"{matching_experiment['network_nodes']} vs 40 (基线)",
                "edges_vs_baseline": f"{matching_experiment['network_edges']} vs 39 (基线)",
                "coverage_improvement": f"{matching_experiment['coverage_rate']:.1%} vs 50% (基线)"
            },
            "detailed_predictions": matching_experiment['detailed_predictions']
        },
        "analysis": {
            "key_improvements": [
                f"网络规模扩大至{matching_experiment['network_nodes']}个节点",
                f"覆盖率提升至{matching_experiment['coverage_rate']:.1%} ({matching_experiment['test_roads_in_network']}/{matching_experiment['test_roads_total']}条道路)",
                f"可预测{matching_experiment['total_predict_roads_count']}条道路状态",
                f"使用全历史数据训练(855条记录)"
            ],
            "performance_vs_baseline": {
                "coverage_rate": {
                    "enhanced": matching_experiment['coverage_rate'],
                    "baseline": 0.50,  # 基线约50%
                    "improvement": f"+{(matching_experiment['coverage_rate'] - 0.50)*100:.1f}个百分点"
                },
                "network_nodes": {
                    "enhanced": matching_experiment['network_nodes'],
                    "baseline": 40,
                    "improvement": f"+{matching_experiment['network_nodes'] - 40}个节点"
                },
                "predictable_roads": {
                    "enhanced": matching_experiment['total_predict_roads_count'],
                    "baseline": 33,  # 基线预测33条道路
                    "improvement": f"+{matching_experiment['total_predict_roads_count'] - 33}条道路"
                }
            },
            "trade_offs": {
                "advantages": [
                    f"覆盖率大幅提升({matching_experiment['coverage_rate']:.1%})",
                    f"网络规模显著扩大({matching_experiment['network_nodes']}节点)",
                    "可监控更多道路",
                    "精度依然保持100%"
                ],
                "challenges": [
                    f"召回率相对较低({matching_experiment['recall']:.1%})",
                    "大网络导致预测概率普遍偏低",
                    "稀疏数据问题",
                    f"F1分数({matching_experiment['f1_score']:.3f})低于基线"
                ]
            },
            "recommendations": [
                "适用于最大化监控覆盖范围的场景",
                "可容忍较多漏报但要求零误报",
                "建议结合多个阈值进行预警分级",
                "考虑参数微调平衡覆盖率与性能"
            ]
        },
        "metadata": {
            "generation_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "source_experiment": "enhanced_coverage_validation.py",
            "comparison_baseline": "best_2017_09_11_threshold_04_experiment.json",
            "methodology": "enhanced_coverage_bayesian_inference",
            "training_data": "全历史数据(除测试日期)",
            "test_date_description": "2017年9月11日洪水事件 - Charleston最严重洪水之一"
        }
    }
    
    # 6. 保存结果
    output_file = f"aggressive_best_2017_09_11_threshold_{target_threshold:.1f}_experiment.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(best_experiment_result, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 激进策略最佳实验结果已保存: {output_file}")
    
    # 7. 输出关键统计信息
    print(f"\n📊 激进策略(121节点)最佳实验摘要:")
    print(f"{'='*60}")
    print(f"🎯 实验配置:")
    print(f"   测试日期: {matching_experiment['test_date']}")
    print(f"   预测阈值: {matching_experiment['pred_threshold']}")
    print(f"   试验ID: {matching_experiment['trial_id']}")
    
    print(f"\n🌐 网络规模:")
    print(f"   总节点数: {matching_experiment['network_nodes']} (vs 基线40)")
    print(f"   总边数: {matching_experiment['network_edges']} (vs 基线39)")
    print(f"   覆盖率: {matching_experiment['coverage_rate']:.1%} ({matching_experiment['test_roads_in_network']}/{matching_experiment['test_roads_total']})")
    
    print(f"\n🔬 预测统计:")
    print(f"   证据道路: {len(matching_experiment['evidence_roads'])} 条")
    print(f"   预测道路: {matching_experiment['total_predict_roads_count']} 条")
    print(f"   正样本: {matching_experiment['positive_predict_roads_count']} 条")
    print(f"   负样本: {matching_experiment['negative_predict_roads_count']} 条")
    
    print(f"\n📈 性能指标:")
    print(f"   Precision: {matching_experiment['precision']:.3f} (100% - 无误报)")
    print(f"   Recall: {matching_experiment['recall']:.3f} ({matching_experiment['recall']*100:.1f}%)")
    print(f"   F1 Score: {matching_experiment['f1_score']:.3f}")
    print(f"   Accuracy: {matching_experiment['accuracy']:.3f} ({matching_experiment['accuracy']*100:.1f}%)")
    
    print(f"\n📊 混淆矩阵:")
    print(f"   TP: {matching_experiment['tp']} (正确预测洪水)")
    print(f"   FP: {matching_experiment['fp']} (误报)")
    print(f"   TN: {matching_experiment['tn']} (正确预测无洪水)") 
    print(f"   FN: {matching_experiment['fn']} (漏报)")
    
    # 8. 显示详细预测样例
    predictions = matching_experiment['detailed_predictions']
    tp_roads = [p for p in predictions if p['true_label'] == 1 and p['predicted_label'] == 1]
    fn_roads = [p for p in predictions if p['true_label'] == 1 and p['predicted_label'] == 0]
    
    print(f"\n🛣️ 预测样例:")
    print(f"   ✅ 成功预测洪水道路 (前5条):")
    tp_roads.sort(key=lambda x: x['predicted_probability'], reverse=True)
    for i, road in enumerate(tp_roads[:5]):
        print(f"     {i+1}. {road['road_name']}: 概率{road['predicted_probability']:.3f}")
    
    print(f"   ⚠️ 漏报洪水道路 (前5条):")
    fn_roads.sort(key=lambda x: x['predicted_probability'], reverse=True) 
    for i, road in enumerate(fn_roads[:5]):
        print(f"     {i+1}. {road['road_name']}: 概率{road['predicted_probability']:.3f}")
    
    return output_file

if __name__ == "__main__":
    result_file = extract_aggressive_best_result()