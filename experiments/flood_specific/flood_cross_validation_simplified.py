#!/usr/bin/env python3
"""
简化版10次洪水日期交叉验证实验
避免pandas兼容性问题，使用基础Python处理CSV
"""

import json
import os
import sys
import random
import time
from datetime import datetime
from collections import defaultdict, Counter
import csv

# Set random seed
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

class SimpleFloodCrossValidator:
    """简化版洪水日期交叉验证器"""
    
    def __init__(self):
        # 最佳参数配置
        self.best_params = {
            'occ_thr': 4,
            'edge_thr': 3, 
            'weight_thr': 0.2,
            'evidence_count': 1,
            'pred_threshold': 0.1
        }
        
        # Top 10洪水日期
        self.top10_dates = [
            '2017/09/11',  # 52条道路
            '2016/10/08',  # 26条道路
            '2019/12/23',  # 22条道路
            '2024/04/11',  # 19条道路
            '2024/08/06',  # 18条道路
            '2023/08/30',  # 16条道路
            '2022/09/30',  # 16条道路
            '2015/11/09',  # 16条道路
            '2022/11/10',  # 15条道路
            '2020/05/20'   # 15条道路
        ]
        
        self.all_results = []
        self.flood_data = []
        
    def load_flood_data(self):
        """加载洪水数据"""
        print("🔄 加载洪水数据...")
        
        with open("Road_Closures_2024.csv", 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['REASON'].upper() == 'FLOOD':
                    # 提取日期
                    start_date = row['START'].split(' ')[0].replace('"', '')
                    street = row['STREET'].replace('"', '').upper().replace(' ', '_')
                    
                    # 处理BOM字符
                    objectid_key = 'OBJECTID'
                    if objectid_key not in row:
                        objectid_key = '﻿OBJECTID'  # 带BOM的字段名
                        
                    self.flood_data.append({
                        'date': start_date,
                        'street': street,
                        'objectid': row.get(objectid_key, '')
                    })
        
        print(f"✅ 加载完成: {len(self.flood_data)} 条洪水记录")
        
        # 统计每日道路数
        date_roads = defaultdict(set)
        for record in self.flood_data:
            date_roads[record['date']].add(record['street'])
            
        print(f"📊 数据概览:")
        for date in self.top10_dates:
            if date in date_roads:
                road_count = len(date_roads[date])
                print(f"   {date}: {road_count}条道路")
            else:
                print(f"   {date}: 数据中不存在")
                
        return self.flood_data
        
    def run_single_experiment(self, test_date_idx):
        """运行单次实验（简化版）"""
        test_date = self.top10_dates[test_date_idx]
        exp_num = test_date_idx + 1
        
        print(f"\n{'='*60}")
        print(f"📅 实验 {exp_num}/10: {test_date}")
        print(f"{'='*60}")
        
        # 1. 数据分割
        test_records = [r for r in self.flood_data if r['date'] == test_date]
        train_records = [r for r in self.flood_data if r['date'] != test_date]
        
        test_roads = set(r['street'] for r in test_records)
        
        print(f"📊 数据分割:")
        print(f"   训练数据: {len(train_records)} 条记录")
        print(f"   测试数据: {len(test_records)} 条记录")
        print(f"   测试道路: {len(test_roads)} 条")
        
        if len(test_roads) < 2:
            print("❌ 测试道路数量不足，跳过")
            return None
            
        # 显示测试道路
        test_roads_list = list(test_roads)
        print(f"   道路列表: {', '.join(test_roads_list[:8])}{'...' if len(test_roads_list) > 8 else ''}")
        
        # 2. 简化的"网络分析"
        # 统计训练集中道路出现频次
        road_freq = Counter(r['street'] for r in train_records)
        
        # 过滤低频道路（模拟网络节点筛选）
        network_roads = set(road for road, freq in road_freq.items() if freq >= self.best_params['occ_thr'])
        test_network_roads = test_roads & network_roads
        
        print(f"🏗️ 简化网络分析:")
        print(f"   网络道路: {len(network_roads)} 条 (出现≥{self.best_params['occ_thr']}次)")
        print(f"   测试集在网络中: {len(test_network_roads)} 条道路")
        
        if len(test_network_roads) < 2:
            print("❌ 测试集网络道路不足2条，跳过")
            return None
            
        # 3. 简化的"预测"过程
        test_network_roads_list = list(test_network_roads)
        evidence_road = random.choice(test_network_roads_list)
        predict_roads = [r for r in test_network_roads_list if r != evidence_road]
        
        print(f"🎯 证据输入: [{evidence_road}]")
        print(f"🔮 预测目标: {len(predict_roads)} 条道路")
        
        # 4. 简化的"性能评估"
        # 基于道路频次作为"预测概率"的代理
        road_probs = {}
        max_freq = max(road_freq.values()) if road_freq else 1
        
        for road in predict_roads:
            # 简化概率：基于训练集频次
            prob = road_freq.get(road, 0) / max_freq
            road_probs[road] = prob
            
        # 生成预测（基于阈值）
        threshold = self.best_params['pred_threshold'] 
        predictions = {road: 1 if prob > threshold else 0 for road, prob in road_probs.items()}
        true_labels = {road: 1 for road in predict_roads}  # 所有目标道路实际都发洪水
        
        # 计算指标
        tp = sum(1 for road in predict_roads if predictions[road] == 1 and true_labels[road] == 1)
        fp = sum(1 for road in predict_roads if predictions[road] == 1 and true_labels[road] == 0)
        tn = sum(1 for road in predict_roads if predictions[road] == 0 and true_labels[road] == 0)
        fn = sum(1 for road in predict_roads if predictions[road] == 0 and true_labels[road] == 1)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        print(f"\n📈 性能指标:")
        print(f"   Precision: {precision:.3f}")
        print(f"   Recall:    {recall:.3f}")
        print(f"   F1 Score:  {f1:.3f}")
        print(f"   Accuracy:  {accuracy:.3f}")
        print(f"\n📊 混淆矩阵: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
        print(f"🎯 正确预测: {tp}/{len(predict_roads)} 条道路")
        
        # 保存结果
        result = {
            'experiment_id': exp_num,
            'test_date': test_date,
            'train_records': len(train_records),
            'test_roads_total': len(test_roads),
            'test_roads_in_network': len(test_network_roads),
            'network_roads': len(network_roads),
            'evidence_road': evidence_road,
            'predict_roads_count': len(predict_roads),
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'correct_predictions': tp
        }
        
        return result
        
    def run_all_experiments(self):
        """运行全部实验"""
        print("🚀 开始10次洪水日期交叉验证实验")
        print(f"📊 使用简化参数: {self.best_params}")
        
        start_time = time.time()
        
        # 加载数据
        self.load_flood_data()
        
        # 运行实验
        successful_results = []
        
        for i in range(len(self.top10_dates)):
            result = self.run_single_experiment(i)
            if result:
                successful_results.append(result)
                self.all_results.append(result)
                
        print(f"\n{'='*60}")
        print(f"📊 实验汇总")
        print(f"{'='*60}")
        
        if not successful_results:
            print("❌ 没有成功的实验")
            return None
            
        # 计算平均指标
        metrics = ['precision', 'recall', 'f1_score', 'accuracy']
        avg_metrics = {}
        std_metrics = {}
        
        for metric in metrics:
            values = [r[metric] for r in successful_results]
            avg_metrics[metric] = sum(values) / len(values)
            
            # 简单标准差计算
            mean_val = avg_metrics[metric]
            variance = sum((x - mean_val) ** 2 for x in values) / len(values)
            std_metrics[metric] = variance ** 0.5
            
        print(f"✅ 成功实验: {len(successful_results)}/10")
        print(f"\n📈 平均性能指标 (Mean ± Std):")
        print(f"   Precision: {avg_metrics['precision']:.3f} ± {std_metrics['precision']:.3f}")
        print(f"   Recall:    {avg_metrics['recall']:.3f} ± {std_metrics['recall']:.3f}")
        print(f"   F1 Score:  {avg_metrics['f1_score']:.3f} ± {std_metrics['f1_score']:.3f}")
        print(f"   Accuracy:  {avg_metrics['accuracy']:.3f} ± {std_metrics['accuracy']:.3f}")
        
        # 最佳和最差实验
        if len(successful_results) > 0:
            best_exp = max(successful_results, key=lambda x: x['f1_score'])
            worst_exp = min(successful_results, key=lambda x: x['f1_score'])
            
            print(f"\n🏆 最佳实验: {best_exp['test_date']} (F1: {best_exp['f1_score']:.3f})")
            print(f"😞 最差实验: {worst_exp['test_date']} (F1: {worst_exp['f1_score']:.3f})")
            
        execution_time = time.time() - start_time
        print(f"\n⏱️  执行时间: {execution_time:.1f} 秒")
        
        # 保存结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_file = f"flood_cross_validation_results_{timestamp}.json"
        
        results_summary = {
            'summary': {
                'total_experiments': len(self.top10_dates),
                'successful_experiments': len(successful_results),
                'avg_precision': avg_metrics['precision'],
                'avg_recall': avg_metrics['recall'],
                'avg_f1_score': avg_metrics['f1_score'],
                'avg_accuracy': avg_metrics['accuracy'],
                'std_precision': std_metrics['precision'],
                'std_recall': std_metrics['recall'],
                'std_f1_score': std_metrics['f1_score'],
                'std_accuracy': std_metrics['accuracy'],
                'execution_time': execution_time,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'detailed_results': successful_results,
            'parameters': self.best_params
        }
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False)
            
        print(f"\n💾 结果已保存到: {result_file}")
        
        return successful_results

def main():
    """主函数"""
    print("🌊 Charleston洪水预测 - 简化版10次交叉验证")
    
    validator = SimpleFloodCrossValidator()
    results = validator.run_all_experiments()
    
    if results:
        print(f"\n🎉 实验完成！成功完成 {len(results)} 次实验")
    else:
        print(f"\n💥 实验失败！")
        
    return validator, results

if __name__ == "__main__":
    validator, results = main()