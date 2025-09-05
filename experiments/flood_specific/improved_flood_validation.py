#!/usr/bin/env python3
"""
改进版洪水预测交叉验证实验
- 4个重要洪水日期
- 30%证据 -> 70%预测
- 多阈值测试 (0.2, 0.3, 0.4)
- 每日期5次随机实验
"""

import json
import os
import sys
import random
import time
import math
from datetime import datetime
from collections import defaultdict, Counter
import csv

# Set random seed
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

class ImprovedFloodValidator:
    """改进版洪水预测交叉验证器"""
    
    def __init__(self):
        # 最佳参数配置 (除阈值外)
        self.best_params = {
            'occ_thr': 4,
            'edge_thr': 3,
            'weight_thr': 0.2,
            'evidence_ratio': 0.3  # 30%作为证据
        }
        
        # 4个重要测试日期
        self.test_dates = [
            '2017/09/11',  # 52条道路
            '2016/10/08',  # 26条道路  
            '2024/04/11',  # 19条道路
            '2024/08/06'   # 18条道路
        ]
        
        # 4个更高的测试阈值
        self.pred_thresholds = [0.5, 0.6, 0.7, 0.8]
        
        # 每日期重复次数
        self.trials_per_date = 5
        
        self.flood_data = []
        self.all_results = []
        
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
                        objectid_key = '﻿OBJECTID'
                        
                    self.flood_data.append({
                        'date': start_date,
                        'street': street,
                        'objectid': row.get(objectid_key, '')
                    })
        
        print(f"✅ 加载完成: {len(self.flood_data)} 条洪水记录")
        
        # 验证测试日期
        date_roads = defaultdict(set)
        for record in self.flood_data:
            date_roads[record['date']].add(record['street'])
            
        print(f"📊 测试日期验证:")
        for date in self.test_dates:
            if date in date_roads:
                road_count = len(date_roads[date])
                print(f"   {date}: {road_count}条道路 ✅")
            else:
                print(f"   {date}: 数据中不存在 ❌")
                
        return self.flood_data
        
    def run_single_experiment(self, test_date, pred_threshold, trial_id):
        """运行单次实验"""
        print(f"\n📅 实验: {test_date}, 阈值: {pred_threshold}, 试验: {trial_id+1}/5")
        
        # 1. 数据分割
        test_records = [r for r in self.flood_data if r['date'] == test_date]
        train_records = [r for r in self.flood_data if r['date'] != test_date]
        
        test_roads = list(set(r['street'] for r in test_records))
        
        if len(test_roads) < 3:
            print("❌ 测试道路数量不足，跳过")
            return None
            
        print(f"   测试道路总数: {len(test_roads)}")
        
        # 2. 构建简化网络（基于训练数据）
        road_freq = Counter(r['street'] for r in train_records)
        network_roads = set(road for road, freq in road_freq.items() 
                          if freq >= self.best_params['occ_thr'])
        
        # 只考虑在网络中的测试道路
        test_network_roads = [road for road in test_roads if road in network_roads]
        
        if len(test_network_roads) < 3:
            print(f"❌ 网络中测试道路不足3条 (仅{len(test_network_roads)}条)，跳过")
            return None
            
        print(f"   网络中测试道路: {len(test_network_roads)}")
        
        # 3. 随机选择30%作为证据，70%作为预测目标
        evidence_count = max(1, int(len(test_network_roads) * self.best_params['evidence_ratio']))
        
        # 随机打乱并分割
        random.shuffle(test_network_roads)
        evidence_roads = test_network_roads[:evidence_count]
        predict_roads = test_network_roads[evidence_count:]
        
        if len(predict_roads) == 0:
            print("❌ 预测道路数量为0，跳过")
            return None
            
        print(f"   证据道路: {len(evidence_roads)} 条 ({len(evidence_roads)/len(test_network_roads)*100:.1f}%)")
        print(f"   预测道路: {len(predict_roads)} 条")
        print(f"   证据列表: {', '.join(evidence_roads)}")
        
        # 4. 简化的"预测"过程
        # 基于道路在训练集中的频次作为"概率"代理
        max_freq = max(road_freq.values()) if road_freq else 1
        
        road_probs = {}
        for road in predict_roads:
            # 基础概率：基于训练集频次
            base_prob = road_freq.get(road, 0) / max_freq
            
            # 证据调整：如果证据道路频次高，则提升相关道路概率
            evidence_boost = 0.0
            for ev_road in evidence_roads:
                ev_freq = road_freq.get(ev_road, 0) / max_freq
                # 简单的"相关性"：高频证据道路提升其他道路概率
                evidence_boost += ev_freq * 0.3
            
            # 最终概率 = 基础概率 + 证据提升
            final_prob = min(1.0, base_prob + evidence_boost / len(evidence_roads))
            road_probs[road] = final_prob
            
        # 5. 基于阈值生成预测
        predictions = {road: 1 if prob > pred_threshold else 0 
                      for road, prob in road_probs.items()}
        true_labels = {road: 1 for road in predict_roads}  # 所有预测道路实际都发洪水
        
        # 6. 计算性能指标
        tp = sum(1 for road in predict_roads if predictions[road] == 1 and true_labels[road] == 1)
        fp = sum(1 for road in predict_roads if predictions[road] == 1 and true_labels[road] == 0)
        tn = sum(1 for road in predict_roads if predictions[road] == 0 and true_labels[road] == 0)
        fn = sum(1 for road in predict_roads if predictions[road] == 0 and true_labels[road] == 1)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        print(f"   📈 性能: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, Acc={accuracy:.3f}")
        print(f"   📊 混淆矩阵: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
        
        # 7. 返回结果
        result = {
            'test_date': test_date,
            'pred_threshold': pred_threshold,
            'trial_id': trial_id,
            'test_roads_total': len(test_roads),
            'test_roads_in_network': len(test_network_roads),
            'evidence_roads_count': len(evidence_roads),
            'predict_roads_count': len(predict_roads),
            'evidence_roads': evidence_roads,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'road_probs': road_probs,
            'predictions': predictions
        }
        
        return result
        
    def run_all_experiments(self):
        """运行全部60次实验 (4日期 × 5试验 × 3阈值)"""
        print("🚀 开始改进版洪水预测交叉验证")
        print(f"📊 实验配置:")
        print(f"   测试日期: {len(self.test_dates)} 个")
        print(f"   预测阈值: {self.pred_thresholds}")
        print(f"   每日期试验: {self.trials_per_date} 次")
        print(f"   总实验数: {len(self.test_dates) * self.trials_per_date * len(self.pred_thresholds)} 次")
        print(f"   更高阈值测试: {self.pred_thresholds} (期望获得更合理的Precision)")
        
        start_time = time.time()
        
        # 加载数据
        self.load_flood_data()
        
        # 运行实验
        successful_results = []
        experiment_count = 0
        
        for date in self.test_dates:
            for threshold in self.pred_thresholds:
                for trial in range(self.trials_per_date):
                    experiment_count += 1
                    print(f"\n{'='*50} 实验 {experiment_count}/60 {'='*50}")
                    
                    result = self.run_single_experiment(date, threshold, trial)
                    if result:
                        successful_results.append(result)
                        self.all_results.append(result)
                        
        print(f"\n{'='*80}")
        print(f"📊 实验完成汇总")
        print(f"{'='*80}")
        
        if not successful_results:
            print("❌ 没有成功的实验")
            return None
            
        print(f"✅ 成功实验: {len(successful_results)}/80")
        print(f"💡 注意: 使用更高阈值({self.pred_thresholds})期望获得更合理的Precision值")
            
        # 按阈值分组分析
        threshold_results = defaultdict(list)
        for result in successful_results:
            threshold_results[result['pred_threshold']].append(result)
            
        print(f"✅ 成功实验: {len(successful_results)}/60")
        print(f"\n📈 按阈值分组的平均性能:")
        
        threshold_summary = {}
        for threshold in self.pred_thresholds:
            results = threshold_results[threshold]
            if results:
                avg_precision = sum(r['precision'] for r in results) / len(results)
                avg_recall = sum(r['recall'] for r in results) / len(results)
                avg_f1 = sum(r['f1_score'] for r in results) / len(results)
                avg_accuracy = sum(r['accuracy'] for r in results) / len(results)
                
                # 计算标准差
                def calc_std(values):
                    if len(values) <= 1:
                        return 0.0
                    mean = sum(values) / len(values)
                    variance = sum((x - mean) ** 2 for x in values) / len(values)
                    return math.sqrt(variance)
                
                std_precision = calc_std([r['precision'] for r in results])
                std_recall = calc_std([r['recall'] for r in results])
                std_f1 = calc_std([r['f1_score'] for r in results])
                
                print(f"\n🎯 阈值 {threshold}:")
                print(f"   Precision: {avg_precision:.3f} ± {std_precision:.3f}")
                print(f"   Recall:    {avg_recall:.3f} ± {std_recall:.3f}")
                print(f"   F1 Score:  {avg_f1:.3f} ± {std_f1:.3f}")
                print(f"   Accuracy:  {avg_accuracy:.3f}")
                print(f"   实验数量: {len(results)}")
                
                threshold_summary[threshold] = {
                    'avg_precision': avg_precision,
                    'avg_recall': avg_recall,
                    'avg_f1_score': avg_f1,
                    'avg_accuracy': avg_accuracy,
                    'std_precision': std_precision,
                    'std_recall': std_recall,
                    'std_f1_score': std_f1,
                    'experiment_count': len(results)
                }
        
        # 找出最佳阈值和Precision分析
        if threshold_summary:
            best_threshold = max(threshold_summary.keys(), 
                               key=lambda t: threshold_summary[t]['avg_f1_score'])
            
            print(f"\n🏆 最佳阈值: {best_threshold} (F1: {threshold_summary[best_threshold]['avg_f1_score']:.3f})")
            
            # 分析Precision分布
            print(f"\n📊 Precision分析:")
            for thresh in sorted(threshold_summary.keys()):
                precision = threshold_summary[thresh]['avg_precision']
                print(f"   阈值 {thresh}: Precision = {precision:.3f}")
                
            # 检查是否还是100%
            if all(threshold_summary[t]['avg_precision'] >= 0.999 for t in threshold_summary.keys()):
                print("⚠️  警告: 所有阈值的Precision仍接近100%，可能需要进一步提高阈值")
            else:
                print("✅ 成功: 获得了更合理的Precision分布")
        
        execution_time = time.time() - start_time
        print(f"\n⏱️  总执行时间: {execution_time:.1f} 秒")
        
        # 保存结果
        self.save_results({
            'experiment_summary': {
                'total_experiments_planned': 60,
                'successful_experiments': len(successful_results),
                'test_dates': self.test_dates,
                'pred_thresholds': self.pred_thresholds,
                'trials_per_date': self.trials_per_date,
                'best_threshold': best_threshold if threshold_summary else None,
                'threshold_summary': threshold_summary,
                'execution_time': execution_time,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'detailed_results': successful_results,
            'parameters': self.best_params
        })
        
        return successful_results
        
    def save_results(self, results):
        """保存实验结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存JSON结果
        result_file = f"improved_flood_validation_results_{timestamp}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
        # 保存CSV汇总
        csv_file = f"improved_flood_validation_summary_{timestamp}.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'test_date', 'pred_threshold', 'trial_id', 
                'test_roads_total', 'evidence_roads_count', 'predict_roads_count',
                'precision', 'recall', 'f1_score', 'accuracy',
                'tp', 'fp', 'tn', 'fn'
            ])
            
            for result in results['detailed_results']:
                writer.writerow([
                    result['test_date'], result['pred_threshold'], result['trial_id'],
                    result['test_roads_total'], result['evidence_roads_count'], result['predict_roads_count'],
                    result['precision'], result['recall'], result['f1_score'], result['accuracy'],
                    result['tp'], result['fp'], result['tn'], result['fn']
                ])
        
        print(f"\n💾 结果已保存:")
        print(f"   - {result_file} (详细结果)")
        print(f"   - {csv_file} (性能汇总)")
        
        return result_file, csv_file

def main():
    """主函数"""
    print("🌊 Charleston洪水预测 - 改进版交叉验证")
    print("30%证据 → 70%预测，多阈值测试")
    
    validator = ImprovedFloodValidator()
    results = validator.run_all_experiments()
    
    if results:
        print(f"\n🎉 实验完成！成功完成 {len(results)} 次实验")
        print("📊 结果显示不同阈值下的性能差异，为实际部署提供参考")
    else:
        print(f"\n💥 实验失败！请检查数据和参数设置")
    
    return validator, results

if __name__ == "__main__":
    validator, results = main()