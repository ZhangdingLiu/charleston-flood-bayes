#!/usr/bin/env python3
"""
增强覆盖率贝叶斯网络洪水预测验证
- 使用全历史数据作为训练集
- 测试不同参数策略以最大化覆盖率
- 针对2017/09/11进行详细评估
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

class EnhancedCoverageValidator:
    """增强覆盖率贝叶斯洪水预测验证器"""
    
    def __init__(self):
        self.test_date = '2017/09/11'  # 固定测试日期
        
        # 三种参数策略
        self.parameter_strategies = {
            'conservative': {
                'name': '保守优化',
                'occ_thr': 3,
                'edge_thr': 2, 
                'weight_thr': 0.15,
                'evidence_ratio': 0.3
            },
            'balanced': {
                'name': '平衡优化',
                'occ_thr': 2,
                'edge_thr': 2,
                'weight_thr': 0.1,
                'evidence_ratio': 0.3
            },
            'aggressive': {
                'name': '激进优化', 
                'occ_thr': 1,
                'edge_thr': 1,
                'weight_thr': 0.05,
                'evidence_ratio': 0.3
            }
        }
        
        # 测试阈值
        self.pred_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        
        # 每个策略的重复试验次数
        self.trials_per_strategy = 5
        
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
        
        # 数据分割统计
        test_records = [r for r in self.flood_data if r['date'] == self.test_date]
        train_records = [r for r in self.flood_data if r['date'] != self.test_date]
        
        print(f"📊 数据分割:")
        print(f"   测试集 ({self.test_date}): {len(test_records)} 条记录")
        print(f"   训练集 (所有其他日期): {len(train_records)} 条记录")
        print(f"   训练/测试比例: {len(train_records)/len(test_records):.1f}:1")
        
        # 测试日期道路统计
        test_roads = set(r['street'] for r in test_records)
        print(f"   测试日期独特道路: {len(test_roads)} 条")
        
        return self.flood_data
    
    def build_bayesian_network(self, train_data, params):
        """构建贝叶斯网络 (基于训练数据和指定参数)"""
        strategy_name = params['name']
        print(f"🏗️ 构建贝叶斯网络 ({strategy_name})...")
        print(f"   参数: occ_thr={params['occ_thr']}, edge_thr={params['edge_thr']}, weight_thr={params['weight_thr']}")
        
        try:
            # 创建基本的训练数据列表
            simplified_data = []
            for record in train_data:
                simplified_data.append({
                    'date': record['date'],
                    'street': record['street'],
                    'objectid': record['objectid']
                })
            
            # 应用出现次数阈值过滤
            road_freq = Counter(r['street'] for r in simplified_data)
            network_roads = [road for road, freq in road_freq.items() 
                            if freq >= params['occ_thr']]
            
            if len(network_roads) < 3:
                print(f"❌ 网络道路不足3条 (仅{len(network_roads)}条)")
                return None, False
                
            print(f"✅ 网络构建完成: {len(network_roads)} 节点 (阈值{params['occ_thr']}过滤)")
            
            # 创建增强的网络对象
            class EnhancedNetwork:
                def __init__(self, roads, train_data, params):
                    self.nodes = set(roads)
                    self.train_data = train_data
                    self.road_freq = Counter(r['street'] for r in train_data)
                    self.params = params
                    
                    # 计算道路共现矩阵
                    self.cooccurrence = self._build_cooccurrence_matrix()
                    
                def _build_cooccurrence_matrix(self):
                    """构建道路共现矩阵"""
                    cooc = defaultdict(lambda: defaultdict(int))
                    
                    # 按日期分组
                    date_roads = defaultdict(set)
                    for record in self.train_data:
                        if record['street'] in self.nodes:
                            date_roads[record['date']].add(record['street'])
                    
                    # 计算共现次数
                    for date, roads in date_roads.items():
                        roads = list(roads)
                        for i, road1 in enumerate(roads):
                            for j, road2 in enumerate(roads):
                                if i != j:
                                    cooc[road1][road2] += 1
                    
                    return cooc
                    
                def number_of_nodes(self):
                    return len(self.nodes)
                    
                def number_of_edges(self):
                    # 计算满足阈值的边数
                    edge_count = 0
                    for road1 in self.nodes:
                        for road2 in self.nodes:
                            if road1 != road2:
                                cooc_count = self.cooccurrence[road1][road2]
                                if cooc_count >= self.params['edge_thr']:
                                    # 计算条件概率
                                    road1_freq = self.road_freq[road1]
                                    if road1_freq > 0:
                                        cond_prob = cooc_count / road1_freq
                                        if cond_prob >= self.params['weight_thr']:
                                            edge_count += 1
                    return edge_count
                    
                def infer_w_evidence(self, road, evidence):
                    """增强的贝叶斯推理"""
                    if road not in self.nodes:
                        return {'flooded': 0.0}
                    
                    # 基础概率：基于训练频次，归一化到最大频次
                    max_freq = max(self.road_freq.values()) if self.road_freq.values() else 1
                    base_prob = self.road_freq.get(road, 0) / max_freq
                    
                    # 证据影响：基于共现关系
                    evidence_boost = 0.0
                    evidence_count = 0
                    
                    for ev_road, ev_value in evidence.items():
                        if ev_value == 1 and ev_road in self.nodes and ev_road != road:
                            # 计算条件概率 P(target_road | evidence_road)
                            cooc_count = self.cooccurrence[ev_road][road]
                            ev_freq = self.road_freq.get(ev_road, 0)
                            
                            if ev_freq > 0 and cooc_count >= self.params['edge_thr']:
                                cond_prob = cooc_count / ev_freq
                                if cond_prob >= self.params['weight_thr']:
                                    evidence_boost += cond_prob * 0.5  # 增强权重
                                    evidence_count += 1
                    
                    # 综合概率计算
                    if evidence_count > 0:
                        # 有有效证据时，结合基础概率和证据
                        evidence_avg = evidence_boost / evidence_count
                        final_prob = min(1.0, base_prob * 0.3 + evidence_avg * 0.7)
                    else:
                        # 无有效证据时，使用基础概率
                        final_prob = base_prob * 0.5  # 降低无证据时的置信度
                    
                    return {'flooded': final_prob}
            
            enhanced_network = EnhancedNetwork(network_roads, simplified_data, params)
            
            # 创建包装对象以兼容原有接口
            class NetworkWrapper:
                def __init__(self, enhanced_net):
                    self.network = enhanced_net
                    
                def infer_w_evidence(self, road, evidence):
                    return self.network.infer_w_evidence(road, evidence)
            
            flood_net = NetworkWrapper(enhanced_network)
            
            print(f"   网络统计: {enhanced_network.number_of_nodes()} 节点, {enhanced_network.number_of_edges()} 条边")
            
            return flood_net, True
            
        except Exception as e:
            print(f"❌ 网络构建失败: {str(e)}")
            return None, False
    
    def run_single_experiment(self, strategy_key, params, pred_threshold, trial_id):
        """运行单次实验"""
        print(f"\n📅 实验: {params['name']}, 阈值: {pred_threshold}, 试验: {trial_id+1}/{self.trials_per_strategy}")
        
        # 1. 数据分割 - 使用全历史数据
        test_records = [r for r in self.flood_data if r['date'] == self.test_date]
        train_records = [r for r in self.flood_data if r['date'] != self.test_date]
        
        test_roads = set(r['street'] for r in test_records)
        
        if len(test_roads) < 3:
            print("❌ 测试道路数量不足，跳过")
            return None
            
        print(f"   测试日期道路总数: {len(test_roads)}")
        
        # 2. 构建贝叶斯网络
        flood_net, success = self.build_bayesian_network(train_records, params)
        if not success:
            return None
        
        # 获取网络中的所有道路
        network_roads = flood_net.network.nodes
        
        # 计算覆盖率
        test_network_roads = list(test_roads & network_roads)
        coverage_rate = len(test_network_roads) / len(test_roads)
        
        print(f"   🎯 覆盖率分析:")
        print(f"   网络总节点: {len(network_roads)}")
        print(f"   测试道路在网络中: {len(test_network_roads)}/{len(test_roads)} = {coverage_rate:.1%}")
        
        if len(test_network_roads) < 3:
            print(f"❌ 网络中测试道路不足3条，跳过")
            return None
        
        # 3. 选择证据和预测目标
        evidence_count = max(1, int(len(test_network_roads) * params['evidence_ratio']))
        
        # 随机选择证据道路
        random.shuffle(test_network_roads)
        evidence_roads = test_network_roads[:evidence_count]
        
        # 预测目标：所有网络中的非证据道路
        all_predict_roads = list(network_roads - set(evidence_roads))
        positive_predict_count = len([road for road in all_predict_roads if road in test_roads])
        negative_predict_count = len(all_predict_roads) - positive_predict_count
        
        print(f"   证据道路: {len(evidence_roads)} 条")
        print(f"   预测目标: {len(all_predict_roads)} 条 (正样本:{positive_predict_count}, 负样本:{negative_predict_count})")
        
        # 4. 贝叶斯推理预测
        evidence = {road: 1 for road in evidence_roads}
        predictions = {}
        true_labels = {}
        detailed_predictions = []
        
        successful_predictions = 0
        failed_predictions = 0
        
        for road in all_predict_roads:
            true_label = 1 if road in test_roads else 0
            
            try:
                result = flood_net.infer_w_evidence(road, evidence)
                prob = result.get('flooded', result.get(1, 0))
                
                predicted_label = 1 if prob >= pred_threshold else 0
                predictions[road] = predicted_label
                true_labels[road] = true_label
                
                detailed_predictions.append({
                    'road_name': road,
                    'predicted_probability': float(prob),
                    'true_label': true_label,
                    'predicted_label': predicted_label,
                    'inference_failed': False
                })
                
                successful_predictions += 1
                
            except Exception as e:
                print(f"   ⚠️ 推理失败 {road}: {str(e)}")
                
                detailed_predictions.append({
                    'road_name': road,
                    'predicted_probability': None,
                    'true_label': true_label,
                    'predicted_label': None,
                    'inference_failed': True,
                    'error_message': str(e)
                })
                
                failed_predictions += 1
                continue
        
        if successful_predictions == 0:
            print("❌ 没有成功的贝叶斯推理，跳过")
            return None
            
        print(f"   推理结果: {successful_predictions} 成功, {failed_predictions} 失败")
        
        # 5. 计算性能指标
        y_true = [true_labels[road] for road in predictions.keys()]
        y_pred = [predictions[road] for road in predictions.keys()]
        
        tp = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 1)
        fp = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 1)
        tn = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 0)
        fn = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 0)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        print(f"   📈 性能: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, Acc={accuracy:.3f}")
        print(f"   📊 混淆矩阵: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
        
        # 6. 返回结果
        result = {
            'strategy': strategy_key,
            'strategy_name': params['name'],
            'test_date': self.test_date,
            'pred_threshold': pred_threshold,
            'trial_id': trial_id,
            'parameters': params,
            'coverage_rate': coverage_rate,
            'test_roads_total': len(test_roads),
            'test_roads_in_network': len(test_network_roads),
            'network_nodes_total': len(network_roads),
            'evidence_roads_count': len(evidence_roads),
            'positive_predict_roads_count': positive_predict_count,
            'negative_predict_roads_count': negative_predict_count,
            'total_predict_roads_count': len(all_predict_roads),
            'successful_predictions': successful_predictions,
            'failed_predictions': failed_predictions,
            'evidence_roads': evidence_roads,
            'detailed_predictions': detailed_predictions,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'network_nodes': flood_net.network.number_of_nodes(),
            'network_edges': flood_net.network.number_of_edges()
        }
        
        return result
    
    def run_all_experiments(self):
        """运行所有参数策略的对比实验"""
        print("🚀 开始增强覆盖率贝叶斯网络实验")
        print("🎯 目标: 最大化2017/09/11测试集的预测覆盖率")
        print(f"📊 实验配置:")
        print(f"   参数策略: {len(self.parameter_strategies)} 种")
        print(f"   预测阈值: {self.pred_thresholds}")
        print(f"   每策略试验: {self.trials_per_strategy} 次")
        print(f"   总实验数: {len(self.parameter_strategies) * len(self.pred_thresholds) * self.trials_per_strategy} 次")
        
        start_time = time.time()
        
        # 加载数据
        self.load_flood_data()
        
        # 运行实验
        successful_results = []
        experiment_count = 0
        
        for strategy_key, params in self.parameter_strategies.items():
            print(f"\n{'='*80}")
            print(f"🔬 开始测试策略: {params['name']}")
            print(f"{'='*80}")
            
            for threshold in self.pred_thresholds:
                for trial in range(self.trials_per_strategy):
                    experiment_count += 1
                    total_experiments = len(self.parameter_strategies) * len(self.pred_thresholds) * self.trials_per_strategy
                    print(f"\n{'='*60} 实验 {experiment_count}/{total_experiments} {'='*60}")
                    
                    result = self.run_single_experiment(strategy_key, params, threshold, trial)
                    if result:
                        successful_results.append(result)
                        self.all_results.append(result)
        
        print(f"\n{'='*80}")
        print(f"📊 增强覆盖率实验完成汇总")
        print(f"{'='*80}")
        
        if not successful_results:
            print("❌ 没有成功的实验")
            return None
        
        total_planned = len(self.parameter_strategies) * len(self.pred_thresholds) * self.trials_per_strategy
        print(f"✅ 成功实验: {len(successful_results)}/{total_planned}")
        
        # 按策略分析覆盖率
        self.analyze_coverage_results(successful_results)
        
        execution_time = time.time() - start_time
        print(f"\n⏱️  总执行时间: {execution_time:.1f} 秒")
        
        # 保存结果
        self.save_results(successful_results, execution_time)
        
        return successful_results
    
    def analyze_coverage_results(self, results):
        """分析覆盖率结果"""
        print(f"\n📈 覆盖率和性能分析:")
        
        # 按策略分组
        strategy_results = defaultdict(list)
        for result in results:
            strategy_results[result['strategy']].append(result)
        
        print(f"\n{'策略':<15} {'覆盖率':<10} {'节点数':<8} {'边数':<8} {'Precision':<12} {'Recall':<10} {'F1':<10}")
        print("-" * 80)
        
        for strategy_key in ['conservative', 'balanced', 'aggressive']:
            if strategy_key in strategy_results:
                strategy_data = strategy_results[strategy_key]
                
                # 计算平均值
                avg_coverage = sum(r['coverage_rate'] for r in strategy_data) / len(strategy_data)
                avg_nodes = sum(r['network_nodes'] for r in strategy_data) / len(strategy_data)
                avg_edges = sum(r['network_edges'] for r in strategy_data) / len(strategy_data)
                avg_precision = sum(r['precision'] for r in strategy_data) / len(strategy_data)
                avg_recall = sum(r['recall'] for r in strategy_data) / len(strategy_data)
                avg_f1 = sum(r['f1_score'] for r in strategy_data) / len(strategy_data)
                
                strategy_name = strategy_data[0]['strategy_name']
                print(f"{strategy_name:<15} {avg_coverage:.1%}      {avg_nodes:.0f}      {avg_edges:.0f}      "
                      f"{avg_precision:.3f}        {avg_recall:.3f}    {avg_f1:.3f}")
        
        # 找出最高覆盖率的结果
        max_coverage_result = max(results, key=lambda x: x['coverage_rate'])
        print(f"\n🏆 最高覆盖率结果:")
        print(f"   策略: {max_coverage_result['strategy_name']}")
        print(f"   覆盖率: {max_coverage_result['coverage_rate']:.1%} ({max_coverage_result['test_roads_in_network']}/{max_coverage_result['test_roads_total']})")
        print(f"   网络规模: {max_coverage_result['network_nodes']} 节点, {max_coverage_result['network_edges']} 边")
        print(f"   性能: P={max_coverage_result['precision']:.3f}, R={max_coverage_result['recall']:.3f}, F1={max_coverage_result['f1_score']:.3f}")
    
    def save_results(self, results, execution_time):
        """保存实验结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存详细JSON结果
        result_file = f"enhanced_coverage_validation_results_{timestamp}.json"
        
        # 计算汇总统计
        summary_stats = {}
        strategy_results = defaultdict(list)
        for result in results:
            strategy_results[result['strategy']].append(result)
        
        for strategy_key, strategy_data in strategy_results.items():
            summary_stats[strategy_key] = {
                'strategy_name': strategy_data[0]['strategy_name'],
                'avg_coverage_rate': sum(r['coverage_rate'] for r in strategy_data) / len(strategy_data),
                'avg_network_nodes': sum(r['network_nodes'] for r in strategy_data) / len(strategy_data),
                'avg_network_edges': sum(r['network_edges'] for r in strategy_data) / len(strategy_data),
                'avg_precision': sum(r['precision'] for r in strategy_data) / len(strategy_data),
                'avg_recall': sum(r['recall'] for r in strategy_data) / len(strategy_data),
                'avg_f1_score': sum(r['f1_score'] for r in strategy_data) / len(strategy_data),
                'experiment_count': len(strategy_data)
            }
        
        result_data = {
            'experiment_summary': {
                'test_date': self.test_date,
                'total_experiments_planned': len(self.parameter_strategies) * len(self.pred_thresholds) * self.trials_per_strategy,
                'successful_experiments': len(results),
                'parameter_strategies': self.parameter_strategies,
                'pred_thresholds': self.pred_thresholds,
                'trials_per_strategy': self.trials_per_strategy,
                'execution_time': execution_time,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'method': 'enhanced_coverage_bayesian_inference',
                'summary_statistics': summary_stats
            },
            'detailed_results': results
        }
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        
        # 保存CSV汇总
        csv_file = f"enhanced_coverage_validation_summary_{timestamp}.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'strategy', 'strategy_name', 'test_date', 'pred_threshold', 'trial_id',
                'coverage_rate', 'test_roads_total', 'test_roads_in_network', 'network_nodes', 'network_edges',
                'positive_predict_count', 'negative_predict_count', 'total_predict_count',
                'successful_predictions', 'failed_predictions',
                'precision', 'recall', 'f1_score', 'accuracy',
                'tp', 'fp', 'tn', 'fn'
            ])
            
            for result in results:
                writer.writerow([
                    result['strategy'], result['strategy_name'], result['test_date'],
                    result['pred_threshold'], result['trial_id'],
                    result['coverage_rate'], result['test_roads_total'], result['test_roads_in_network'],
                    result['network_nodes'], result['network_edges'],
                    result['positive_predict_roads_count'], result['negative_predict_roads_count'],
                    result['total_predict_roads_count'],
                    result['successful_predictions'], result['failed_predictions'],
                    result['precision'], result['recall'], result['f1_score'], result['accuracy'],
                    result['tp'], result['fp'], result['tn'], result['fn']
                ])
        
        print(f"\n💾 结果已保存:")
        print(f"   - {result_file} (详细结果)")
        print(f"   - {csv_file} (性能汇总)")
        
        return result_file, csv_file

def main():
    """主函数"""
    print("🌊 Charleston洪水预测 - 增强覆盖率贝叶斯网络验证")
    print("🎯 使用全历史数据最大化预测覆盖率")
    
    validator = EnhancedCoverageValidator()
    results = validator.run_all_experiments()
    
    if results:
        print(f"\n🎉 增强覆盖率实验完成！成功完成 {len(results)} 次实验")
        print("📈 关键改进: 使用全历史数据 + 参数优化 + 覆盖率最大化")
    else:
        print(f"\n💥 实验失败！请检查数据和参数设置")
    
    return validator, results

if __name__ == "__main__":
    validator, results = main()