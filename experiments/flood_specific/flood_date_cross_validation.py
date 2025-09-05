#!/usr/bin/env python3
"""
10次洪水日期交叉验证实验
使用Top 10洪水日期进行Leave-One-Out交叉验证，全面评估贝叶斯模型性能
"""

import pandas as pd
import numpy as np
import warnings
import time
import json
import os
import sys
import random
from datetime import datetime
from collections import defaultdict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from pathlib import Path

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from model import FloodBayesNetwork
except ImportError:
    try:
        from core.model import FloodBayesNetwork
    except ImportError:
        print("❌ Cannot import FloodBayesNetwork, please ensure model.py or core/model.py exists")
        sys.exit(1)

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class FloodDateCrossValidator:
    """10次洪水日期交叉验证器"""
    
    def __init__(self):
        # 最佳参数配置 (基于参数优化结果)
        self.best_params = {
            'occ_thr': 4,
            'edge_thr': 3,
            'weight_thr': 0.2,
            'evidence_count': 1,
            'pred_threshold': 0.1,
            'negative_candidates': 9,
            'marginal_prob_threshold': 0.08
        }
        
        # Top 10洪水日期
        self.top10_dates = [
            '2017-09-11',  # 52条道路
            '2016-10-08',  # 26条道路
            '2019-12-23',  # 22条道路
            '2024-04-11',  # 19条道路
            '2024-08-06',  # 18条道路
            '2023-08-30',  # 16条道路
            '2022-09-30',  # 16条道路
            '2015-11-09',  # 16条道路
            '2022-11-10',  # 15条道路
            '2020-05-20'   # 15条道路
        ]
        
        self.all_results = []
        self.df = None
        
    def load_and_preprocess_data(self):
        """加载和预处理洪水数据"""
        print("🔄 加载洪水数据...")
        
        # 加载数据
        self.df = pd.read_csv("Road_Closures_2024.csv")
        self.df = self.df[self.df["REASON"].str.upper() == "FLOOD"].copy()
        
        # 预处理
        self.df["time_create"] = pd.to_datetime(self.df["START"], utc=True)
        self.df["link_id"] = self.df["STREET"].str.upper().str.replace(" ", "_")
        self.df["link_id"] = self.df["link_id"].astype(str)
        self.df["id"] = self.df["OBJECTID"].astype(str)
        self.df['flood_date'] = self.df['time_create'].dt.floor('D')
        self.df['date_str'] = self.df['flood_date'].dt.strftime('%Y-%m-%d')
        
        print(f"✅ 数据加载完成: {len(self.df)} 条洪水记录")
        print(f"📅 时间范围: {self.df['flood_date'].min().strftime('%Y-%m-%d')} 至 {self.df['flood_date'].max().strftime('%Y-%m-%d')}")
        
        # 验证Top 10日期在数据中的存在
        available_dates = set(self.df['date_str'].unique())
        missing_dates = []
        for date in self.top10_dates:
            if date not in available_dates:
                missing_dates.append(date)
        
        if missing_dates:
            print(f"⚠️ 以下日期在数据中不存在，将跳过: {missing_dates}")
            self.top10_dates = [d for d in self.top10_dates if d not in missing_dates]
            print(f"✅ 实际测试日期数: {len(self.top10_dates)}")
            
        return self.df
        
    def run_single_experiment(self, test_date_idx):
        """运行单次实验"""
        test_date = self.top10_dates[test_date_idx]
        exp_num = test_date_idx + 1
        
        print(f"\n{'='*80}")
        print(f"📅 实验 {exp_num}/{len(self.top10_dates)}: {test_date}")
        print(f"{'='*80}")
        
        # 1. 数据分割
        test_df = self.df[self.df['date_str'] == test_date].copy()
        train_df = self.df[self.df['date_str'] != test_date].copy()
        
        test_roads = set(test_df['link_id'].unique())
        
        print(f"📊 数据分割:")
        print(f"   训练数据: {len(train_df)} 条记录")
        print(f"   测试数据: {len(test_df)} 条记录 ({len(test_roads)} 条道路)")
        print(f"   测试道路: {', '.join(list(test_roads)[:8])}{'...' if len(test_roads) > 8 else ''}")
        
        if len(test_roads) < 2:
            print("❌ 测试道路数量不足2条，跳过此实验")
            return None
            
        # 2. 构建贝叶斯网络
        print(f"\n🏗️ 构建贝叶斯网络...")
        flood_net = FloodBayesNetwork(t_window="D")
        
        try:
            # 构建网络
            flood_net.build_network_by_co_occurrence(
                train_df,
                occ_thr=self.best_params['occ_thr'],
                edge_thr=self.best_params['edge_thr'],
                weight_thr=self.best_params['weight_thr'],
                report=False
            )
            
            # 拟合条件概率
            flood_net.fit_conditional(train_df, max_parents=2, alpha=1.0)
            flood_net.build_bayes_network()
            
            network_roads = set(flood_net.network.nodes())
            test_network_roads = test_roads & network_roads
            
            print(f"✅ 网络构建完成:")
            print(f"   网络节点: {len(network_roads)} 条道路")
            print(f"   网络边数: {flood_net.network.number_of_edges()} 条")
            print(f"   测试集在网络中: {len(test_network_roads)} 条道路")
            
            if len(test_network_roads) < 2:
                print("❌ 测试集在网络中的道路数量不足2条，跳过此实验")
                return None
                
        except Exception as e:
            print(f"❌ 网络构建失败: {e}")
            return None
            
        # 3. 预测和评估
        print(f"\n🔮 预测和评估...")
        
        # 随机选择1条道路作为证据
        test_network_roads_list = list(test_network_roads)
        evidence_road = random.choice(test_network_roads_list)
        predict_roads = [r for r in test_network_roads_list if r != evidence_road]
        
        print(f"🎯 证据输入: [{evidence_road}] (1条道路)")
        print(f"🔍 预测目标: {len(predict_roads)} 条道路")
        print(f"   道路列表: {', '.join(predict_roads[:6])}{'...' if len(predict_roads) > 6 else ''}")
        
        # 执行预测
        try:
            evidence = {evidence_road: 1}  # 证据道路发生洪水
            
            # 获取预测概率
            pred_probs = {}
            for road in predict_roads:
                try:
                    prob = flood_net.query_probability(road, evidence)
                    pred_probs[road] = prob
                except:
                    pred_probs[road] = 0.0
            
            # 生成预测标签
            pred_threshold = self.best_params['pred_threshold']
            predictions = {road: 1 if prob > pred_threshold else 0 for road, prob in pred_probs.items()}
            true_labels = {road: 1 for road in predict_roads}  # 测试日期所有道路都发洪水
            
            # 计算性能指标
            y_true = [true_labels[road] for road in predict_roads]
            y_pred = [predictions[road] for road in predict_roads]
            
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            accuracy = accuracy_score(y_true, y_pred)
            
            # 混淆矩阵
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            print(f"\n📈 性能指标:")
            print(f"   Precision: {precision:.3f}")
            print(f"   Recall:    {recall:.3f}")
            print(f"   F1 Score:  {f1:.3f}")
            print(f"   Accuracy:  {accuracy:.3f}")
            print(f"\n📊 混淆矩阵:")
            print(f"   TP: {tp}, FP: {fp}")
            print(f"   TN: {tn}, FN: {fn}")
            
            # 预测详情
            correct_predictions = sum(1 for road in predict_roads if predictions[road] == 1)
            print(f"\n🎯 预测详情:")
            print(f"   正确预测: {correct_predictions}/{len(predict_roads)} 条道路")
            print(f"   预测概率范围: {min(pred_probs.values()):.3f} - {max(pred_probs.values()):.3f}")
            
            # 保存结果
            experiment_result = {
                'experiment_id': exp_num,
                'test_date': test_date,
                'train_records': len(train_df),
                'test_roads_total': len(test_roads),
                'test_roads_in_network': len(test_network_roads),
                'network_nodes': len(network_roads),
                'network_edges': flood_net.network.number_of_edges(),
                'evidence_road': evidence_road,
                'predict_roads_count': len(predict_roads),
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'accuracy': accuracy,
                'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
                'pred_probs': pred_probs,
                'predictions': predictions,
                'correct_predictions': correct_predictions
            }
            
            return experiment_result
            
        except Exception as e:
            print(f"❌ 预测评估失败: {e}")
            return None
            
    def run_all_experiments(self):
        """运行全部10次实验"""
        print(f"\n🚀 开始10次洪水日期交叉验证实验")
        print(f"📊 使用最佳参数配置: {self.best_params}")
        print(f"📅 测试日期: {len(self.top10_dates)} 个")
        
        start_time = time.time()
        
        # 加载数据
        self.load_and_preprocess_data()
        
        # 运行每次实验
        successful_experiments = []
        
        for i in range(len(self.top10_dates)):
            result = self.run_single_experiment(i)
            if result is not None:
                successful_experiments.append(result)
                self.all_results.append(result)
        
        print(f"\n{'='*80}")
        print(f"📊 实验完成汇总")
        print(f"{'='*80}")
        
        if len(successful_experiments) == 0:
            print("❌ 没有成功的实验")
            return None
            
        # 计算平均性能指标
        avg_precision = np.mean([r['precision'] for r in successful_experiments])
        avg_recall = np.mean([r['recall'] for r in successful_experiments])
        avg_f1 = np.mean([r['f1_score'] for r in successful_experiments])
        avg_accuracy = np.mean([r['accuracy'] for r in successful_experiments])
        
        std_precision = np.std([r['precision'] for r in successful_experiments])
        std_recall = np.std([r['recall'] for r in successful_experiments])
        std_f1 = np.std([r['f1_score'] for r in successful_experiments])
        std_accuracy = np.std([r['accuracy'] for r in successful_experiments])
        
        print(f"✅ 成功实验: {len(successful_experiments)}/{len(self.top10_dates)}")
        print(f"\n📈 平均性能指标 (Mean ± Std):")
        print(f"   Precision: {avg_precision:.3f} ± {std_precision:.3f}")
        print(f"   Recall:    {avg_recall:.3f} ± {std_recall:.3f}")
        print(f"   F1 Score:  {avg_f1:.3f} ± {std_f1:.3f}")
        print(f"   Accuracy:  {avg_accuracy:.3f} ± {std_accuracy:.3f}")
        
        # 找出最佳和最差实验
        best_exp = max(successful_experiments, key=lambda x: x['f1_score'])
        worst_exp = min(successful_experiments, key=lambda x: x['f1_score'])
        
        print(f"\n🏆 最佳实验: {best_exp['test_date']} (F1: {best_exp['f1_score']:.3f})")
        print(f"😞 最差实验: {worst_exp['test_date']} (F1: {worst_exp['f1_score']:.3f})")
        
        execution_time = time.time() - start_time
        print(f"\n⏱️  总执行时间: {execution_time:.1f} 秒")
        
        # 保存结果
        self.save_results({
            'experiment_summary': {
                'total_experiments': len(self.top10_dates),
                'successful_experiments': len(successful_experiments),
                'avg_precision': float(avg_precision),
                'avg_recall': float(avg_recall),
                'avg_f1_score': float(avg_f1),
                'avg_accuracy': float(avg_accuracy),
                'std_precision': float(std_precision),
                'std_recall': float(std_recall),
                'std_f1_score': float(std_f1),
                'std_accuracy': float(std_accuracy),
                'best_experiment': best_exp['test_date'],
                'worst_experiment': worst_exp['test_date'],
                'execution_time': execution_time,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'detailed_results': successful_experiments,
            'parameters': self.best_params
        })
        
        return successful_experiments
        
    def save_results(self, results):
        """保存实验结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 创建结果目录
        result_dir = Path(f"results/flood_date_cross_validation_{timestamp}")
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存JSON结果
        with open(result_dir / "experiment_results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
        # 保存CSV汇总
        summary_data = []
        for result in results['detailed_results']:
            summary_data.append({
                'experiment_id': result['experiment_id'],
                'test_date': result['test_date'],
                'test_roads_total': result['test_roads_total'],
                'test_roads_in_network': result['test_roads_in_network'],
                'network_nodes': result['network_nodes'],
                'evidence_road': result['evidence_road'],
                'predict_roads_count': result['predict_roads_count'],
                'precision': result['precision'],
                'recall': result['recall'],
                'f1_score': result['f1_score'],
                'accuracy': result['accuracy'],
                'correct_predictions': result['correct_predictions']
            })
            
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(result_dir / "performance_summary.csv", index=False)
        
        print(f"\n💾 结果已保存到: {result_dir}")
        print(f"   - experiment_results.json (详细结果)")
        print(f"   - performance_summary.csv (性能汇总)")
        
        return result_dir

def main():
    """主函数"""
    print("🌊 Charleston洪水预测 - 10次洪水日期交叉验证")
    print("基于Top 10历史洪水事件的贝叶斯网络性能评估")
    
    validator = FloodDateCrossValidator()
    results = validator.run_all_experiments()
    
    if results:
        print(f"\n🎉 实验完成！成功完成 {len(results)} 次实验")
    else:
        print(f"\n💥 实验失败！请检查数据和参数设置")
    
    return validator, results

if __name__ == "__main__":
    validator, results = main()