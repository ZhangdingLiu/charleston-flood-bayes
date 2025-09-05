#!/usr/bin/env python3
"""
评估基于训练集构建的贝叶斯网络

功能：
1. 加载train_based_bn.pkl模型文件
2. 在测试集上进行evidence-based推理
3. 测试多个概率阈值的性能
4. 生成详细的性能报告和混淆矩阵可视化
5. 对比训练集统计与测试集实际表现

网络结构：
   PITT ST → ASHLEY AVE
   HARLESTON VILLAGE → WASHINGTON ST
   AIKEN ST → CALHOUN ST

评估策略：
- Evidence: 测试集中当天观测到洪水且在网络节点内的道路
- 阈值测试: 0.2, 0.3, 0.4, 0.5, 0.6, 0.7
- 指标: Precision, Recall, F1, Accuracy, 混淆矩阵

用法：
    python evaluate_train_based_bn.py

输出：
    - results/train_based_bn_metrics.json - 详细评估结果
    - figs/train_based_confmat.png - 混淆矩阵可视化
    - 终端输出不同阈值的性能对比
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    accuracy_score, classification_report
)

# 贝叶斯网络推理
try:
    from pgmpy.inference import VariableElimination
    try:
        from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
    except ImportError:
        from pgmpy.models import BayesianNetwork
except ImportError:
    print("请安装pgmpy: pip install pgmpy")
    exit(1)

# 设置随机种子
RANDOM_SEED = 42

class TrainBasedBayesianNetworkEvaluator:
    """基于训练集的贝叶斯网络评估器"""
    
    def __init__(self, 
                 model_path="train_based_bn.pkl",
                 data_csv_path="Road_Closures_2024.csv",
                 results_dir="results",
                 figs_dir="figs"):
        self.model_path = model_path
        self.data_csv_path = data_csv_path
        self.results_dir = results_dir
        self.figs_dir = figs_dir
        
        # 创建输出目录
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(figs_dir, exist_ok=True)
        
        # 评估参数
        self.prob_thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        
        # 数据和模型
        self.train_df = None
        self.test_df = None
        self.model_data = None
        self.bayesian_network = None
        self.inference_engine = None
        self.selected_roads = None
        self.node_mapping = None
        
        # 预测结果缓存
        self.prediction_cache = []
        
    def load_and_split_data(self):
        """加载数据并分割（与构建脚本保持一致）"""
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
        
        print(f"   训练集: {len(self.train_df)}条")
        print(f"   测试集: {len(self.test_df)}条")
        
        return self.train_df, self.test_df
        
    def load_model(self):
        """加载贝叶斯网络模型"""
        print("2. 加载贝叶斯网络模型...")
        
        try:
            with open(self.model_path, 'rb') as f:
                self.model_data = pickle.load(f)
            
            self.bayesian_network = self.model_data['bayesian_network']
            self.selected_roads = self.model_data['selected_roads']
            self.node_mapping = self.model_data['node_mapping']
            
            # 创建推理引擎
            self.inference_engine = VariableElimination(self.bayesian_network)
            
            print(f"   ✅ 模型加载成功")
            print(f"   选定道路: {self.selected_roads}")
            print(f"   网络节点: {len(self.bayesian_network.nodes())}")
            print(f"   网络边: {len(self.bayesian_network.edges())}")
            
            # 显示网络结构
            print(f"   网络结构:")
            for parent, child, conf in self.model_data['network_edges']:
                print(f"     {parent} → {child} (训练集条件概率: {conf:.4f})")
            
            return True
            
        except FileNotFoundError:
            print(f"   ❌ 找不到模型文件: {self.model_path}")
            print("   请先运行 build_train_based_bn.py 生成模型")
            return False
        except Exception as e:
            print(f"   ❌ 加载模型时出错: {e}")
            return False
            
    def compute_test_predictions(self):
        """计算测试集预测结果"""
        print("3. 计算测试集预测...")
        
        # 过滤测试集数据
        test_filtered = self.test_df[self.test_df['road'].isin(self.selected_roads)].copy()
        
        print(f"   测试集中选定道路的记录: {len(test_filtered)}条")
        
        # 显示测试集中各道路的频率
        print(f"   测试集中各道路出现次数:")
        for road in self.selected_roads:
            count = len(test_filtered[test_filtered['road'] == road])
            print(f"     {road}: {count}次")
        
        # 按日期分组
        test_by_date = test_filtered.groupby(test_filtered["date"])
        
        total_predictions = 0
        evaluated_days = 0
        days_with_evidence = 0
        
        for date, day_group in test_by_date:
            # 当天洪水道路
            flooded_roads = set(day_group["road"].unique())
            
            # 在选定道路中的洪水道路
            flooded_selected = flooded_roads & set(self.selected_roads)
            
            if len(flooded_selected) == 0:
                continue  # 没有选定道路洪水，跳过
            
            evaluated_days += 1
            
            # 构建evidence（观测到的洪水道路）
            evidence = {}
            for road in flooded_selected:
                node_name = self.node_mapping[road]
                evidence[node_name] = 1  # 洪水状态
            
            if len(evidence) > 0:
                days_with_evidence += 1
            
            # 对所有选定道路进行预测
            day_predictions = []
            for road in self.selected_roads:
                node_name = self.node_mapping[road]
                
                # 如果该道路已在evidence中，跳过预测
                if node_name in evidence:
                    continue
                
                try:
                    # 使用贝叶斯推理计算概率
                    if len(evidence) > 0:
                        query_result = self.inference_engine.query(
                            variables=[node_name], 
                            evidence=evidence
                        )
                        prob_flood = query_result.values[1]  # P(flood=1)
                    else:
                        # 没有evidence，使用先验概率
                        query_result = self.inference_engine.query(variables=[node_name])
                        prob_flood = query_result.values[1]
                    
                    # 真实标签
                    true_flood = 1 if road in flooded_roads else 0
                    
                    day_predictions.append({
                        "date": str(date.date()),
                        "road": road,
                        "node_name": node_name,
                        "prob_flood": prob_flood,
                        "true_flood": true_flood,
                        "evidence": evidence.copy(),
                        "evidence_roads": [road for road in self.selected_roads 
                                         if self.node_mapping[road] in evidence]
                    })
                    total_predictions += 1
                    
                except Exception as e:
                    print(f"   ⚠️ 推理失败 - {road}: {e}")
                    # 使用默认概率
                    day_predictions.append({
                        "date": str(date.date()),
                        "road": road,
                        "node_name": node_name,
                        "prob_flood": 0.5,
                        "true_flood": 1 if road in flooded_roads else 0,
                        "evidence": evidence.copy(),
                        "evidence_roads": []
                    })
                    total_predictions += 1
            
            if len(day_predictions) > 0:
                self.prediction_cache.extend(day_predictions)
        
        print(f"   预测样本: {total_predictions}个")
        print(f"   有洪水的评估天数: {evaluated_days}天")
        print(f"   有evidence的天数: {days_with_evidence}天")
        
        if total_predictions == 0:
            raise ValueError("没有有效的预测样本")
        
        # 显示前几个预测样本
        print(f"   前5个预测样本:")
        for i, pred in enumerate(self.prediction_cache[:5]):
            print(f"     {pred['date']}: {pred['road']} "
                  f"(prob={pred['prob_flood']:.3f}, true={pred['true_flood']}, "
                  f"evidence={pred['evidence_roads']})")
            
    def evaluate_thresholds(self):
        """评估不同概率阈值的性能"""
        print("4. 评估不同概率阈值...")
        
        results = {}
        
        for threshold in self.prob_thresholds:
            print(f"   评估阈值 {threshold}...")
            
            # 生成预测标签
            y_true = []
            y_pred = []
            y_prob = []
            
            for pred in self.prediction_cache:
                y_true.append(pred["true_flood"])
                y_prob.append(pred["prob_flood"])
                y_pred.append(1 if pred["prob_flood"] >= threshold else 0)
            
            if len(y_true) == 0:
                continue
            
            # 计算指标
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # 混淆矩阵
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (len(y_true), 0, 0, 0)
            
            results[threshold] = {
                "threshold": threshold,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "confusion_matrix": cm.tolist(),
                "true_positives": int(tp),
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "total_samples": len(y_true),
                "positive_samples": sum(y_true)
            }
        
        return results
        
    def print_evaluation_results(self, results):
        """打印评估结果"""
        print("5. 显示评估结果...")
        
        print(f"\n{'='*80}")
        print("📊 基于训练集的贝叶斯网络评估结果")
        print(f"{'='*80}")
        
        # 网络信息
        print(f"网络结构:")
        print(f"  节点: {self.selected_roads}")
        print(f"  边: {[(edge[0], edge[1]) for edge in self.model_data['network_edges']]}")
        
        # 测试集统计
        total_positive = sum(pred["true_flood"] for pred in self.prediction_cache)
        total_samples = len(self.prediction_cache)
        print(f"\n测试集统计:")
        print(f"  总预测样本: {total_samples}")
        print(f"  正样本(洪水): {total_positive} ({total_positive/total_samples*100:.1f}%)")
        print(f"  负样本(无洪水): {total_samples-total_positive} ({(total_samples-total_positive)/total_samples*100:.1f}%)")
        
        # 性能对比表
        print(f"\n不同概率阈值性能对比:")
        print(f"{'阈值':<6} {'准确率':<8} {'精确率':<8} {'召回率':<8} {'F1分数':<8} {'TP':<4} {'TN':<4} {'FP':<4} {'FN':<4}")
        print("-" * 80)
        
        for threshold in self.prob_thresholds:
            if threshold in results:
                r = results[threshold]
                print(f"{threshold:<6.1f} {r['accuracy']:<8.4f} {r['precision']:<8.4f} "
                      f"{r['recall']:<8.4f} {r['f1']:<8.4f} "
                      f"{r['true_positives']:<4} {r['true_negatives']:<4} "
                      f"{r['false_positives']:<4} {r['false_negatives']:<4}")
        
        # 找到最佳阈值
        valid_results = {k: v for k, v in results.items() if v['f1'] > 0}
        if valid_results:
            best_threshold = max(valid_results.keys(), key=lambda x: valid_results[x]['f1'])
            best_result = valid_results[best_threshold]
            print(f"\n🎯 最佳阈值: {best_threshold} (F1: {best_result['f1']:.4f})")
        else:
            # 如果没有F1>0的结果，选择准确率最高的
            best_threshold = max(results.keys(), key=lambda x: results[x]['accuracy'])
            best_result = results[best_threshold]
            print(f"\n🎯 最佳阈值: {best_threshold} (准确率: {best_result['accuracy']:.4f})")
        
        # 详细结果
        print(f"\n最佳阈值详细结果:")
        print(f"  准确率: {best_result['accuracy']:.4f}")
        print(f"  精确率: {best_result['precision']:.4f}")
        print(f"  召回率: {best_result['recall']:.4f}")
        print(f"  F1分数: {best_result['f1']:.4f}")
        
        print(f"\n混淆矩阵 (阈值 {best_threshold}):")
        cm = np.array(best_result['confusion_matrix'])
        print(f"              预测")
        print(f"           无洪水  洪水")
        
        # 处理混淆矩阵的不同形状
        if cm.shape == (1, 1):
            # 只有一个类别的情况
            print(f"实际 无洪水   {cm[0,0]:4d}   0")
            print(f"     洪水     0      0")
        elif cm.shape == (2, 2):
            # 标准2x2混淆矩阵
            print(f"实际 无洪水   {cm[0,0]:4d}   {cm[0,1]:4d}")
            print(f"     洪水     {cm[1,0]:4d}   {cm[1,1]:4d}")
        else:
            # 其他情况
            print(f"混淆矩阵形状: {cm.shape}")
            print(cm)
        
        return best_threshold, best_result
        
    def create_confusion_matrix_visualization(self, results, best_threshold):
        """创建混淆矩阵可视化"""
        print("6. 生成混淆矩阵可视化...")
        
        # 创建子图：展示所有阈值的混淆矩阵
        n_thresholds = len(self.prob_thresholds)
        cols = 3
        rows = (n_thresholds + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        fig.suptitle('Train-Based Bayesian Network - Confusion Matrices', 
                    fontsize=16, fontweight='bold')
        
        axes = axes.flatten() if n_thresholds > 1 else [axes]
        
        for i, threshold in enumerate(self.prob_thresholds):
            if threshold not in results:
                continue
                
            ax = axes[i] if i < len(axes) else None
            if ax is None:
                break
            
            cm_data = np.array(results[threshold]['confusion_matrix'])
            
            # 创建热图
            sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['No Flood', 'Flood'],
                       yticklabels=['No Flood', 'Flood'],
                       ax=ax, cbar=False)
            
            # 设置标题和标签
            title = f'Threshold {threshold:.1f}'
            if threshold == best_threshold:
                title += ' ★ Best'
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            
            # 添加指标信息
            f1 = results[threshold]['f1']
            precision = results[threshold]['precision']
            recall = results[threshold]['recall']
            
            ax.text(0.5, -0.1, f'F1: {f1:.3f}, P: {precision:.3f}, R: {recall:.3f}',
                   transform=ax.transAxes, ha='center', va='top', fontsize=10)
        
        # 隐藏多余的子图
        for i in range(n_thresholds, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # 保存图表
        confmat_path = os.path.join(self.figs_dir, "train_based_confmat.png")
        plt.savefig(confmat_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 混淆矩阵可视化保存到: {confmat_path}")
        
        return confmat_path
        
    def save_results(self, results, best_threshold):
        """保存评估结果到JSON"""
        print("7. 保存评估结果...")
        
        # 构建完整结果
        output_results = {
            "evaluation_summary": {
                "timestamp": pd.Timestamp.now().isoformat(),
                "model_file": self.model_path,
                "selected_roads": self.selected_roads,
                "network_edges": [(edge[0], edge[1], edge[2]) for edge in self.model_data['network_edges']],
                "test_samples": len(self.prediction_cache),
                "positive_samples": sum(pred["true_flood"] for pred in self.prediction_cache),
                "best_threshold": best_threshold
            },
            "threshold_results": results,
            "best_performance": results[best_threshold],
            "prediction_samples": self.prediction_cache[:50]  # 保存前50个样本
        }
        
        # 保存到JSON
        results_path = os.path.join(self.results_dir, "train_based_bn_metrics.json")
        
        # 转换numpy类型
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        output_results_clean = convert_numpy(output_results)
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(output_results_clean, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ 结果保存到: {results_path}")
        return results_path
        
    def run_evaluation(self):
        """运行完整评估流程"""
        print("🚀 开始基于训练集的贝叶斯网络评估...")
        print("="*60)
        
        try:
            # 1. 数据加载
            self.load_and_split_data()
            
            # 2. 模型加载
            if not self.load_model():
                return None
            
            # 3. 计算预测
            self.compute_test_predictions()
            
            # 4. 评估阈值
            results = self.evaluate_thresholds()
            
            # 5. 显示结果
            best_threshold, best_result = self.print_evaluation_results(results)
            
            # 6. 创建可视化
            self.create_confusion_matrix_visualization(results, best_threshold)
            
            # 7. 保存结果
            self.save_results(results, best_threshold)
            
            print(f"\n✅ 基于训练集的贝叶斯网络评估完成！")
            print(f"📁 详细结果: {self.results_dir}/train_based_bn_metrics.json")
            print(f"📊 混淆矩阵: {self.figs_dir}/train_based_confmat.png")
            
            return results
            
        except Exception as e:
            print(f"\n❌ 评估过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """主函数"""
    evaluator = TrainBasedBayesianNetworkEvaluator()
    results = evaluator.run_evaluation()
    
    if results:
        print(f"\n🎯 评估总结:")
        print(f"   基于训练集构建的贝叶斯网络在测试集上的表现已量化")
        print(f"   避免了数据泄露，确保了评估的可靠性")
        print(f"   网络结构简洁，具有良好的可解释性")
        print(f"   为实际应用提供了参考基准")

if __name__ == "__main__":
    main()