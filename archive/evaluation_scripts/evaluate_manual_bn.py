#!/usr/bin/env python3
"""
评估手动构建的4条道路极小贝叶斯网络

功能：
1. 加载与build_manual_bn.py相同的数据分割
2. 读取manual_bn.pkl模型文件
3. 在测试集上评估不同概率阈值的性能
4. 生成混淆矩阵可视化和详细评估报告

网络结构：
   BEE ST → SMITH ST
   E BAY ST → VANDERHORST ST

评估策略：
- Evidence: 测试集中当天观测到洪水且在网络节点内的道路
- 阈值测试: 0.3, 0.4, 0.5, 0.6
- 指标: Precision, Recall, F1, 混淆矩阵

用法：
    python evaluate_manual_bn.py

输出：
    - results/manual_bn_metrics.json - 详细评估结果
    - figs/manual_confmat.png - 混淆矩阵可视化
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

class ManualBayesianNetworkEvaluator:
    """手动贝叶斯网络评估器"""
    
    def __init__(self, 
                 model_path="manual_bn.pkl",
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
        self.prob_thresholds = [0.3, 0.4, 0.5, 0.6]
        
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
        """加载数据并分割（与build_manual_bn.py保持一致）"""
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
            
            return True
            
        except FileNotFoundError:
            print(f"   ❌ 找不到模型文件: {self.model_path}")
            print("   请先运行 build_manual_bn.py 生成模型")
            return False
        except Exception as e:
            print(f"   ❌ 加载模型时出错: {e}")
            return False
            
    def compute_test_predictions(self):
        """计算测试集预测结果"""
        print("3. 计算测试集预测...")
        
        # 过滤测试集数据
        test_filtered = self.test_df[self.test_df['road'].isin(self.selected_roads)].copy()
        
        # 按日期分组
        test_by_date = test_filtered.groupby(test_filtered["date"])
        
        total_predictions = 0
        evaluated_days = 0
        
        for date, day_group in test_by_date:
            # 当天洪水道路
            flooded_roads = set(day_group["road"].unique())
            
            # 在选定道路中的洪水道路
            flooded_selected = flooded_roads & set(self.selected_roads)
            
            if len(flooded_selected) == 0:
                continue  # 没有选定道路洪水，跳过
            
            # 构建evidence（观测到的洪水道路）
            evidence = {}
            for road in flooded_selected:
                node_name = self.node_mapping[road]
                evidence[node_name] = 1  # 洪水状态
            
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
                        "evidence": evidence.copy()
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
                        "evidence": evidence.copy()
                    })
                    total_predictions += 1
            
            if len(day_predictions) > 0:
                self.prediction_cache.extend(day_predictions)
                evaluated_days += 1
        
        print(f"   预测样本: {total_predictions}个")
        print(f"   评估天数: {evaluated_days}天")
        
        if total_predictions == 0:
            raise ValueError("没有有效的预测样本")
            
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
        print("📊 手动贝叶斯网络评估结果")
        print(f"{'='*80}")
        
        # 网络信息
        print(f"网络结构:")
        print(f"  节点: {self.selected_roads}")
        print(f"  边: {self.model_data['network_edges']}")
        
        # 性能对比表
        print(f"\n不同概率阈值性能对比:")
        print(f"{'阈值':<6} {'准确率':<8} {'精确率':<8} {'召回率':<8} {'F1分数':<8} {'样本数':<6}")
        print("-" * 60)
        
        for threshold in self.prob_thresholds:
            if threshold in results:
                r = results[threshold]
                print(f"{threshold:<6.1f} {r['accuracy']:<8.4f} {r['precision']:<8.4f} "
                      f"{r['recall']:<8.4f} {r['f1']:<8.4f} {r['total_samples']:<6}")
        
        # 找到最佳阈值
        best_threshold = max(results.keys(), key=lambda x: results[x]['f1'])
        best_result = results[best_threshold]
        
        print(f"\n🎯 最佳阈值: {best_threshold} (F1: {best_result['f1']:.4f})")
        
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
        print(f"实际 无洪水   {cm[0,0]:4d}   {cm[0,1]:4d}")
        print(f"     洪水     {cm[1,0]:4d}   {cm[1,1]:4d}")
        
        return best_threshold, best_result
        
    def create_confusion_matrix_visualization(self, results, best_threshold):
        """创建混淆矩阵可视化"""
        print("6. 生成混淆矩阵可视化...")
        
        best_result = results[best_threshold]
        cm = np.array(best_result['confusion_matrix'])
        
        # 创建图表
        plt.figure(figsize=(12, 10))
        
        # 创建子图：展示所有阈值的混淆矩阵
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Manual Bayesian Network - Confusion Matrices', 
                    fontsize=16, fontweight='bold')
        
        for i, threshold in enumerate(self.prob_thresholds):
            if threshold not in results:
                continue
                
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
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
        
        plt.tight_layout()
        
        # 保存图表
        confmat_path = os.path.join(self.figs_dir, "manual_confmat.png")
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
                "network_edges": self.model_data['network_edges'],
                "test_samples": len(self.prediction_cache),
                "best_threshold": best_threshold
            },
            "threshold_results": results,
            "best_performance": results[best_threshold],
            "prediction_samples": self.prediction_cache[:50]  # 保存前50个样本
        }
        
        # 保存到JSON
        results_path = os.path.join(self.results_dir, "manual_bn_metrics.json")
        
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
        print("🚀 开始手动贝叶斯网络评估...")
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
            
            print(f"\n✅ 手动贝叶斯网络评估完成！")
            print(f"📁 详细结果: {self.results_dir}/manual_bn_metrics.json")
            print(f"📊 混淆矩阵: {self.figs_dir}/manual_confmat.png")
            
            return results
            
        except Exception as e:
            print(f"\n❌ 评估过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """主函数"""
    evaluator = ManualBayesianNetworkEvaluator()
    results = evaluator.run_evaluation()
    
    if results:
        print(f"\n🎯 评估总结:")
        print(f"   极小贝叶斯网络在测试集上的表现已量化")
        print(f"   4条道路的简单网络结构便于解释和部署")
        print(f"   不同阈值提供了灵活的预警策略选择")

if __name__ == "__main__":
    main()