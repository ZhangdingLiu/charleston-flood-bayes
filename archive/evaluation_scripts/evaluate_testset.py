#!/usr/bin/env python3
"""
Charleston洪水贝叶斯网络测试集评估脚本

功能：
1. 加载与main_clean.py相同的数据分割（使用相同随机种子）
2. 重建优化后的贝叶斯网络
3. 在测试集上系统评估网络性能
4. 生成详细的评估指标和可视化

用法：
    python evaluate_testset.py

输出：
    - 终端输出评估指标表格
    - results/test_metrics.json - 详细评估结果
    - figs/confusion_matrix.png - 混淆矩阵热图
    - figs/metric_bar.png - 指标柱状图

注意：
    确保已运行main_clean.py生成网络文件
"""

import random
import numpy as np
import pandas as pd
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any

# 机器学习和评估
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    brier_score_loss, classification_report
)

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns

# 贝叶斯网络
from model import FloodBayesNetwork

# 设置随机种子（与main_clean.py保持一致）
RANDOM_SEED = 42
random.seed(0)
np.random.seed(0)

class TestSetEvaluator:
    """测试集评估器"""
    
    def __init__(self, 
                 network_csv_path="charleston_flood_network.csv",
                 data_csv_path="Road_Closures_2024.csv",
                 results_dir="results",
                 figs_dir="figs"):
        self.network_csv_path = network_csv_path
        self.data_csv_path = data_csv_path
        self.results_dir = results_dir
        self.figs_dir = figs_dir
        
        # 创建输出目录
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(figs_dir, exist_ok=True)
        
        # 初始化数据和模型
        self.train_df = None
        self.test_df = None
        self.flood_net = None
        self.network_nodes = set()
        
        # 评估结果
        self.y_true = []
        self.y_prob = []
        self.y_pred = []
        self.evaluation_details = []
        
    def load_and_split_data(self):
        """加载数据并按相同方式分割（与main_clean.py保持一致）"""
        print("1. 加载和分割数据...")
        
        # 加载数据
        df = pd.read_csv(self.data_csv_path)
        df = df[df["REASON"].str.upper() == "FLOOD"].copy()
        
        # 数据预处理（与main_clean.py完全一致）
        df["time_create"] = pd.to_datetime(df["START"], utc=True)
        df["link_id"] = df["STREET"].str.upper().str.replace(" ", "_")
        df["link_id"] = df["link_id"].astype(str)
        df["id"] = df["OBJECTID"].astype(str)
        
        # 使用相同的随机种子分割
        self.train_df, self.test_df = train_test_split(
            df, test_size=0.3, random_state=RANDOM_SEED
        )
        
        print(f"   总洪水记录: {len(df)}")
        print(f"   训练集: {len(self.train_df)}条")
        print(f"   测试集: {len(self.test_df)}条")
        
        return self.train_df, self.test_df
        
    def rebuild_bayesian_network(self):
        """重建贝叶斯网络（使用与main_clean.py相同的参数）"""
        print("2. 重建贝叶斯网络...")
        
        # 创建网络实例
        self.flood_net = FloodBayesNetwork(t_window="D")
        
        # 拟合边际概率
        self.flood_net.fit_marginal(self.train_df)
        
        # 使用优化参数构建网络
        self.flood_net.build_network_by_co_occurrence(
            self.train_df,
            occ_thr=10,    # 与main_clean.py一致
            edge_thr=3,
            weight_thr=0.4,
            report=False
        )
        
        # 拟合条件概率
        self.flood_net.fit_conditional(self.train_df, max_parents=2, alpha=1)
        
        # 构建最终贝叶斯网络
        try:
            self.flood_net.build_bayes_network()
            self.flood_net.check_bayesian_network()
            print("   ✅ 贝叶斯网络重建成功")
        except Exception as e:
            print(f"   ⚠️ 贝叶斯网络构建警告: {e}")
            print("   继续使用网络结构进行评估")
        
        # 获取网络节点
        self.network_nodes = set(self.flood_net.network.nodes())
        print(f"   网络节点数: {len(self.network_nodes)}")
        
        return self.flood_net
        
    def get_road_frequencies(self):
        """获取训练集中的道路频次"""
        return Counter(self.train_df['link_id'])
        
    def evaluate_on_test_set(self):
        """在测试集上进行评估"""
        print("3. 在测试集上评估...")
        
        road_frequencies = self.get_road_frequencies()
        
        # 按日期分组测试集
        test_by_date = self.test_df.groupby(self.test_df["time_create"].dt.floor("D"))
        
        total_days = len(test_by_date)
        evaluated_days = 0
        
        for date, day_group in test_by_date:
            # 当天洪水道路
            flooded_roads = set(day_group["link_id"].unique())
            
            # 过滤：只考虑在网络中的道路
            flooded_in_network = flooded_roads & self.network_nodes
            
            if len(flooded_in_network) == 0:
                continue  # 跳过没有网络节点的日期
                
            # 选择evidence：频次≥2且在网络中的道路
            potential_evidence = {road for road in flooded_in_network 
                                if road_frequencies.get(road, 0) >= 2}
            
            if len(potential_evidence) == 0:
                continue  # 跳过没有合适evidence的日期
                
            # 使用前3个作为evidence（或所有如果少于3个）
            evidence_roads = list(potential_evidence)[:3]
            evidence = {road: 1 for road in evidence_roads}
            
            # 目标道路：网络中的其他道路
            target_roads = self.network_nodes - set(evidence_roads)
            
            if len(target_roads) == 0:
                continue  # 跳过没有目标道路的日期
                
            # 对每个目标道路进行预测
            for road in target_roads:
                try:
                    # 获取洪水概率
                    result = self.flood_net.infer_w_evidence(road, evidence)
                    prob_flood = result.get("flooded", 0.5)  # 默认0.5如果推理失败
                    
                    # 真实标签
                    true_flood = 1 if road in flooded_roads else 0
                    
                    # 预测标签（阈值0.5）
                    pred_flood = 1 if prob_flood >= 0.5 else 0
                    
                    # 记录结果
                    self.y_true.append(true_flood)
                    self.y_prob.append(prob_flood)
                    self.y_pred.append(pred_flood)
                    
                    # 详细记录
                    self.evaluation_details.append({
                        "date": str(date.date()),
                        "target_road": road,
                        "evidence": evidence.copy(),
                        "true_flood": true_flood,
                        "prob_flood": prob_flood,
                        "pred_flood": pred_flood
                    })
                    
                except Exception as e:
                    print(f"   ⚠️ 推理失败 - 道路: {road}, 错误: {e}")
                    continue
                    
            evaluated_days += 1
            
        print(f"   评估样本数: {len(self.y_true)}")
        print(f"   评估天数: {evaluated_days}/{total_days}")
        
        if len(self.y_true) == 0:
            raise ValueError("没有有效的评估样本，请检查数据和网络设置")
            
    def calculate_metrics(self) -> Dict[str, float]:
        """计算评估指标"""
        print("4. 计算评估指标...")
        
        if len(self.y_true) == 0:
            raise ValueError("没有评估数据")
            
        # 基本分类指标
        accuracy = accuracy_score(self.y_true, self.y_pred)
        precision = precision_score(self.y_true, self.y_pred, zero_division=0)
        recall = recall_score(self.y_true, self.y_pred, zero_division=0)
        f1 = f1_score(self.y_true, self.y_pred, zero_division=0)
        
        # 概率指标
        try:
            roc_auc = roc_auc_score(self.y_true, self.y_prob) if len(set(self.y_true)) > 1 else 0.5
            pr_auc = average_precision_score(self.y_true, self.y_prob) if len(set(self.y_true)) > 1 else np.mean(self.y_true)
            brier = brier_score_loss(self.y_true, self.y_prob)
        except ValueError:
            roc_auc = pr_auc = brier = 0.0
            
        # 数据统计
        positive_ratio = np.mean(self.y_true)
        total_samples = len(self.y_true)
        
        # 混淆矩阵
        cm = confusion_matrix(self.y_true, self.y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "brier_score": brier,
            "positive_ratio": positive_ratio,
            "total_samples": total_samples,
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn)
        }
        
        print(f"   计算完成 - {total_samples}个样本")
        return metrics
        
    def print_metrics_table(self, metrics: Dict[str, float]):
        """打印整洁的指标表格"""
        print("\n" + "="*60)
        print("📊 测试集评估结果")
        print("="*60)
        
        # 主要指标
        print(f"{'指标':<15} {'值':<10} {'说明'}")
        print("-" * 50)
        print(f"{'Accuracy':<15} {metrics['accuracy']:<10.4f} 整体准确率")
        print(f"{'Precision':<15} {metrics['precision']:<10.4f} 精确率")
        print(f"{'Recall':<15} {metrics['recall']:<10.4f} 召回率")
        print(f"{'F1-Score':<15} {metrics['f1_score']:<10.4f} F1分数")
        print(f"{'ROC-AUC':<15} {metrics['roc_auc']:<10.4f} ROC曲线下面积")
        print(f"{'PR-AUC':<15} {metrics['pr_auc']:<10.4f} PR曲线下面积")
        print(f"{'Brier Score':<15} {metrics['brier_score']:<10.4f} 概率准确性")
        
        # 数据统计
        print(f"\n{'数据统计':<15} {'值':<10} {'说明'}")
        print("-" * 50)
        print(f"{'总样本':<15} {metrics['total_samples']:<10} 评估样本数")
        print(f"{'正样本比例':<15} {metrics['positive_ratio']:<10.4f} 洪水事件比例")
        print(f"{'True Pos':<15} {metrics['true_positives']:<10} 正确预测洪水")
        print(f"{'True Neg':<15} {metrics['true_negatives']:<10} 正确预测无洪水")
        print(f"{'False Pos':<15} {metrics['false_positives']:<10} 误报洪水")
        print(f"{'False Neg':<15} {metrics['false_negatives']:<10} 漏报洪水")
        
    def save_results(self, metrics: Dict[str, float]):
        """保存评估结果到JSON"""
        print("5. 保存评估结果...")
        
        # 完整结果
        results = {
            "evaluation_summary": {
                "timestamp": pd.Timestamp.now().isoformat(),
                "network_file": self.network_csv_path,
                "data_file": self.data_csv_path,
                "train_samples": len(self.train_df),
                "test_samples": len(self.test_df),
                "evaluation_samples": len(self.y_true)
            },
            "metrics": metrics,
            "network_info": {
                "nodes": len(self.network_nodes),
                "edges": self.flood_net.network.number_of_edges(),
                "network_nodes": sorted(list(self.network_nodes))
            },
            "evaluation_details": self.evaluation_details[:100]  # 只保存前100个详细记录
        }
        
        # 保存到JSON
        results_path = os.path.join(self.results_dir, "test_metrics.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
        print(f"   ✅ 结果保存到: {results_path}")
        
    def create_visualizations(self, metrics: Dict[str, float]):
        """创建可视化图表"""
        print("6. 生成可视化图表...")
        
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. 混淆矩阵热图
        self._create_confusion_matrix()
        
        # 2. 指标柱状图
        self._create_metrics_bar_chart(metrics)
        
        print("   ✅ 可视化图表生成完成")
        
    def _create_confusion_matrix(self):
        """创建混淆矩阵热图"""
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Flood', 'Flood'],
                   yticklabels=['No Flood', 'Flood'])
        plt.title('Confusion Matrix - Test Set Evaluation', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        
        # 添加统计信息
        total = cm.sum()
        accuracy = (cm[0,0] + cm[1,1]) / total if total > 0 else 0
        plt.figtext(0.02, 0.02, f'Total Samples: {total}, Accuracy: {accuracy:.3f}', 
                   fontsize=10, ha='left')
        
        plt.tight_layout()
        
        # 保存
        cm_path = os.path.join(self.figs_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 混淆矩阵保存到: {cm_path}")
        
    def _create_metrics_bar_chart(self, metrics: Dict[str, float]):
        """创建指标柱状图"""
        # 选择主要指标
        main_metrics = {
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1_score'],
            'ROC-AUC': metrics['roc_auc'],
            'PR-AUC': metrics['pr_auc']
        }
        
        plt.figure(figsize=(12, 8))
        
        # 创建柱状图
        bars = plt.bar(main_metrics.keys(), main_metrics.values(), 
                      color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'],
                      alpha=0.8, edgecolor='black', linewidth=1)
        
        # 添加数值标签
        for bar, value in zip(bars, main_metrics.values()):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.title('Test Set Evaluation Metrics', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Score', fontsize=12)
        plt.ylim(0, 1.1)
        plt.grid(axis='y', alpha=0.3)
        
        # 添加统计信息
        plt.figtext(0.02, 0.02, 
                   f'Total Samples: {metrics["total_samples"]}, '
                   f'Positive Ratio: {metrics["positive_ratio"]:.3f}',
                   fontsize=10, ha='left')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 保存
        bar_path = os.path.join(self.figs_dir, "metric_bar.png")
        plt.savefig(bar_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📈 指标柱状图保存到: {bar_path}")
        
    def run_evaluation(self):
        """运行完整评估流程"""
        print("🚀 开始测试集评估...")
        print("="*60)
        
        try:
            # 1. 数据准备
            self.load_and_split_data()
            
            # 2. 重建网络
            self.rebuild_bayesian_network()
            
            # 3. 测试集评估
            self.evaluate_on_test_set()
            
            # 4. 计算指标
            metrics = self.calculate_metrics()
            
            # 5. 输出结果
            self.print_metrics_table(metrics)
            
            # 6. 保存结果
            self.save_results(metrics)
            
            # 7. 生成可视化
            self.create_visualizations(metrics)
            
            print("\n" + "="*60)
            print("✅ 测试集评估完成！")
            print(f"📁 详细结果: {self.results_dir}/test_metrics.json")
            print(f"📊 可视化图表: {self.figs_dir}/")
            print("="*60)
            
            return metrics
            
        except Exception as e:
            print(f"\n❌ 评估过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """主函数"""
    # 创建评估器
    evaluator = TestSetEvaluator()
    
    # 运行评估
    metrics = evaluator.run_evaluation()
    
    if metrics:
        print("\n🎯 评估成功完成！")
        print(f"主要指标 - F1: {metrics['f1_score']:.3f}, "
              f"Accuracy: {metrics['accuracy']:.3f}, "
              f"ROC-AUC: {metrics['roc_auc']:.3f}")
    else:
        print("\n❌ 评估失败")

if __name__ == "__main__":
    main()