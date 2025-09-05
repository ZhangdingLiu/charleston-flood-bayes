#!/usr/bin/env python3
"""
Charleston洪水贝叶斯网络阈值和Top-K优化脚本

功能：
1. 使用固定的优化网络
2. 网格搜索最佳概率阈值和Top-K参数
3. 评估不同参数组合的性能
4. 生成热图和最优参数推荐

用法：
    python evaluate_threshold_k.py

输出：
    - 终端输出最佳参数组合
    - results/grid_metrics.csv - 网格搜索结果
    - figs/pr_heatmap.png - F1分数热图

注意：
    确保已运行main_clean.py生成网络文件
"""

import random
import numpy as np
import pandas as pd
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

class ThresholdKOptimizer:
    """阈值和Top-K参数优化器"""
    
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
        
        # 网格搜索参数
        self.prob_thr_list = [0.2, 0.3, 0.4, 0.5]
        self.topk_list = [1, 3, 5]
        
        # 初始化数据和模型
        self.train_df = None
        self.test_df = None
        self.flood_net = None
        self.network_nodes = set()
        
        # 预计算的预测结果（避免重复计算）
        self.prediction_cache = []
        
    def load_and_split_data(self):
        """加载数据并按相同方式分割"""
        print("1. 加载和分割数据...")
        
        # 加载数据（与evaluate_testset.py完全一致）
        df = pd.read_csv(self.data_csv_path)
        df = df[df["REASON"].str.upper() == "FLOOD"].copy()
        
        # 数据预处理
        df["time_create"] = pd.to_datetime(df["START"], utc=True)
        df["link_id"] = df["STREET"].str.upper().str.replace(" ", "_")
        df["link_id"] = df["link_id"].astype(str)
        df["id"] = df["OBJECTID"].astype(str)
        
        # 使用相同的随机种子分割
        self.train_df, self.test_df = train_test_split(
            df, test_size=0.3, random_state=RANDOM_SEED
        )
        
        print(f"   训练集: {len(self.train_df)}条")
        print(f"   测试集: {len(self.test_df)}条")
        
        return self.train_df, self.test_df
        
    def rebuild_bayesian_network(self):
        """重建贝叶斯网络（固定网络配置）"""
        print("2. 重建固定贝叶斯网络...")
        
        # 创建网络实例
        self.flood_net = FloodBayesNetwork(t_window="D")
        
        # 拟合边际概率
        self.flood_net.fit_marginal(self.train_df)
        
        # 使用固定的优化参数
        self.flood_net.build_network_by_co_occurrence(
            self.train_df,
            occ_thr=10,    # 固定参数
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
            print("   ✅ 固定网络重建成功")
        except Exception as e:
            print(f"   ⚠️ 网络构建警告: {e}")
        
        # 获取网络节点
        self.network_nodes = set(self.flood_net.network.nodes())
        print(f"   网络节点数: {len(self.network_nodes)}")
        
        return self.flood_net
        
    def compute_predictions_cache(self):
        """预计算所有预测结果，避免重复计算"""
        print("3. 预计算预测结果...")
        
        road_frequencies = Counter(self.train_df['link_id'])
        test_by_date = self.test_df.groupby(self.test_df["time_create"].dt.floor("D"))
        
        total_predictions = 0
        
        for date, day_group in test_by_date:
            # 当天洪水道路
            flooded_roads = set(day_group["link_id"].unique())
            flooded_in_network = flooded_roads & self.network_nodes
            
            if len(flooded_in_network) == 0:
                continue
                
            # 选择evidence
            potential_evidence = {road for road in flooded_in_network 
                                if road_frequencies.get(road, 0) >= 2}
            
            if len(potential_evidence) == 0:
                continue
                
            evidence_roads = list(potential_evidence)[:3]
            evidence = {road: 1 for road in evidence_roads}
            target_roads = self.network_nodes - set(evidence_roads)
            
            if len(target_roads) == 0:
                continue
                
            # 对每个目标道路计算预测概率
            day_predictions = []
            for road in target_roads:
                try:
                    result = self.flood_net.infer_w_evidence(road, evidence)
                    prob_flood = result.get("flooded", 0.5)
                    true_flood = 1 if road in flooded_roads else 0
                    
                    day_predictions.append({
                        "date": str(date.date()),
                        "road": road,
                        "prob_flood": prob_flood,
                        "true_flood": true_flood,
                        "evidence": evidence.copy()
                    })
                    total_predictions += 1
                    
                except Exception as e:
                    continue
            
            # 按概率排序（用于Top-K）
            day_predictions.sort(key=lambda x: x["prob_flood"], reverse=True)
            self.prediction_cache.append(day_predictions)
        
        print(f"   预计算完成: {total_predictions}个预测, {len(self.prediction_cache)}天")
        
    def evaluate_threshold_strategy(self, prob_thr: float) -> Dict[str, float]:
        """评估概率阈值策略"""
        y_true, y_pred = [], []
        
        for day_predictions in self.prediction_cache:
            for pred in day_predictions:
                y_true.append(pred["true_flood"])
                y_pred.append(1 if pred["prob_flood"] >= prob_thr else 0)
        
        if len(y_true) == 0:
            return {"precision": 0, "recall": 0, "f1": 0, "hits_at_k": 0, "total_samples": 0}
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "hits_at_k": 0,  # 不适用于阈值策略
            "total_samples": len(y_true)
        }
        
    def evaluate_topk_strategy(self, k: int) -> Dict[str, float]:
        """评估Top-K策略"""
        total_hits = 0
        total_possible = 0
        all_true = []
        all_pred = []
        
        for day_predictions in self.prediction_cache:
            if len(day_predictions) == 0:
                continue
                
            # 计算当天实际洪水数
            actual_floods = sum(1 for pred in day_predictions if pred["true_flood"] == 1)
            total_possible += min(k, actual_floods) if actual_floods > 0 else 0
            
            # Top-K预测
            topk_roads = set()
            for i, pred in enumerate(day_predictions):
                if i < k:
                    topk_roads.add(pred["road"])
                    all_pred.append(1)  # Top-K中的道路标记为预测洪水
                else:
                    all_pred.append(0)  # 其他道路标记为预测无洪水
                all_true.append(pred["true_flood"])
            
            # 计算命中数
            day_hits = sum(1 for pred in day_predictions[:k] if pred["true_flood"] == 1)
            total_hits += day_hits
        
        # 计算Hits@K
        hits_at_k = total_hits / total_possible if total_possible > 0 else 0
        
        # 计算传统指标
        if len(all_true) > 0:
            precision = precision_score(all_true, all_pred, zero_division=0)
            recall = recall_score(all_true, all_pred, zero_division=0)
            f1 = f1_score(all_true, all_pred, zero_division=0)
        else:
            precision = recall = f1 = 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "hits_at_k": hits_at_k,
            "total_samples": len(all_true)
        }
        
    def run_grid_search(self) -> pd.DataFrame:
        """运行网格搜索"""
        print("4. 运行网格搜索...")
        
        results = []
        total_combinations = len(self.prob_thr_list) + len(self.topk_list)
        current = 0
        
        # 评估概率阈值策略
        for prob_thr in self.prob_thr_list:
            current += 1
            print(f"   进度 {current}/{total_combinations}: 概率阈值 {prob_thr}")
            
            metrics = self.evaluate_threshold_strategy(prob_thr)
            results.append({
                "strategy": "threshold",
                "prob_thr": prob_thr,
                "topk": 0,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "hits_at_k": metrics["hits_at_k"],
                "total_samples": metrics["total_samples"]
            })
        
        # 评估Top-K策略
        for k in self.topk_list:
            current += 1
            print(f"   进度 {current}/{total_combinations}: Top-{k}")
            
            metrics = self.evaluate_topk_strategy(k)
            results.append({
                "strategy": "topk",
                "prob_thr": 0.0,  # 不适用
                "topk": k,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "hits_at_k": metrics["hits_at_k"],
                "total_samples": metrics["total_samples"]
            })
        
        df = pd.DataFrame(results)
        print("   ✅ 网格搜索完成")
        return df
        
    def save_results(self, results_df: pd.DataFrame):
        """保存结果到CSV"""
        print("5. 保存网格搜索结果...")
        
        csv_path = os.path.join(self.results_dir, "grid_metrics.csv")
        results_df.to_csv(csv_path, index=False, float_format='%.4f')
        
        print(f"   ✅ 结果保存到: {csv_path}")
        
    def print_top_results(self, results_df: pd.DataFrame):
        """打印按F1排序的前5个结果"""
        print("6. 最佳参数组合分析...")
        
        # 按F1分数排序
        sorted_df = results_df.sort_values('f1', ascending=False)
        
        print(f"\n{'='*80}")
        print("🏆 按F1分数排序的最佳参数组合 (前5)")
        print(f"{'='*80}")
        
        print(f"{'排名':<4} {'策略':<10} {'阈值':<6} {'Top-K':<6} {'Precision':<10} {'Recall':<8} {'F1':<8} {'Hits@K':<8}")
        print("-" * 80)
        
        for i, (_, row) in enumerate(sorted_df.head(5).iterrows()):
            rank_symbol = "★" if i == 0 else f"{i+1:2d}"
            strategy_display = "阈值" if row['strategy'] == 'threshold' else f"Top-{int(row['topk'])}"
            thr_display = f"{row['prob_thr']:.1f}" if row['strategy'] == 'threshold' else "-"
            k_display = f"{int(row['topk'])}" if row['strategy'] == 'topk' else "-"
            
            print(f"{rank_symbol:<4} {strategy_display:<10} {thr_display:<6} {k_display:<6} "
                  f"{row['precision']:<10.4f} {row['recall']:<8.4f} {row['f1']:<8.4f} {row['hits_at_k']:<8.4f}")
        
        # 标注最佳配置
        best_row = sorted_df.iloc[0]
        print(f"\n🎯 推荐配置 (最高F1分数):")
        if best_row['strategy'] == 'threshold':
            print(f"   策略: 概率阈值 = {best_row['prob_thr']:.1f}")
        else:
            print(f"   策略: Top-{int(best_row['topk'])} 预警")
        print(f"   性能: F1={best_row['f1']:.4f}, Precision={best_row['precision']:.4f}, Recall={best_row['recall']:.4f}")
        
        if best_row['strategy'] == 'topk':
            print(f"   Hits@{int(best_row['topk'])}: {best_row['hits_at_k']:.4f}")
        
    def create_heatmap(self, results_df: pd.DataFrame):
        """创建F1分数热图"""
        print("7. 生成热图可视化...")
        
        # 准备热图数据
        # 由于我们有两种不同的策略，需要特殊处理
        plt.figure(figsize=(12, 8))
        
        # 分别处理阈值和Top-K策略
        threshold_data = results_df[results_df['strategy'] == 'threshold']
        topk_data = results_df[results_df['strategy'] == 'topk']
        
        # 创建子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 阈值策略柱状图
        if not threshold_data.empty:
            bars1 = ax1.bar(threshold_data['prob_thr'], threshold_data['f1'], 
                           color='skyblue', alpha=0.8, edgecolor='black')
            ax1.set_xlabel('Probability Threshold')
            ax1.set_ylabel('F1 Score')
            ax1.set_title('F1 Score vs Probability Threshold')
            ax1.grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, f1 in zip(bars1, threshold_data['f1']):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{f1:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Top-K策略柱状图
        if not topk_data.empty:
            bars2 = ax2.bar(topk_data['topk'], topk_data['f1'], 
                           color='lightcoral', alpha=0.8, edgecolor='black')
            ax2.set_xlabel('Top-K')
            ax2.set_ylabel('F1 Score')
            ax2.set_title('F1 Score vs Top-K')
            ax2.grid(True, alpha=0.3)
            ax2.set_xticks(topk_data['topk'])
            
            # 添加数值标签
            for bar, f1 in zip(bars2, topk_data['f1']):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{f1:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # 保存
        heatmap_path = os.path.join(self.figs_dir, "pr_heatmap.png")
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 热图保存到: {heatmap_path}")
        
        # 额外创建组合比较图
        self._create_combined_comparison(results_df)
        
    def _create_combined_comparison(self, results_df: pd.DataFrame):
        """创建策略对比图"""
        plt.figure(figsize=(14, 10))
        
        # 创建多指标对比
        metrics = ['precision', 'recall', 'f1']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Threshold vs Top-K Strategy Comparison', fontsize=16, fontweight='bold')
        
        # 1. F1分数对比
        ax1 = axes[0, 0]
        threshold_data = results_df[results_df['strategy'] == 'threshold']
        topk_data = results_df[results_df['strategy'] == 'topk']
        
        if not threshold_data.empty:
            ax1.plot(threshold_data['prob_thr'], threshold_data['f1'], 
                    'o-', color='blue', label='Threshold Strategy', linewidth=2, markersize=8)
        
        # 为Top-K创建虚拟x轴位置
        if not topk_data.empty:
            x_pos = [0.6, 0.7, 0.8]  # 对应k=1,3,5的位置
            ax1.plot(x_pos, topk_data['f1'], 
                    's-', color='red', label='Top-K Strategy', linewidth=2, markersize=8)
            
            # 添加Top-K标签
            for x, k, f1 in zip(x_pos, topk_data['topk'], topk_data['f1']):
                ax1.annotate(f'K={int(k)}', (x, f1), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=9)
        
        ax1.set_xlabel('Threshold / Scaled K Position')
        ax1.set_ylabel('F1 Score')
        ax1.set_title('F1 Score Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Precision-Recall散点图
        ax2 = axes[0, 1]
        if not threshold_data.empty:
            ax2.scatter(threshold_data['recall'], threshold_data['precision'], 
                       c='blue', s=100, alpha=0.7, label='Threshold', marker='o')
            for _, row in threshold_data.iterrows():
                ax2.annotate(f'{row["prob_thr"]:.1f}', 
                           (row['recall'], row['precision']), 
                           textcoords="offset points", xytext=(5,5), fontsize=9)
        
        if not topk_data.empty:
            ax2.scatter(topk_data['recall'], topk_data['precision'], 
                       c='red', s=100, alpha=0.7, label='Top-K', marker='s')
            for _, row in topk_data.iterrows():
                ax2.annotate(f'K={int(row["topk"])}', 
                           (row['recall'], row['precision']), 
                           textcoords="offset points", xytext=(5,5), fontsize=9)
        
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Plot')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Hits@K专用图
        ax3 = axes[1, 0]
        if not topk_data.empty:
            bars = ax3.bar(topk_data['topk'], topk_data['hits_at_k'], 
                          color='green', alpha=0.7, edgecolor='black')
            ax3.set_xlabel('K')
            ax3.set_ylabel('Hits@K')
            ax3.set_title('Hits@K Performance')
            ax3.set_xticks(topk_data['topk'])
            
            for bar, hits in zip(bars, topk_data['hits_at_k']):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f'{hits:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. 综合排名
        ax4 = axes[1, 1]
        sorted_df = results_df.sort_values('f1', ascending=True)  # 升序，便于水平条形图
        
        y_pos = range(len(sorted_df))
        colors_rank = ['red' if i == len(sorted_df)-1 else 'lightblue' for i in range(len(sorted_df))]
        
        bars = ax4.barh(y_pos, sorted_df['f1'], color=colors_rank, alpha=0.8, edgecolor='black')
        
        # 设置标签
        labels = []
        for _, row in sorted_df.iterrows():
            if row['strategy'] == 'threshold':
                labels.append(f"Thr={row['prob_thr']:.1f}")
            else:
                labels.append(f"Top-{int(row['topk'])}")
        
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(labels)
        ax4.set_xlabel('F1 Score')
        ax4.set_title('Strategy Ranking by F1 Score')
        
        # 标注最佳
        best_idx = len(sorted_df) - 1
        ax4.text(sorted_df.iloc[best_idx]['f1'] + 0.005, best_idx, '★ Best', 
                va='center', fontweight='bold', color='red')
        
        plt.tight_layout()
        
        # 保存组合图
        combined_path = os.path.join(self.figs_dir, "strategy_comparison.png")
        plt.savefig(combined_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📈 策略对比图保存到: {combined_path}")
        
    def run_optimization(self):
        """运行完整的优化流程"""
        print("🚀 开始阈值和Top-K参数优化...")
        print("="*60)
        
        try:
            # 1. 数据准备
            self.load_and_split_data()
            
            # 2. 重建固定网络
            self.rebuild_bayesian_network()
            
            # 3. 预计算预测结果
            self.compute_predictions_cache()
            
            # 4. 网格搜索
            results_df = self.run_grid_search()
            
            # 5. 保存结果
            self.save_results(results_df)
            
            # 6. 显示最佳结果
            self.print_top_results(results_df)
            
            # 7. 生成可视化
            self.create_heatmap(results_df)
            
            print("\n" + "="*60)
            print("✅ 参数优化完成！")
            print(f"📁 详细结果: {self.results_dir}/grid_metrics.csv")
            print(f"📊 可视化图表: {self.figs_dir}/")
            print("="*60)
            
            return results_df
            
        except Exception as e:
            print(f"\n❌ 优化过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """主函数"""
    # 创建优化器
    optimizer = ThresholdKOptimizer()
    
    # 运行优化
    results_df = optimizer.run_optimization()
    
    if results_df is not None:
        best_row = results_df.sort_values('f1', ascending=False).iloc[0]
        print(f"\n🎯 最优配置建议:")
        if best_row['strategy'] == 'threshold':
            print(f"   使用概率阈值策略: prob_thr = {best_row['prob_thr']:.1f}")
        else:
            print(f"   使用Top-{int(best_row['topk'])}预警策略")
        print(f"   预期性能: F1={best_row['f1']:.4f}")
    else:
        print("\n❌ 优化失败")

if __name__ == "__main__":
    main()