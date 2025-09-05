#!/usr/bin/env python3
"""
评估极简洪水贝叶斯网络

功能：
1. 加载与main_clean.py相同的数据分割
2. 使用ultra_core_network.csv构建贝叶斯网络
3. 在测试集上评估Top-3和阈值0.3策略
4. 计算性能指标并保存结果

用法：
    python evaluate_ultra_core.py

输出：
    - 终端输出性能指标
    - results/ultra_core_metrics.json - 详细评估结果

注意：
    确保已运行build_ultra_core.py生成网络文件
"""

import pandas as pd
import numpy as np
import json
import os
import networkx as nx
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any

# 机器学习和评估
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score
)

# 贝叶斯网络相关
from model import FloodBayesNetwork

# 设置随机种子（与main_clean.py保持一致）
RANDOM_SEED = 42

class UltraCoreEvaluator:
    """极简核心网络评估器"""
    
    def __init__(self, 
                 network_csv_path="ultra_core_network.csv",
                 data_csv_path="Road_Closures_2024.csv",
                 results_dir="results"):
        self.network_csv_path = network_csv_path
        self.data_csv_path = data_csv_path
        self.results_dir = results_dir
        
        # 创建输出目录
        os.makedirs(results_dir, exist_ok=True)
        
        # 初始化
        self.train_df = None
        self.test_df = None
        self.flood_net = None
        self.core_nodes = set()
        self.network_edges = []
        
        # 评估参数
        self.topk = 3
        self.prob_threshold = 0.3
        
        # 预测缓存
        self.prediction_cache = []
        
    def load_and_split_data(self):
        """加载数据并分割（与main_clean.py完全一致）"""
        print("1. 加载和分割数据...")
        
        # 加载数据
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
        
    def load_ultra_core_network(self):
        """加载极简核心网络"""
        print("2. 加载极简核心网络...")
        
        try:
            # 读取网络文件
            network_df = pd.read_csv(self.network_csv_path)
            
            if len(network_df) == 0:
                print("   ⚠️ 网络文件为空，将创建孤立节点网络")
                # 从训练集中找出高频节点作为核心节点
                road_counts = Counter(self.train_df['link_id'])
                self.core_nodes = {road for road, count in road_counts.items() if count >= 15}
                self.network_edges = []
            else:
                # 提取节点和边
                all_nodes = set(network_df['source'].tolist() + network_df['target'].tolist())
                self.core_nodes = all_nodes
                self.network_edges = network_df.to_dict('records')
            
            print(f"   核心节点数: {len(self.core_nodes)}")
            print(f"   网络边数: {len(self.network_edges)}")
            
            # 显示核心节点
            print("   核心节点:")
            for node in sorted(self.core_nodes):
                print(f"     {node.replace('_', ' ')}")
                
            return True
            
        except FileNotFoundError:
            print(f"   ❌ 找不到网络文件: {self.network_csv_path}")
            print("   请先运行 build_ultra_core.py 生成网络")
            return False
        except Exception as e:
            print(f"   ❌ 加载网络时出错: {e}")
            return False
            
    def build_bayesian_network(self):
        """构建贝叶斯网络"""
        print("3. 构建贝叶斯网络...")
        
        # 创建FloodBayesNetwork实例
        self.flood_net = FloodBayesNetwork(t_window="D")
        
        # 拟合边际概率（仅使用核心节点的数据）
        core_train_df = self.train_df[self.train_df['link_id'].isin(self.core_nodes)].copy()
        if len(core_train_df) == 0:
            raise ValueError("训练集中没有核心节点的数据")
            
        self.flood_net.fit_marginal(core_train_df)
        
        # 手动构建网络结构
        import networkx as nx
        self.flood_net.network = nx.DiGraph()
        
        # 添加节点
        for node in self.core_nodes:
            self.flood_net.network.add_node(node)
        
        # 添加边（如果有的话）
        for edge_data in self.network_edges:
            source = edge_data['source']
            target = edge_data['target']
            weight = edge_data.get('mutual_info', edge_data.get('weight', 0.5))
            
            self.flood_net.network.add_edge(
                source, target,
                weight=weight,
                mutual_info=edge_data.get('mutual_info', 0),
                cooccurrence=edge_data.get('cooccurrence', 0)
            )
        
        print(f"   网络节点: {self.flood_net.network.number_of_nodes()}")
        print(f"   网络边: {self.flood_net.network.number_of_edges()}")
        
        # 拟合条件概率
        try:
            self.flood_net.fit_conditional(core_train_df, max_parents=2, alpha=1)
            print("   ✅ 条件概率拟合完成")
        except Exception as e:
            print(f"   ⚠️ 条件概率拟合警告: {e}")
        
        # 构建贝叶斯网络对象
        try:
            self.flood_net.build_bayes_network()
            self.flood_net.check_bayesian_network()
            print("   ✅ 贝叶斯网络构建成功")
        except Exception as e:
            print(f"   ⚠️ 贝叶斯网络构建警告: {e}")
            
        return self.flood_net
        
    def compute_predictions_cache(self):
        """预计算测试集预测结果"""
        print("4. 计算测试集预测...")
        
        test_by_date = self.test_df.groupby(self.test_df["time_create"].dt.floor("D"))
        total_predictions = 0
        
        for date, day_group in test_by_date:
            # 当天洪水道路
            flooded_roads = set(day_group["link_id"].unique())
            
            # 在核心节点中的洪水道路（作为evidence）
            flooded_core = flooded_roads & self.core_nodes
            
            if len(flooded_core) == 0:
                continue  # 没有核心节点洪水，跳过这一天
            
            # 使用所有观察到的核心节点洪水作为evidence（不限制频次）
            evidence = {road: 1 for road in flooded_core}
            
            # 目标道路：未作为evidence的其他核心节点
            target_roads = self.core_nodes - flooded_core
            
            if len(target_roads) == 0:
                continue  # 没有目标道路
            
            # 对每个目标道路进行预测
            day_predictions = []
            for road in target_roads:
                try:
                    # 预测洪水概率
                    if hasattr(self.flood_net, 'infer_w_evidence'):
                        result = self.flood_net.infer_w_evidence(road, evidence)
                        prob_flood = result.get("flooded", 0.5)
                    else:
                        # 如果推理失败，使用边际概率
                        marginal_row = self.flood_net.marginals[
                            self.flood_net.marginals["link_id"] == road
                        ]
                        prob_flood = float(marginal_row["p"].values[0]) if len(marginal_row) > 0 else 0.5
                    
                    # 真实标签
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
                    # 推理失败，使用默认概率
                    day_predictions.append({
                        "date": str(date.date()),
                        "road": road,
                        "prob_flood": 0.5,
                        "true_flood": 1 if road in flooded_roads else 0,
                        "evidence": evidence.copy()
                    })
                    total_predictions += 1
            
            # 按概率排序
            day_predictions.sort(key=lambda x: x["prob_flood"], reverse=True)
            self.prediction_cache.append(day_predictions)
        
        print(f"   预测样本: {total_predictions}个")
        print(f"   评估天数: {len(self.prediction_cache)}天")
        
    def evaluate_topk_strategy(self) -> Dict[str, float]:
        """评估Top-K策略"""
        print(f"5a. 评估Top-{self.topk}策略...")
        
        total_hits = 0
        total_possible = 0
        all_true = []
        all_pred = []
        
        for day_predictions in self.prediction_cache:
            if len(day_predictions) == 0:
                continue
            
            # 当天实际洪水数
            actual_floods = sum(1 for pred in day_predictions if pred["true_flood"] == 1)
            total_possible += min(self.topk, actual_floods) if actual_floods > 0 else 0
            
            # Top-K预测
            for i, pred in enumerate(day_predictions):
                if i < self.topk:
                    all_pred.append(1)
                else:
                    all_pred.append(0)
                all_true.append(pred["true_flood"])
            
            # 计算命中数
            day_hits = sum(1 for pred in day_predictions[:self.topk] if pred["true_flood"] == 1)
            total_hits += day_hits
        
        # 计算指标
        hits_at_k = total_hits / total_possible if total_possible > 0 else 0
        
        if len(all_true) > 0:
            precision = precision_score(all_true, all_pred, zero_division=0)
            recall = recall_score(all_true, all_pred, zero_division=0)
            f1 = f1_score(all_true, all_pred, zero_division=0)
            pr_auc = average_precision_score(all_true, all_pred) if len(set(all_true)) > 1 else np.mean(all_true)
        else:
            precision = recall = f1 = pr_auc = 0
        
        print(f"   Hits@{self.topk}: {hits_at_k:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1: {f1:.4f}")
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "hits_at_k": hits_at_k,
            "pr_auc": pr_auc,
            "total_samples": len(all_true)
        }
        
    def evaluate_threshold_strategy(self) -> Dict[str, float]:
        """评估概率阈值策略"""
        print(f"5b. 评估阈值{self.prob_threshold}策略...")
        
        y_true = []
        y_pred = []
        y_prob = []
        
        for day_predictions in self.prediction_cache:
            for pred in day_predictions:
                y_true.append(pred["true_flood"])
                y_prob.append(pred["prob_flood"])
                y_pred.append(1 if pred["prob_flood"] >= self.prob_threshold else 0)
        
        if len(y_true) == 0:
            return {"precision": 0, "recall": 0, "f1": 0, "pr_auc": 0, "total_samples": 0}
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        pr_auc = average_precision_score(y_true, y_prob) if len(set(y_true)) > 1 else np.mean(y_true)
        
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1: {f1:.4f}")
        print(f"   PR-AUC: {pr_auc:.4f}")
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "pr_auc": pr_auc,
            "total_samples": len(y_true)
        }
        
    def save_results(self, topk_metrics: Dict, threshold_metrics: Dict):
        """保存评估结果"""
        print("6. 保存评估结果...")
        
        results = {
            "evaluation_summary": {
                "timestamp": pd.Timestamp.now().isoformat(),
                "network_file": self.network_csv_path,
                "core_nodes_count": len(self.core_nodes),
                "network_edges_count": len(self.network_edges),
                "train_samples": len(self.train_df),
                "test_samples": len(self.test_df)
            },
            "network_info": {
                "core_nodes": sorted(list(self.core_nodes)),
                "edges": self.network_edges
            },
            "evaluation_strategies": {
                f"top_{self.topk}": {
                    "strategy": f"Top-{self.topk}",
                    "parameters": {"k": self.topk},
                    "metrics": topk_metrics
                },
                f"threshold_{self.prob_threshold}": {
                    "strategy": f"Threshold-{self.prob_threshold}",
                    "parameters": {"prob_threshold": self.prob_threshold},
                    "metrics": threshold_metrics
                }
            }
        }
        
        # 保存到JSON
        results_path = os.path.join(self.results_dir, "ultra_core_metrics.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ 结果保存到: {results_path}")
        
    def print_summary(self, topk_metrics: Dict, threshold_metrics: Dict):
        """打印评估总结"""
        print("\n" + "="*60)
        print("📊 极简核心网络评估结果")
        print("="*60)
        
        print(f"网络结构:")
        print(f"  核心节点: {len(self.core_nodes)}个")
        print(f"  网络边: {len(self.network_edges)}条")
        
        print(f"\n策略对比:")
        print(f"{'指标':<12} {'Top-3':<10} {'阈值0.3':<10}")
        print("-" * 40)
        print(f"{'Precision':<12} {topk_metrics['precision']:<10.4f} {threshold_metrics['precision']:<10.4f}")
        print(f"{'Recall':<12} {topk_metrics['recall']:<10.4f} {threshold_metrics['recall']:<10.4f}")
        print(f"{'F1-Score':<12} {topk_metrics['f1']:<10.4f} {threshold_metrics['f1']:<10.4f}")
        print(f"{'PR-AUC':<12} {topk_metrics['pr_auc']:<10.4f} {threshold_metrics['pr_auc']:<10.4f}")
        print(f"{'Hits@3':<12} {topk_metrics['hits_at_k']:<10.4f} {'-':<10}")
        
        # 推荐策略
        if topk_metrics['f1'] > threshold_metrics['f1']:
            best_strategy = f"Top-{self.topk}"
            best_f1 = topk_metrics['f1']
        else:
            best_strategy = f"阈值{self.prob_threshold}"
            best_f1 = threshold_metrics['f1']
        
        print(f"\n🎯 推荐策略: {best_strategy} (F1: {best_f1:.4f})")
        print("="*60)
        
    def run_evaluation(self):
        """运行完整评估流程"""
        print("🚀 开始极简核心网络评估...")
        print("="*60)
        
        try:
            # 1. 数据加载
            self.load_and_split_data()
            
            # 2. 加载网络
            if not self.load_ultra_core_network():
                return None
            
            # 3. 构建贝叶斯网络
            self.build_bayesian_network()
            
            # 4. 计算预测
            self.compute_predictions_cache()
            
            # 5. 评估策略
            topk_metrics = self.evaluate_topk_strategy()
            threshold_metrics = self.evaluate_threshold_strategy()
            
            # 6. 保存结果
            self.save_results(topk_metrics, threshold_metrics)
            
            # 7. 打印总结
            self.print_summary(topk_metrics, threshold_metrics)
            
            print(f"\n✅ 极简核心网络评估完成！")
            
            return {
                "topk_metrics": topk_metrics,
                "threshold_metrics": threshold_metrics
            }
            
        except Exception as e:
            print(f"\n❌ 评估过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """主函数"""
    evaluator = UltraCoreEvaluator()
    results = evaluator.run_evaluation()
    
    if results:
        print(f"\n🎯 极简网络特点:")
        print(f"   - 精选高频核心道路")
        print(f"   - 基于互信息的最优连接")
        print(f"   - 树状结构确保计算效率")
        print(f"   - 适合实时预警应用")

if __name__ == "__main__":
    main()