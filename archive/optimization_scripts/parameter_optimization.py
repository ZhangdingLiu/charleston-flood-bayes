import pandas as pd
import numpy as np
from model import FloodBayesNetwork
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    brier_score_loss, log_loss, roc_auc_score,
    average_precision_score, accuracy_score, f1_score,
    recall_score, precision_score
)
import math
import warnings
from collections import defaultdict, Counter
import os
import json
import random
from typing import Dict, List, Tuple

# 设置随机种子
random.seed(0)
np.random.seed(0)

class ParameterOptimizer:
    def __init__(self, csv_path: str = "Road_Closures_2024.csv"):
        self.csv_path = csv_path
        self.df = None
        self.train_df = None
        self.test_df = None
        self.load_data()
        
    def load_data(self):
        """加载并预处理数据"""
        self.df = pd.read_csv(self.csv_path)
        self.df = self.df[self.df["REASON"].str.upper() == "FLOOD"].copy()
        
        # 数据预处理
        self.df["time_create"] = pd.to_datetime(self.df["START"], utc=True)
        self.df["link_id"] = self.df["STREET"].str.upper().str.replace(" ", "_")
        self.df["link_id"] = self.df["link_id"].astype(str)
        self.df["id"] = self.df["OBJECTID"].astype(str)
        
        # 训练测试集分割
        self.train_df, self.test_df = train_test_split(
            self.df, test_size=0.3, random_state=42
        )
        
        print(f"总数据量: {len(self.df)}")
        print(f"训练集: {len(self.train_df)}, 测试集: {len(self.test_df)}")
        
    def _binary_entropy(self, p: float) -> float:
        """计算二元熵"""
        if p in (0, 1):
            return 0.0
        return -(p*math.log2(p) + (1-p)*math.log2(1-p))
        
    def evaluate_bn_enhanced(self, flood_net, test_df, evidence_size=3, k=5, prob_thr=0.5):
        """
        优化后的评估函数，更好地处理稀疏数据
        """
        bn_nodes = set(flood_net.network_bayes.nodes())
        y_true, y_prob = [], []
        
        # 统计信息
        hits_at_k = total_at_k = 0
        h_clim, h_post = 0.0, 0.0
        
        # 获取道路频次信息用于选择evidence
        road_counts = Counter(self.train_df['link_id'])
        frequent_roads = {road for road, count in road_counts.items() if count >= 2}
        
        # 按日期迭代
        for day, group in test_df.groupby(test_df["time_create"].dt.floor("D")):
            flooded = [r for r in group["link_id"] if r in bn_nodes]
            
            # 优化evidence选择策略：优先选择频次>=2的道路
            frequent_flooded = [r for r in flooded if r in frequent_roads]
            if len(frequent_flooded) >= evidence_size:
                evidence_roads = frequent_flooded[:evidence_size]
            else:
                evidence_roads = frequent_flooded + flooded[:evidence_size-len(frequent_flooded)]
            
            evidence = {r: 1 for r in evidence_roads}
            
            # 计算基线熵
            for r in bn_nodes:
                p0 = float(flood_net.marginals.loc[
                           flood_net.marginals["link_id"] == r, "p"].values[0])
                h_clim += self._binary_entropy(p0)
            
            # 预测其他道路
            for r in bn_nodes:
                if r in evidence:
                    continue
                    
                try:
                    p_flood = flood_net.infer_w_evidence(r, evidence)["flooded"]
                    y_prob.append(p_flood)
                    y_true.append(1 if r in flooded else 0)
                    h_post += self._binary_entropy(p_flood)
                except Exception as e:
                    # 跳过推理失败的情况
                    continue
            
            # Recall@k计算
            try:
                topk = sorted(
                    ((r, flood_net.infer_w_evidence(r, evidence)['flooded'])
                     for r in bn_nodes if r not in evidence),
                    key=lambda x: x[1], reverse=True
                )[:k]
                
                hits_at_k += sum(1 for r, _ in topk if r in flooded)
                total_at_k += min(k, len(flooded))
            except Exception:
                continue
        
        # 计算指标
        if len(y_true) == 0:
            return {metric: 0.0 for metric in ["Brier", "LogLoss", "ROC-AUC", "PR-AUC", 
                                               "Accuracy", "Precision", "F1", f"Recall@{k}", 
                                               "ΔH(bits)", "BSS", "Samples"]}
        
        # 处理类别不平衡
        pos_ratio = sum(y_true) / len(y_true)
        
        y_hat = (np.array(y_prob) >= prob_thr).astype(int)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            metrics = {
                "Brier": brier_score_loss(y_true, y_prob),
                "LogLoss": log_loss(y_true, y_prob),
                "ROC-AUC": roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.5,
                "PR-AUC": average_precision_score(y_true, y_prob) if len(set(y_true)) > 1 else pos_ratio,
                "Accuracy": accuracy_score(y_true, y_hat),
                "Precision": precision_score(y_true, y_hat, zero_division=0),
                "F1": f1_score(y_true, y_hat, zero_division=0),
                f"Recall@{k}": hits_at_k / total_at_k if total_at_k > 0 else 0.0,
                "ΔH(bits)": h_clim - h_post,
                "Samples": len(y_true),
                "Pos_ratio": pos_ratio
            }
            
            # Brier Skill Score
            clim_bs = brier_score_loss(y_true, [pos_ratio]*len(y_true))
            metrics["BSS"] = 1 - metrics["Brier"]/clim_bs if clim_bs > 0 else 0.0
            
        return metrics
        
    def analyze_network_statistics(self, network):
        """分析网络统计信息"""
        if network.number_of_nodes() == 0:
            return {
                "nodes": 0, "edges": 0, "density": 0.0,
                "avg_degree": 0.0, "max_degree": 0,
                "avg_in_degree": 0.0, "max_in_degree": 0,
                "avg_out_degree": 0.0, "max_out_degree": 0,
                "isolated_nodes": 0
            }
        
        # 基本统计
        n_nodes = network.number_of_nodes()
        n_edges = network.number_of_edges()
        
        # 度分布
        in_degrees = dict(network.in_degree())
        out_degrees = dict(network.out_degree())
        total_degrees = {node: in_degrees[node] + out_degrees[node] for node in network.nodes()}
        
        # 密度
        max_edges = n_nodes * (n_nodes - 1)
        density = n_edges / max_edges if max_edges > 0 else 0.0
        
        # 孤立节点
        isolated = sum(1 for node in network.nodes() if total_degrees[node] == 0)
        
        return {
            "nodes": n_nodes,
            "edges": n_edges,
            "density": density,
            "avg_degree": np.mean(list(total_degrees.values())),
            "max_degree": max(total_degrees.values()) if total_degrees else 0,
            "avg_in_degree": np.mean(list(in_degrees.values())),
            "max_in_degree": max(in_degrees.values()) if in_degrees else 0,
            "avg_out_degree": np.mean(list(out_degrees.values())),
            "max_out_degree": max(out_degrees.values()) if out_degrees else 0,
            "isolated_nodes": isolated
        }
        
    def analyze_edge_weights(self, network):
        """分析边权重分布"""
        if network.number_of_edges() == 0:
            return {
                "weight_mean": 0.0, "weight_std": 0.0, "weight_min": 0.0,
                "weight_max": 0.0, "weight_median": 0.0, "weight_q25": 0.0,
                "weight_q75": 0.0
            }
        
        weights = [network[u][v]['weight'] for u, v in network.edges()]
        
        return {
            "weight_mean": np.mean(weights),
            "weight_std": np.std(weights),
            "weight_min": np.min(weights),
            "weight_max": np.max(weights),
            "weight_median": np.median(weights),
            "weight_q25": np.percentile(weights, 25),
            "weight_q75": np.percentile(weights, 75)
        }
        
    def test_parameter_combinations(self):
        """测试三种参数策略组合"""
        
        # 定义三种策略
        strategies = {
            "conservative": {
                "occ_thr": 3,
                "edge_thr": 4,
                "weight_thr": 0.5,
                "description": "保守策略 - 严格筛选，确保连接质量"
            },
            "moderate": {
                "occ_thr": 2,
                "edge_thr": 3,
                "weight_thr": 0.4,
                "description": "中等策略 - 平衡网络大小和连接质量"
            },
            "relaxed": {
                "occ_thr": 2,
                "edge_thr": 3,
                "weight_thr": 0.3,
                "description": "宽松策略 - 保留更多连接，增加网络覆盖"
            }
        }
        
        results = {}
        
        for strategy_name, params in strategies.items():
            print(f"\n=== 测试{strategy_name}策略 ===")
            print(f"参数: {params}")
            
            try:
                # 创建网络
                flood_net = FloodBayesNetwork(t_window="D")
                flood_net.fit_marginal(self.train_df)
                
                # 构建网络
                flood_net.build_network_by_co_occurrence(
                    self.train_df,
                    occ_thr=params["occ_thr"],
                    edge_thr=params["edge_thr"],
                    weight_thr=params["weight_thr"],
                    report=False
                )
                
                # 分析网络统计
                network_stats = self.analyze_network_statistics(flood_net.network)
                edge_stats = self.analyze_edge_weights(flood_net.network)
                
                print(f"网络大小: {network_stats['nodes']}节点, {network_stats['edges']}边")
                print(f"网络密度: {network_stats['density']:.4f}")
                print(f"平均度: {network_stats['avg_degree']:.2f}")
                
                # 如果网络太小，跳过性能评估
                if network_stats['nodes'] < 3:
                    print("⚠️ 网络太小，跳过性能评估")
                    results[strategy_name] = {
                        "params": params,
                        "network_stats": network_stats,
                        "edge_stats": edge_stats,
                        "performance": None,
                        "status": "network_too_small"
                    }
                    continue
                
                # 拟合条件概率
                flood_net.fit_conditional(self.train_df, max_parents=2, alpha=1)
                flood_net.build_bayes_network()
                
                # 性能评估
                performance = self.evaluate_bn_enhanced(flood_net, self.test_df)
                
                print(f"性能指标:")
                print(f"  ROC-AUC: {performance['ROC-AUC']:.4f}")
                print(f"  PR-AUC: {performance['PR-AUC']:.4f}")
                print(f"  F1: {performance['F1']:.4f}")
                print(f"  Precision: {performance['Precision']:.4f}")
                print(f"  Recall@5: {performance['Recall@5']:.4f}")
                
                results[strategy_name] = {
                    "params": params,
                    "network_stats": network_stats,
                    "edge_stats": edge_stats,
                    "performance": performance,
                    "status": "success"
                }
                
            except Exception as e:
                print(f"❌ 策略{strategy_name}失败: {e}")
                results[strategy_name] = {
                    "params": params,
                    "network_stats": None,
                    "edge_stats": None,
                    "performance": None,
                    "status": f"failed: {str(e)}"
                }
        
        return results
        
    def generate_comparison_report(self, results):
        """生成详细的比较报告"""
        report = []
        report.append("=" * 80)
        report.append("贝叶斯网络参数优化比较报告")
        report.append("=" * 80)
        
        # 策略总览
        report.append(f"\n【策略总览】")
        for strategy, data in results.items():
            if data["status"] == "success":
                stats = data["network_stats"]
                perf = data["performance"]
                report.append(f"{strategy:12s}: {stats['nodes']:2d}节点, {stats['edges']:2d}边, "
                             f"ROC-AUC={perf['ROC-AUC']:.3f}, F1={perf['F1']:.3f}")
            else:
                report.append(f"{strategy:12s}: {data['status']}")
        
        # 详细分析
        for strategy, data in results.items():
            if data["status"] != "success":
                continue
                
            report.append(f"\n【{strategy.upper()}策略详细分析】")
            
            # 参数设置
            params = data["params"]
            report.append(f"参数设置: occ_thr={params['occ_thr']}, edge_thr={params['edge_thr']}, "
                         f"weight_thr={params['weight_thr']}")
            
            # 网络统计
            stats = data["network_stats"]
            report.append(f"网络统计:")
            report.append(f"  - 节点数: {stats['nodes']}")
            report.append(f"  - 边数: {stats['edges']}")
            report.append(f"  - 密度: {stats['density']:.4f}")
            report.append(f"  - 平均度: {stats['avg_degree']:.2f}")
            report.append(f"  - 最大度: {stats['max_degree']}")
            report.append(f"  - 孤立节点: {stats['isolated_nodes']}")
            
            # 边权重分布
            edge_stats = data["edge_stats"]
            report.append(f"边权重分布:")
            report.append(f"  - 平均权重: {edge_stats['weight_mean']:.4f}")
            report.append(f"  - 权重标准差: {edge_stats['weight_std']:.4f}")
            report.append(f"  - 权重范围: [{edge_stats['weight_min']:.4f}, {edge_stats['weight_max']:.4f}]")
            
            # 性能指标
            perf = data["performance"]
            report.append(f"性能指标:")
            report.append(f"  - ROC-AUC: {perf['ROC-AUC']:.4f}")
            report.append(f"  - PR-AUC: {perf['PR-AUC']:.4f}")
            report.append(f"  - F1分数: {perf['F1']:.4f}")
            report.append(f"  - 精确率: {perf['Precision']:.4f}")
            report.append(f"  - Recall@5: {perf['Recall@5']:.4f}")
            report.append(f"  - Brier分数: {perf['Brier']:.4f}")
            report.append(f"  - 样本数: {perf['Samples']}")
            
        # 推荐策略
        report.append(f"\n【推荐策略】")
        
        # 根据不同目标推荐策略
        successful_strategies = {k: v for k, v in results.items() if v["status"] == "success"}
        
        if successful_strategies:
            # 按网络大小推荐
            size_suitable = {k: v for k, v in successful_strategies.items() 
                           if 10 <= v["network_stats"]["nodes"] <= 30}
            
            if size_suitable:
                best_f1 = max(size_suitable.keys(), 
                             key=lambda x: size_suitable[x]["performance"]["F1"])
                best_auc = max(size_suitable.keys(), 
                              key=lambda x: size_suitable[x]["performance"]["ROC-AUC"])
                
                report.append(f"✅ 推荐用于实际应用: {best_f1}策略")
                report.append(f"   - 原因: 网络大小合适且F1分数最高")
                report.append(f"   - 网络规模: {successful_strategies[best_f1]['network_stats']['nodes']}节点")
                report.append(f"   - F1分数: {successful_strategies[best_f1]['performance']['F1']:.4f}")
                
                if best_auc != best_f1:
                    report.append(f"🔄 备选方案: {best_auc}策略")
                    report.append(f"   - 原因: ROC-AUC最高")
                    report.append(f"   - ROC-AUC: {successful_strategies[best_auc]['performance']['ROC-AUC']:.4f}")
            else:
                report.append("⚠️ 没有策略产生理想的网络大小(10-30节点)")
                report.append("   建议进一步调整参数或考虑其他建模方法")
        
        return "\n".join(report)
        
    def save_results(self, results, output_path="parameter_optimization_results.json"):
        """保存结果到文件"""
        # 转换numpy类型为Python原生类型
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
        
        results_cleaned = convert_numpy(results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_cleaned, f, indent=2, ensure_ascii=False)
        
        print(f"结果已保存到: {output_path}")
        
    def run_optimization(self):
        """运行完整的参数优化分析"""
        print("开始贝叶斯网络参数优化分析...")
        
        # 测试参数组合
        results = self.test_parameter_combinations()
        
        # 生成报告
        report = self.generate_comparison_report(results)
        print("\n" + report)
        
        # 保存结果
        self.save_results(results)
        
        return results, report

def main():
    """主函数"""
    try:
        optimizer = ParameterOptimizer()
        results, report = optimizer.run_optimization()
        
        # 保存报告到文件
        with open("parameter_optimization_report.txt", 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("\n" + "="*60)
        print("参数优化分析完成!")
        print("详细报告已保存到: parameter_optimization_report.txt")
        print("JSON结果已保存到: parameter_optimization_results.json")
        
    except Exception as e:
        print(f"参数优化过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()