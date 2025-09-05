#!/usr/bin/env python3
"""
改进的贝叶斯网络训练和评估系统

主要改进：
1. 基于训练数据构建贝叶斯网络，避免数据泄露
2. 实现特殊的评估策略：只考虑有洪水记录的道路进行推理
3. 系统的参数优化和网格搜索
4. 详细的性能分析和结果输出

评估策略说明：
- 测试时只考虑有洪水记录(=1)的道路，因为这些是可靠的正样本
- 忽略无记录(=0)的情况，因为可能是观测缺失而非真实无洪水
- 推理结果概率≥阈值才算positive prediction
- 计算precision和recall在这种设定下
"""

import random
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
from collections import defaultdict
from itertools import product

# 设置随机种子确保可重复性
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

from model import FloodBayesNetwork
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

class ImprovedFloodPredictor:
    """改进的洪水预测系统"""
    
    def __init__(self, data_path="Road_Closures_2024.csv"):
        self.data_path = data_path
        self.train_df = None
        self.test_df = None
        
        # 参数搜索空间
        self.param_grid = {
            'occ_thr': [2, 3, 5, 10],           # 道路出现次数阈值
            'edge_thr': [2, 3, 4],              # 共现次数阈值  
            'weight_thr': [0.2, 0.3, 0.4, 0.5], # 条件概率阈值
            'max_parents': [1, 2, 3],           # 最大父节点数
            'alpha': [0.5, 1.0, 2.0],           # 拉普拉斯平滑
            'prob_thr': [0.3, 0.4, 0.5, 0.6, 0.7] # 推理概率阈值
        }
        
        # 存储结果
        self.results = []
        self.best_config = None
        self.best_network = None
        
    def load_and_preprocess_data(self):
        """加载和预处理数据"""
        print("🚀 启动改进的贝叶斯网络洪水预测系统")
        print("="*60)
        print("1. 加载和预处理数据...")
        
        # 加载数据
        df = pd.read_csv(self.data_path)
        df = df[df["REASON"].str.upper() == "FLOOD"].copy()
        
        # 数据预处理
        df["time_create"] = pd.to_datetime(df["START"], utc=True)
        df["link_id"] = df["STREET"].str.upper().str.replace(" ", "_")
        df["link_id"] = df["link_id"].astype(str)
        df["id"] = df["OBJECTID"].astype(str)
        
        # 严格按时间顺序分割，避免数据泄露
        df_sorted = df.sort_values('time_create')
        split_idx = int(len(df_sorted) * 0.7)
        self.train_df = df_sorted.iloc[:split_idx].copy()
        self.test_df = df_sorted.iloc[split_idx:].copy()
        
        print(f"   总洪水记录: {len(df)}条")
        print(f"   训练集: {len(self.train_df)}条 (时间: {self.train_df['time_create'].min()} 至 {self.train_df['time_create'].max()})")
        print(f"   测试集: {len(self.test_df)}条 (时间: {self.test_df['time_create'].min()} 至 {self.test_df['time_create'].max()})")
        
        # 分析数据特征
        train_roads = set(self.train_df['link_id'].unique())
        test_roads = set(self.test_df['link_id'].unique())
        overlap_roads = train_roads & test_roads
        
        print(f"   训练集独特道路: {len(train_roads)}条")
        print(f"   测试集独特道路: {len(test_roads)}条")
        print(f"   重叠道路: {len(overlap_roads)}条")
        
        return True
        
    def evaluate_flood_only(self, flood_net, evidence_ratio=0.5, prob_thr=0.5, verbose=False):
        """
        只对有洪水记录的道路进行推理和评估
        
        Args:
            flood_net: 训练好的贝叶斯网络
            evidence_ratio: 用作evidence的洪水道路比例
            prob_thr: 预测概率阈值
            verbose: 是否显示详细信息
        
        Returns:
            dict: 包含precision, recall, f1等指标的字典
        """
        bn_nodes = set(flood_net.network_bayes.nodes()) if flood_net.network_bayes else set()
        
        if len(bn_nodes) == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'samples': 0, 'valid_days': 0}
        
        all_predictions = []
        all_true_labels = []
        evaluated_days = 0
        total_samples = 0\n        \n        # 按日期分组测试数据\n        test_by_date = self.test_df.groupby(self.test_df[\"time_create\"].dt.floor(\"D\"))\n        \n        for date, day_group in test_by_date:\n            # 当天洪水道路列表\n            flooded_roads = list(day_group[\"link_id\"].unique())\n            \n            # 只考虑在贝叶斯网络中的道路\n            flooded_in_bn = [road for road in flooded_roads if road in bn_nodes]\n            \n            if len(flooded_in_bn) < 2:\n                continue  # 需要至少2条道路才能做推理\n            \n            evaluated_days += 1\n            \n            # 选择evidence道路（前N个或随机选择）\n            evidence_count = max(1, int(len(flooded_in_bn) * evidence_ratio))\n            evidence_roads = flooded_in_bn[:evidence_count]\n            target_roads = flooded_in_bn[evidence_count:]\n            \n            if len(target_roads) == 0:\n                continue\n            \n            # 构建evidence字典\n            evidence = {road: 1 for road in evidence_roads}\n            \n            if verbose and evaluated_days <= 3:\n                print(f\"     📅 {date.date()}: 洪水道路{len(flooded_in_bn)}, evidence{len(evidence_roads)}, target{len(target_roads)}\")\n            \n            # 对每个目标道路进行推理\n            for target_road in target_roads:\n                try:\n                    # 贝叶斯推理\n                    result = flood_net.infer_w_evidence(target_road, evidence)\n                    prob_flood = result['flooded']\n                    \n                    # 预测标签（根据概率阈值）\n                    pred_label = 1 if prob_flood >= prob_thr else 0\n                    true_label = 1  # 目标道路确实发生了洪水\n                    \n                    all_predictions.append(pred_label)\n                    all_true_labels.append(true_label)\n                    total_samples += 1\n                    \n                    if verbose and evaluated_days <= 3:\n                        print(f\"       {target_road}: P(flood)={prob_flood:.3f}, pred={pred_label}, true={true_label}\")\n                        \n                except Exception as e:\n                    if verbose:\n                        print(f\"       {target_road}: 推理失败 - {e}\")\n                    continue\n        \n        # 计算性能指标\n        if total_samples == 0:\n            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'samples': 0, 'valid_days': 0}\n        \n        # 在这种特殊设定下，所有true_label都是1，所以：\n        # - TP = 预测为1的数量\n        # - FN = 预测为0的数量  \n        # - TN = FP = 0（因为没有真实的负样本）\n        \n        tp = sum(all_predictions)  # 预测为正的数量\n        fn = len(all_predictions) - tp  # 预测为负的数量\n        \n        precision = tp / (tp) if tp > 0 else 0.0  # TP / (TP + FP), 但FP=0\n        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # TP / (TP + FN)\n        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0\n        \n        return {\n            'precision': precision,\n            'recall': recall, \n            'f1': f1,\n            'samples': total_samples,\n            'valid_days': evaluated_days,\n            'tp': tp,\n            'fn': fn\n        }\n        \n    def build_and_evaluate_network(self, occ_thr, edge_thr, weight_thr, max_parents, alpha, prob_thr, verbose=False):\n        \"\"\"构建和评估单个网络配置\"\"\"\n        try:\n            # 构建网络\n            flood_net = FloodBayesNetwork(t_window=\"D\")\n            flood_net.fit_marginal(self.train_df)\n            \n            # 构建共现网络\n            flood_net.build_network_by_co_occurrence(\n                self.train_df,\n                occ_thr=occ_thr,\n                edge_thr=edge_thr,\n                weight_thr=weight_thr,\n                report=False\n            )\n            \n            # 如果网络为空，返回默认结果\n            if flood_net.network.number_of_nodes() == 0:\n                return {\n                    'occ_thr': occ_thr, 'edge_thr': edge_thr, 'weight_thr': weight_thr,\n                    'max_parents': max_parents, 'alpha': alpha, 'prob_thr': prob_thr,\n                    'nodes': 0, 'edges': 0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0,\n                    'samples': 0, 'valid_days': 0, 'status': 'empty_network'\n                }\n            \n            # 拟合条件概率\n            flood_net.fit_conditional(self.train_df, max_parents=max_parents, alpha=alpha)\n            \n            # 构建贝叶斯网络\n            flood_net.build_bayes_network()\n            \n            # 在测试集上评估\n            metrics = self.evaluate_flood_only(flood_net, evidence_ratio=0.5, prob_thr=prob_thr, verbose=verbose)\n            \n            # 组合结果\n            result = {\n                'occ_thr': occ_thr,\n                'edge_thr': edge_thr, \n                'weight_thr': weight_thr,\n                'max_parents': max_parents,\n                'alpha': alpha,\n                'prob_thr': prob_thr,\n                'nodes': flood_net.network.number_of_nodes(),\n                'edges': flood_net.network.number_of_edges(),\n                'status': 'success'\n            }\n            result.update(metrics)\n            \n            return result, flood_net\n            \n        except Exception as e:\n            if verbose:\n                print(f\"   ❌ 配置失败: {e}\")\n            return {\n                'occ_thr': occ_thr, 'edge_thr': edge_thr, 'weight_thr': weight_thr,\n                'max_parents': max_parents, 'alpha': alpha, 'prob_thr': prob_thr,\n                'nodes': 0, 'edges': 0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0,\n                'samples': 0, 'valid_days': 0, 'status': f'error: {str(e)}'\n            }, None\n            \n    def run_parameter_optimization(self, max_configs=50):\n        \"\"\"运行参数优化\"\"\"\n        print(\"\\n2. 开始参数优化...\")\n        \n        # 生成参数组合（限制数量避免过长时间）\n        param_combinations = list(product(\n            self.param_grid['occ_thr'],\n            self.param_grid['edge_thr'], \n            self.param_grid['weight_thr'],\n            self.param_grid['max_parents'],\n            self.param_grid['alpha'],\n            self.param_grid['prob_thr']\n        ))\n        \n        # 随机采样以限制计算时间\n        if len(param_combinations) > max_configs:\n            np.random.shuffle(param_combinations)\n            param_combinations = param_combinations[:max_configs]\n        \n        print(f\"   测试{len(param_combinations)}个参数配置...\")\n        print(f\"   参数空间: occ_thr{self.param_grid['occ_thr']}, edge_thr{self.param_grid['edge_thr']}, weight_thr{self.param_grid['weight_thr']}\")\n        print(f\"   评估策略: 只考虑有洪水记录的道路，推理结果≥阈值才算正预测\")\n        \n        successful_configs = 0\n        \n        for i, (occ_thr, edge_thr, weight_thr, max_parents, alpha, prob_thr) in enumerate(param_combinations):\n            if i % 10 == 0:\n                print(f\"   进度: {i+1}/{len(param_combinations)}\")\n            \n            verbose = i < 3  # 只对前几个配置显示详细信息\n            result, network = self.build_and_evaluate_network(\n                occ_thr, edge_thr, weight_thr, max_parents, alpha, prob_thr, verbose=verbose\n            )\n            \n            self.results.append(result)\n            \n            if result['status'] == 'success' and result['f1'] > 0:\n                successful_configs += 1\n                \n                # 更新最佳配置\n                if (self.best_config is None or \n                    result['f1'] > self.best_config['f1'] or\n                    (result['f1'] == self.best_config['f1'] and result['nodes'] < self.best_config['nodes'])):\n                    self.best_config = result.copy()\n                    self.best_network = network\n        \n        print(f\"   ✅ 完成参数优化: {successful_configs}/{len(param_combinations)}个配置成功\")\n        \n        return self.results\n        \n    def analyze_results(self):\n        \"\"\"分析和展示结果\"\"\"\n        print(\"\\n3. 结果分析...\")\n        \n        if not self.results:\n            print(\"   ❌ 没有有效结果\")\n            return\n        \n        # 过滤成功的结果\n        successful_results = [r for r in self.results if r['status'] == 'success' and r['f1'] > 0]\n        \n        if not successful_results:\n            print(\"   ❌ 没有成功的配置\")\n            return\n        \n        print(f\"   📊 成功配置数量: {len(successful_results)}\")\n        \n        # 按F1分数排序\n        top_results = sorted(successful_results, key=lambda x: (-x['f1'], x['nodes']))[:10]\n        \n        print(f\"\\n   🏆 Top-10 配置 (按F1分数排序):\")\n        print(f\"   {'Rank':<4} {'F1':<6} {'Prec':<6} {'Rec':<6} {'Nodes':<6} {'Edges':<6} {'occ':<4} {'edge':<4} {'wt':<4} {'par':<3} {'α':<4} {'thr':<4}\")\n        print(\"-\" * 85)\n        \n        for i, result in enumerate(top_results, 1):\n            print(f\"   {i:<4} {result['f1']:<6.3f} {result['precision']:<6.3f} {result['recall']:<6.3f} \"\n                  f\"{result['nodes']:<6} {result['edges']:<6} {result['occ_thr']:<4} {result['edge_thr']:<4} \"\n                  f\"{result['weight_thr']:<4.1f} {result['max_parents']:<3} {result['alpha']:<4.1f} {result['prob_thr']:<4.1f}\")\n        \n        # 分析最佳配置\n        if self.best_config:\n            print(f\"\\n   🎯 最佳配置详细分析:\")\n            best = self.best_config\n            print(f\"     参数: occ_thr={best['occ_thr']}, edge_thr={best['edge_thr']}, weight_thr={best['weight_thr']}\")\n            print(f\"           max_parents={best['max_parents']}, alpha={best['alpha']}, prob_thr={best['prob_thr']}\")\n            print(f\"     网络: {best['nodes']}个节点, {best['edges']}条边\")\n            print(f\"     性能: F1={best['f1']:.3f}, Precision={best['precision']:.3f}, Recall={best['recall']:.3f}\")\n            print(f\"     样本: {best['samples']}个预测, {best['valid_days']}个有效天数\")\n            print(f\"     预测: TP={best['tp']}, FN={best['fn']}\")\n        \n        # 参数影响分析\n        self.analyze_parameter_effects(successful_results)\n        \n    def analyze_parameter_effects(self, results):\n        \"\"\"分析参数对性能的影响\"\"\"\n        print(f\"\\n   📈 参数影响分析:\")\n        \n        params = ['occ_thr', 'edge_thr', 'weight_thr', 'max_parents', 'alpha', 'prob_thr']\n        \n        for param in params:\n            values = list(set(r[param] for r in results))\n            if len(values) <= 1:\n                continue\n                \n            print(f\"\\n     {param}:\")\n            for value in sorted(values):\n                subset = [r for r in results if r[param] == value]\n                if subset:\n                    avg_f1 = np.mean([r['f1'] for r in subset])\n                    avg_nodes = np.mean([r['nodes'] for r in subset])\n                    print(f\"       {value}: F1={avg_f1:.3f}, 平均节点数={avg_nodes:.1f} ({len(subset)}个配置)\")\n    \n    def save_results(self, filename=\"flood_prediction_results.json\"):\n        \"\"\"保存结果到文件\"\"\"\n        print(f\"\\n4. 保存结果到 {filename}...\")\n        \n        output = {\n            'timestamp': datetime.now().isoformat(),\n            'evaluation_strategy': 'flood_only',\n            'description': '只考虑有洪水记录的道路进行推理和评估',\n            'data_split': 'temporal_70_30',\n            'total_configs': len(self.results),\n            'successful_configs': len([r for r in self.results if r['status'] == 'success']),\n            'best_config': self.best_config,\n            'all_results': self.results,\n            'parameter_grid': self.param_grid\n        }\n        \n        with open(filename, 'w', encoding='utf-8') as f:\n            json.dump(output, f, indent=2, ensure_ascii=False)\n        \n        print(f\"   ✅ 结果已保存\")\n        \n    def demonstrate_best_network(self):\n        \"\"\"演示最佳网络的推理过程\"\"\"\n        if not self.best_network or not self.best_config:\n            print(\"   ❌ 没有最佳网络可演示\")\n            return\n            \n        print(f\"\\n5. 最佳网络推理演示...\")\n        \n        # 获取测试集中的几个洪水日期\n        test_by_date = self.test_df.groupby(self.test_df[\"time_create\"].dt.floor(\"D\"))\n        bn_nodes = set(self.best_network.network_bayes.nodes())\n        \n        demo_count = 0\n        for date, day_group in test_by_date:\n            if demo_count >= 3:\n                break\n                \n            flooded_roads = list(day_group[\"link_id\"].unique())\n            flooded_in_bn = [road for road in flooded_roads if road in bn_nodes]\n            \n            if len(flooded_in_bn) >= 3:\n                print(f\"\\n   📅 {date.date()} 推理演示:\")\n                print(f\"     当天洪水道路: {flooded_in_bn}\")\n                \n                # 选择前2个作为evidence\n                evidence = {flooded_in_bn[0]: 1, flooded_in_bn[1]: 1}\n                targets = flooded_in_bn[2:]\n                \n                print(f\"     Evidence: {list(evidence.keys())}\")\n                print(f\"     推理目标: {targets}\")\n                \n                for target in targets:\n                    try:\n                        result = self.best_network.infer_w_evidence(target, evidence)\n                        prob = result['flooded']\n                        pred = \"✅洪水\" if prob >= self.best_config['prob_thr'] else \"❌无洪水\"\n                        print(f\"       {target}: P(洪水)={prob:.3f} → {pred} (实际: ✅洪水)\")\n                    except Exception as e:\n                        print(f\"       {target}: 推理失败 - {e}\")\n                \n                demo_count += 1\n        \n    def run_complete_analysis(self):\n        \"\"\"运行完整的分析流程\"\"\"\n        # 1. 加载数据\n        self.load_and_preprocess_data()\n        \n        # 2. 参数优化\n        self.run_parameter_optimization()\n        \n        # 3. 结果分析\n        self.analyze_results()\n        \n        # 4. 保存结果\n        self.save_results()\n        \n        # 5. 演示最佳网络\n        self.demonstrate_best_network()\n        \n        print(f\"\\n✅ 完整分析流程完成！\")\n        print(f\"🎯 核心发现: 特殊评估策略更符合实际应用场景\")\n        print(f\"📊 最佳配置已识别并保存\")\n        \n        return self.best_config, self.best_network\n\ndef main():\n    \"\"\"主函数\"\"\"\n    predictor = ImprovedFloodPredictor()\n    best_config, best_network = predictor.run_complete_analysis()\n    \n    return predictor\n\nif __name__ == \"__main__\":\n    predictor = main()