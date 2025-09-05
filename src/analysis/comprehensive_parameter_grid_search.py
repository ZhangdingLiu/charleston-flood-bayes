#!/usr/bin/env python3
"""
Comprehensive Parameter Grid Search
全面的参数网格搜索模块

执行贝叶斯网络洪水预测模型的全参数空间搜索，评估所有可能的参数组合
并保存详细的性能结果。

作者：Claude AI
日期：2025-01-21
"""

import random
import numpy as np
import pandas as pd
import warnings
from datetime import datetime
import itertools
import time
import json
import os
import sys
from collections import defaultdict
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from model import FloodBayesNetwork
except ImportError:
    try:
        from core.model import FloodBayesNetwork
    except ImportError:
        print("❌ 无法导入FloodBayesNetwork，请确保model.py或core/model.py存在")
        sys.exit(1)

# 设置随机种子保证可重复性
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class ParameterGridSearcher:
    """参数网格搜索器"""
    
    def __init__(self, param_grid=None, use_validation_split=True):
        """
        初始化参数网格搜索器
        
        Args:
            param_grid (dict): 参数网格字典
            use_validation_split (bool): 是否使用验证集分割
        """
        self.param_grid = param_grid or self._get_default_param_grid()
        self.use_validation_split = use_validation_split
        self.results = []
        self.experiment_config = {}
        
    def _get_default_param_grid(self):
        """获取默认参数网格"""
        return {
            # 网络构建参数
            'occ_thr': [2, 3, 4, 5],           # 道路最小出现次数
            'edge_thr': [1, 2, 3],             # 边的最小共现次数  
            'weight_thr': [0.2, 0.3, 0.4, 0.5], # 边权重阈值
            
            # 评估参数
            'evidence_count': [1, 2, 3, 4],    # 证据道路数量
            'pred_threshold': [0.1, 0.2, 0.3, 0.4, 0.5], # 预测阈值
            
            # 负样本策略
            'neg_pos_ratio': [1.0, 1.5, 2.0], # 负正样本比例
            'marginal_prob_threshold': [0.03, 0.05, 0.08] # 边际概率阈值
        }
    
    def load_and_preprocess_data(self):
        """加载和预处理数据"""
        print("="*80)
        print("加载和预处理洪水数据")
        print("="*80)
        
        # 加载数据
        df = pd.read_csv("Road_Closures_2024.csv")
        df = df[df["REASON"].str.upper() == "FLOOD"].copy()
        
        # 预处理
        df["time_create"] = pd.to_datetime(df["START"], utc=True)
        df["link_id"] = df["STREET"].str.upper().str.replace(" ", "_")
        df["link_id"] = df["link_id"].astype(str)
        df["id"] = df["OBJECTID"].astype(str)
        df["year"] = df["time_create"].dt.year
        df['flood_date'] = df['time_create'].dt.floor('D')
        
        print(f"数据加载完成: {len(df)} 条洪水记录")
        print(f"时间范围: {df['time_create'].min().strftime('%Y-%m-%d')} 到 {df['time_create'].max().strftime('%Y-%m-%d')}")
        print(f"唯一道路数: {df['link_id'].nunique()}")
        print(f"洪水天数: {df['flood_date'].nunique()}")
        
        return df
    
    def split_data_by_flood_days(self, df, train_ratio=0.6, valid_ratio=0.2, test_ratio=0.2):
        """按洪水天数进行时间分割"""
        print(f"\n{'='*60}")
        print("按洪水天数进行时间分割")
        print(f"{'='*60}")
        
        # 按洪水天分组
        flood_days = df.groupby('flood_date').size().sort_index()
        unique_days = flood_days.index.tolist()
        
        print(f"总洪水天数: {len(unique_days)}")
        print(f"日期范围: {unique_days[0].strftime('%Y-%m-%d')} 到 {unique_days[-1].strftime('%Y-%m-%d')}")
        
        # 时间分割
        n_days = len(unique_days)
        train_end = int(n_days * train_ratio)
        valid_end = int(n_days * (train_ratio + valid_ratio))
        
        train_days = unique_days[:train_end]
        valid_days = unique_days[train_end:valid_end] if self.use_validation_split else []
        test_days = unique_days[valid_end:] if self.use_validation_split else unique_days[train_end:]
        
        # 分割数据
        train_df = df[df['flood_date'].isin(train_days)].copy()
        valid_df = df[df['flood_date'].isin(valid_days)].copy() if self.use_validation_split else pd.DataFrame()
        test_df = df[df['flood_date'].isin(test_days)].copy()
        
        print(f"训练集: {len(train_days)} 天, {len(train_df)} 条记录")
        if self.use_validation_split:
            print(f"验证集: {len(valid_days)} 天, {len(valid_df)} 条记录")
        print(f"测试集: {len(test_days)} 天, {len(test_df)} 条记录")
        
        return train_df, valid_df, test_df
    
    def build_bayesian_network(self, train_df, occ_thr, edge_thr, weight_thr):
        """构建贝叶斯网络"""
        try:
            # 构建网络
            flood_net = FloodBayesNetwork(t_window="D")
            flood_net.fit_marginal(train_df)
            flood_net.build_network_by_co_occurrence(
                train_df, 
                occ_thr=occ_thr, 
                edge_thr=edge_thr, 
                weight_thr=weight_thr, 
                report=False
            )
            flood_net.fit_conditional(train_df, max_parents=2, alpha=1.0)
            flood_net.build_bayes_network()
            
            return flood_net, True, ""
        except Exception as e:
            return None, False, str(e)
    
    def evaluate_network(self, flood_net, test_df, evidence_count, pred_threshold, 
                        neg_pos_ratio, marginal_prob_threshold):
        """评估网络性能"""
        try:
            bn_nodes = set(flood_net.network_bayes.nodes())
            marginals_dict = dict(zip(flood_net.marginals['link_id'], flood_net.marginals['p']))
            
            # 获取负样本候选
            negative_candidates = [
                road for road, prob in marginals_dict.items() 
                if road in bn_nodes and prob <= marginal_prob_threshold
            ]
            
            if len(negative_candidates) < 2:
                return None, "负样本候选不足"
            
            # 按日期分组测试数据
            test_by_date = test_df.groupby(test_df["flood_date"])
            
            predictions = []
            valid_days = 0
            total_days = 0
            
            for date, day_group in test_by_date:
                total_days += 1
                
                # 当天洪水道路列表
                flooded_roads = list(day_group["link_id"].unique())
                flooded_in_bn = [road for road in flooded_roads if road in bn_nodes]
                
                if len(flooded_in_bn) < evidence_count + 1:
                    continue
                    
                valid_days += 1
                
                # Evidence选择
                evidence_roads = flooded_in_bn[:evidence_count]
                target_roads = flooded_in_bn[evidence_count:]
                
                evidence = {road: 1 for road in evidence_roads}
                
                # 处理正样本
                for target_road in target_roads:
                    try:
                        result = flood_net.infer_w_evidence(target_road, evidence)
                        prob_flood = result['flooded']
                        
                        predictions.append({
                            'type': 'positive',
                            'road': target_road,
                            'true_label': 1,
                            'prob_flood': prob_flood,
                            'date': date
                        })
                    except:
                        continue
                
                # 处理负样本
                available_negatives = [road for road in negative_candidates if road not in flooded_roads]
                neg_count = min(len(available_negatives), int(len(target_roads) * neg_pos_ratio))
                selected_negatives = available_negatives[:neg_count]
                
                for neg_road in selected_negatives:
                    try:
                        result = flood_net.infer_w_evidence(neg_road, evidence)
                        prob_flood = result['flooded']
                        
                        predictions.append({
                            'type': 'negative',
                            'road': neg_road,
                            'true_label': 0,
                            'prob_flood': prob_flood,
                            'date': date
                        })
                    except:
                        continue
            
            if len(predictions) < 10:
                return None, "预测样本不足"
            
            # 计算性能指标
            tp = fp = tn = fn = 0
            
            for pred in predictions:
                prob = pred['prob_flood']
                true_label = pred['true_label']
                
                # 应用阈值决策
                if prob >= pred_threshold:
                    prediction = 1
                else:
                    prediction = 0
                
                # 计算混淆矩阵
                if prediction == 1 and true_label == 1:
                    tp += 1
                elif prediction == 1 and true_label == 0:
                    fp += 1
                elif prediction == 0 and true_label == 1:
                    fn += 1
                elif prediction == 0 and true_label == 0:
                    tn += 1
            
            # 计算指标
            total_samples = tp + fp + tn + fn
            positive_samples = sum(1 for p in predictions if p['true_label'] == 1)
            negative_samples = sum(1 for p in predictions if p['true_label'] == 0)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = (tp + tn) / total_samples if total_samples > 0 else 0.0
            
            metrics = {
                'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
                'total_samples': total_samples,
                'positive_samples': positive_samples,
                'negative_samples': negative_samples,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'accuracy': accuracy,
                'valid_days': valid_days,
                'total_days': total_days,
                'network_nodes': len(bn_nodes),
                'negative_candidates_count': len(negative_candidates)
            }
            
            return metrics, "成功"
            
        except Exception as e:
            return None, f"评估失败: {str(e)}"
    
    def run_grid_search(self, save_dir="results"):
        """运行网格搜索"""
        print("="*80)
        print("开始参数网格搜索")
        print("="*80)
        
        # 创建结果目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = f"{save_dir}/parameter_optimization_{timestamp}"
        os.makedirs(result_dir, exist_ok=True)
        
        # 保存实验配置
        self.experiment_config = {
            'timestamp': timestamp,
            'param_grid': self.param_grid,
            'use_validation_split': self.use_validation_split,
            'random_seed': RANDOM_SEED
        }
        
        with open(f"{result_dir}/experiment_config.json", 'w') as f:
            json.dump(self.experiment_config, f, indent=2, ensure_ascii=False)
        
        # 加载数据
        df = self.load_and_preprocess_data()
        train_df, valid_df, test_df = self.split_data_by_flood_days(df)
        
        # 生成所有参数组合
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        all_combinations = list(itertools.product(*param_values))
        
        total_combinations = len(all_combinations)
        print(f"\n总参数组合数: {total_combinations}")
        
        # 网格搜索
        results = []
        failed_count = 0
        
        for i, combination in enumerate(all_combinations):
            param_dict = dict(zip(param_names, combination))
            
            print(f"\n进度: {i+1}/{total_combinations} ({(i+1)/total_combinations*100:.1f}%)")
            print(f"当前参数: {param_dict}")
            
            start_time = time.time()
            
            # 构建网络
            flood_net, build_success, build_error = self.build_bayesian_network(
                train_df, param_dict['occ_thr'], param_dict['edge_thr'], param_dict['weight_thr']
            )
            
            if not build_success:
                print(f"❌ 网络构建失败: {build_error}")
                failed_count += 1
                continue
            
            # 评估网络
            metrics, eval_error = self.evaluate_network(
                flood_net, test_df, 
                param_dict['evidence_count'], param_dict['pred_threshold'],
                param_dict['neg_pos_ratio'], param_dict['marginal_prob_threshold']
            )
            
            runtime = time.time() - start_time
            
            if metrics is None:
                print(f"❌ 评估失败: {eval_error}")
                failed_count += 1
                continue
            
            # 合并结果
            result = {**param_dict, **metrics, 'runtime_seconds': runtime}
            results.append(result)
            
            print(f"✅ 成功 - P:{metrics['precision']:.3f}, R:{metrics['recall']:.3f}, F1:{metrics['f1_score']:.3f}")
            
            # 每100个组合保存一次中间结果
            if len(results) % 100 == 0:
                self._save_intermediate_results(results, result_dir)
        
        print(f"\n="*80)
        print("网格搜索完成!")
        print(f"成功评估的组合: {len(results)}")
        print(f"失败的组合: {failed_count}")
        print(f"成功率: {len(results)/(len(results)+failed_count)*100:.1f}%")
        
        # 保存最终结果
        self._save_final_results(results, result_dir)
        
        self.results = results
        return results, result_dir
    
    def _save_intermediate_results(self, results, result_dir):
        """保存中间结果"""
        if results:
            df_results = pd.DataFrame(results)
            df_results.to_csv(f"{result_dir}/intermediate_results.csv", index=False)
            print(f"💾 中间结果已保存: {len(results)} 个组合")
    
    def _save_final_results(self, results, result_dir):
        """保存最终结果"""
        if not results:
            print("⚠️ 没有成功的结果可保存")
            return
        
        # 转换为DataFrame
        df_results = pd.DataFrame(results)
        
        # 保存完整结果
        df_results.to_csv(f"{result_dir}/complete_results.csv", index=False)
        print(f"💾 完整结果已保存: {result_dir}/complete_results.csv")
        
        # 保存性能摘要
        summary = {
            'total_combinations_tested': len(results),
            'best_f1_score': df_results['f1_score'].max(),
            'best_precision': df_results['precision'].max(),
            'best_recall': df_results['recall'].max(),
            'parameter_ranges': {
                param: {
                    'min': df_results[param].min(),
                    'max': df_results[param].max(),
                    'unique_values': sorted(df_results[param].unique().tolist())
                } for param in self.param_grid.keys()
            },
            'performance_statistics': {
                'f1_score': {
                    'mean': df_results['f1_score'].mean(),
                    'std': df_results['f1_score'].std(),
                    'min': df_results['f1_score'].min(),
                    'max': df_results['f1_score'].max()
                },
                'precision': {
                    'mean': df_results['precision'].mean(),
                    'std': df_results['precision'].std(),
                    'min': df_results['precision'].min(),
                    'max': df_results['precision'].max()
                },
                'recall': {
                    'mean': df_results['recall'].mean(),
                    'std': df_results['recall'].std(),
                    'min': df_results['recall'].min(),
                    'max': df_results['recall'].max()
                }
            }
        }
        
        with open(f"{result_dir}/performance_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"📊 性能摘要已保存: {result_dir}/performance_summary.json")
        
        # 删除中间结果文件
        intermediate_file = f"{result_dir}/intermediate_results.csv"
        if os.path.exists(intermediate_file):
            os.remove(intermediate_file)

def main():
    """主函数 - 演示用法"""
    # 创建网格搜索器
    searcher = ParameterGridSearcher()
    
    # 运行网格搜索
    results, result_dir = searcher.run_grid_search()
    
    print(f"\n🎉 参数网格搜索完成！")
    print(f"结果保存在: {result_dir}")
    
    return searcher, results, result_dir

if __name__ == "__main__":
    searcher, results, result_dir = main()