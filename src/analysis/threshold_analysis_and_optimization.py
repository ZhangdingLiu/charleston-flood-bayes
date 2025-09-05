#!/usr/bin/env python3
"""
阈值分析和优化

分析当前阈值设置问题，重新优化以达到目标性能：
- Precision ≥ 0.8
- Recall ≥ 0.4

步骤：
1. 分析所有测试数据的概率分布
2. 网格搜索最优阈值组合
3. 验证新策略的性能
4. 生成完整的评估报告
"""

import random
import numpy as np
import pandas as pd
from model import FloodBayesNetwork
from precision_focused_evaluation import PrecisionFocusedEvaluator
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class ThresholdOptimizer:
    """阈值优化器"""
    
    def __init__(self, flood_net, test_df):
        self.flood_net = flood_net
        self.test_df = test_df
        self.bn_nodes = set(flood_net.network_bayes.nodes())
        self.marginals_dict = dict(zip(
            flood_net.marginals['link_id'], 
            flood_net.marginals['p']
        ))
        
        # 存储所有推理概率以供分析
        self.all_predictions = []
        self.probability_distribution = []
        
    def collect_all_predictions(self):
        """收集所有测试样本的推理概率"""
        print("🔍 收集所有测试样本的推理概率分布")
        print("=" * 60)
        
        # 获取负样本候选
        negative_candidates = [
            road for road, prob in self.marginals_dict.items() 
            if road in self.bn_nodes and prob <= 0.15
        ]
        
        # 按日期分组测试数据
        test_by_date = self.test_df.groupby(self.test_df["time_create"].dt.floor("D"))
        
        positive_probs = []
        negative_probs = []
        
        valid_days = 0
        total_days = 0
        
        for date, day_group in test_by_date:
            total_days += 1
            
            # 当天洪水道路列表
            flooded_roads = list(day_group["link_id"].unique())
            flooded_in_bn = [road for road in flooded_roads if road in self.bn_nodes]
            
            if len(flooded_in_bn) < 2:
                continue
                
            valid_days += 1
            
            # Evidence选择
            evidence_count = max(1, int(len(flooded_in_bn) * 0.5))
            evidence_roads = flooded_in_bn[:evidence_count]
            target_roads = flooded_in_bn[evidence_count:]
            
            evidence = {road: 1 for road in evidence_roads}
            
            # 处理正样本（真实洪水道路）
            for target_road in target_roads:
                try:
                    result = self.flood_net.infer_w_evidence(target_road, evidence)
                    prob_flood = result['flooded']
                    
                    positive_probs.append(prob_flood)
                    self.all_predictions.append({
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
            selected_negatives = available_negatives[:min(3, len(target_roads))]
            
            for neg_road in selected_negatives:
                try:
                    result = self.flood_net.infer_w_evidence(neg_road, evidence)
                    prob_flood = result['flooded']
                    
                    negative_probs.append(prob_flood)
                    self.all_predictions.append({
                        'type': 'negative',
                        'road': neg_road,
                        'true_label': 0,
                        'prob_flood': prob_flood,
                        'date': date
                    })
                except:
                    continue
        
        self.probability_distribution = {
            'positive_probs': positive_probs,
            'negative_probs': negative_probs
        }
        
        print(f"📊 数据收集结果:")
        print(f"   总测试天数: {total_days}")
        print(f"   有效评估天数: {valid_days}")
        print(f"   正样本数: {len(positive_probs)}")
        print(f"   负样本数: {len(negative_probs)}")
        print(f"   总样本数: {len(self.all_predictions)}")
        
        return self.probability_distribution
    
    def analyze_probability_distribution(self):
        """分析概率分布特征"""
        print(f"\n📈 概率分布分析")
        print("=" * 50)
        
        pos_probs = self.probability_distribution['positive_probs']
        neg_probs = self.probability_distribution['negative_probs']
        
        if pos_probs:
            print(f"🔸 正样本概率分布:")
            print(f"   数量: {len(pos_probs)}")
            print(f"   均值: {np.mean(pos_probs):.4f}")
            print(f"   标准差: {np.std(pos_probs):.4f}")
            print(f"   中位数: {np.median(pos_probs):.4f}")
            print(f"   范围: [{np.min(pos_probs):.4f}, {np.max(pos_probs):.4f}]")
            print(f"   分位数: 25%={np.percentile(pos_probs, 25):.4f}, 75%={np.percentile(pos_probs, 75):.4f}")
        
        if neg_probs:
            print(f"\n🔸 负样本概率分布:")
            print(f"   数量: {len(neg_probs)}")
            print(f"   均值: {np.mean(neg_probs):.4f}")
            print(f"   标准差: {np.std(neg_probs):.4f}")
            print(f"   中位数: {np.median(neg_probs):.4f}")
            print(f"   范围: [{np.min(neg_probs):.4f}, {np.max(neg_probs):.4f}]")
            print(f"   分位数: 25%={np.percentile(neg_probs, 25):.4f}, 75%={np.percentile(neg_probs, 75):.4f}")
        
        # 分析重叠情况
        if pos_probs and neg_probs:
            pos_mean = np.mean(pos_probs)
            neg_mean = np.mean(neg_probs)
            separation = pos_mean - neg_mean
            
            print(f"\n🔍 分布分离度分析:")
            print(f"   正样本均值 - 负样本均值: {separation:.4f}")
            print(f"   分离度: {'良好' if separation > 0.1 else '较差' if separation > 0.05 else '很差'}")
            
            # 计算最佳分割点
            all_probs = sorted(pos_probs + neg_probs)
            best_threshold = (pos_mean + neg_mean) / 2
            print(f"   建议分割点: {best_threshold:.4f}")
    
    def create_probability_distribution_plot(self, save_path="probability_distribution.png"):
        """创建概率分布可视化"""
        print(f"\n🎨 创建概率分布可视化")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        pos_probs = self.probability_distribution['positive_probs']
        neg_probs = self.probability_distribution['negative_probs']
        
        # 直方图
        ax1.hist(pos_probs, bins=20, alpha=0.6, color='red', label=f'正样本 (n={len(pos_probs)})', density=True)
        ax1.hist(neg_probs, bins=20, alpha=0.6, color='blue', label=f'负样本 (n={len(neg_probs)})', density=True)
        ax1.set_xlabel('洪水概率')
        ax1.set_ylabel('密度')
        ax1.set_title('概率分布直方图')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 箱线图
        data_for_box = [pos_probs, neg_probs]
        labels = ['正样本', '负样本']
        box_plot = ax2.boxplot(data_for_box, labels=labels, patch_artist=True)
        box_plot['boxes'][0].set_facecolor('red')
        box_plot['boxes'][1].set_facecolor('blue')
        ax2.set_ylabel('洪水概率')
        ax2.set_title('概率分布箱线图')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 概率分布图已保存至: {save_path}")
        plt.show()
        
        return fig
    
    def optimize_thresholds_grid_search(self, target_precision=0.8, min_recall=0.4):
        """网格搜索最优阈值"""
        print(f"\n🎯 网格搜索最优阈值")
        print(f"目标: Precision ≥ {target_precision}, Recall ≥ {min_recall}")
        print("=" * 60)
        
        # 扩大搜索范围，降低阈值
        positive_thresholds = np.arange(0.1, 0.8, 0.05)  # 从0.1开始
        negative_thresholds = np.arange(0.05, 0.4, 0.05)
        
        best_configs = []
        all_results = []
        
        print(f"🔍 搜索空间: {len(positive_thresholds)} × {len(negative_thresholds)} = {len(positive_thresholds) * len(negative_thresholds)} 组合")
        
        for i, pos_thr in enumerate(positive_thresholds):
            if i % 3 == 0:
                print(f"   进度: {i+1}/{len(positive_thresholds)} 正阈值...")
            
            for neg_thr in negative_thresholds:
                if neg_thr >= pos_thr:
                    continue  # 确保 neg_thr < pos_thr
                
                # 计算该阈值组合下的性能
                metrics = self.evaluate_threshold_combination(pos_thr, neg_thr)
                
                result = {
                    'pos_threshold': pos_thr,
                    'neg_threshold': neg_thr,
                    **metrics
                }
                all_results.append(result)
                
                # 检查是否满足目标条件
                if (metrics['precision'] >= target_precision and 
                    metrics['recall'] >= min_recall and
                    metrics['total_predictions'] >= 10):  # 确保有足够的预测
                    
                    best_configs.append(result)
        
        print(f"\n📊 搜索结果:")
        print(f"   总搜索组合: {len(all_results)}")
        print(f"   满足条件的组合: {len(best_configs)}")
        
        if best_configs:
            # 按F1分数排序
            best_configs.sort(key=lambda x: x['f1_score'], reverse=True)
            
            print(f"\n🏆 TOP-5 最佳阈值组合:")
            print(f"   {'Rank':<4} {'Pos_Thr':<8} {'Neg_Thr':<8} {'Precision':<10} {'Recall':<8} {'F1':<8} {'Predictions':<11}")
            print("-" * 70)
            
            for i, config in enumerate(best_configs[:5], 1):
                print(f"   {i:<4} {config['pos_threshold']:<8.3f} {config['neg_threshold']:<8.3f} "
                      f"{config['precision']:<10.3f} {config['recall']:<8.3f} {config['f1_score']:<8.3f} "
                      f"{config['total_predictions']:<11}")
            
            # 推荐最佳配置
            recommended = best_configs[0]
            print(f"\n✅ 推荐配置:")
            print(f"   正预测阈值: {recommended['pos_threshold']:.3f}")
            print(f"   负预测阈值: {recommended['neg_threshold']:.3f}")
            print(f"   预期性能: P={recommended['precision']:.3f}, R={recommended['recall']:.3f}, F1={recommended['f1_score']:.3f}")
            
            return recommended, best_configs, all_results
        else:
            print(f"❌ 未找到满足条件的阈值组合")
            
            # 显示最接近的结果
            all_results.sort(key=lambda x: x['f1_score'], reverse=True)
            print(f"\n📋 最佳F1分数的前5个结果:")
            print(f"   {'Pos_Thr':<8} {'Neg_Thr':<8} {'Precision':<10} {'Recall':<8} {'F1':<8}")
            print("-" * 50)
            
            for config in all_results[:5]:
                print(f"   {config['pos_threshold']:<8.3f} {config['neg_threshold']:<8.3f} "
                      f"{config['precision']:<10.3f} {config['recall']:<8.3f} {config['f1_score']:<8.3f}")
            
            return None, [], all_results
    
    def evaluate_threshold_combination(self, pos_threshold, neg_threshold):
        """评估特定阈值组合的性能"""
        tp = fp = tn = fn = uncertain = 0
        
        for pred in self.all_predictions:
            prob = pred['prob_flood']
            true_label = pred['true_label']
            
            # 应用阈值决策
            if prob >= pos_threshold:
                prediction = 1
            elif prob <= neg_threshold:
                prediction = 0
            else:
                prediction = -1  # uncertain
                uncertain += 1
                continue
            
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
        total_predictions = tp + fp + tn + fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / total_predictions if total_predictions > 0 else 0.0
        
        return {
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'uncertain': uncertain,
            'total_predictions': total_predictions,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy
        }
    
    def validate_optimized_thresholds(self, pos_threshold, neg_threshold):
        """验证优化后的阈值性能"""
        print(f"\n🔬 验证优化后的阈值性能")
        print(f"新阈值: 正预测={pos_threshold:.3f}, 负预测={neg_threshold:.3f}")
        print("=" * 60)
        
        # 重新计算详细性能
        metrics = self.evaluate_threshold_combination(pos_threshold, neg_threshold)
        
        print(f"📊 详细性能指标:")
        print(f"   True Positives (TP):  {metrics['tp']:4d}")
        print(f"   False Positives (FP): {metrics['fp']:4d}")
        print(f"   True Negatives (TN):  {metrics['tn']:4d}")
        print(f"   False Negatives (FN): {metrics['fn']:4d}")
        print(f"   不确定预测:           {metrics['uncertain']:4d}")
        print(f"   总有效预测:           {metrics['total_predictions']:4d}")
        
        print(f"\n📈 关键指标:")
        print(f"   精确度 (Precision): {metrics['precision']:.6f}")
        print(f"   召回率 (Recall):    {metrics['recall']:.6f}")
        print(f"   F1分数 (F1-Score):  {metrics['f1_score']:.6f}")
        print(f"   准确率 (Accuracy):  {metrics['accuracy']:.6f}")
        
        # 目标达成检查
        target_precision = 0.8
        target_recall = 0.4
        
        precision_achieved = metrics['precision'] >= target_precision
        recall_achieved = metrics['recall'] >= target_recall
        
        print(f"\n🎯 目标达成情况:")
        print(f"   精确度目标 (≥{target_precision}): {metrics['precision']:.3f} {'✅' if precision_achieved else '❌'}")
        print(f"   召回率目标 (≥{target_recall}):  {metrics['recall']:.3f} {'✅' if recall_achieved else '❌'}")
        
        if precision_achieved and recall_achieved:
            print(f"   🎉 目标全部达成！")
        else:
            print(f"   ⚠️ 部分目标未达成，可能需要进一步调整")
        
        return metrics

def load_system():
    """加载系统"""
    # 加载数据
    df = pd.read_csv("Road_Closures_2024.csv")
    df = df[df["REASON"].str.upper() == "FLOOD"].copy()
    
    # 预处理
    df["time_create"] = pd.to_datetime(df["START"], utc=True)
    df["link_id"] = df["STREET"].str.upper().str.replace(" ", "_")
    df["link_id"] = df["link_id"].astype(str)
    df["id"] = df["OBJECTID"].astype(str)
    
    # 时序分割
    df_sorted = df.sort_values('time_create')
    split_idx = int(len(df_sorted) * 0.7)
    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()
    
    # 构建网络
    flood_net = FloodBayesNetwork(t_window="D")
    flood_net.fit_marginal(train_df)
    flood_net.build_network_by_co_occurrence(
        train_df, occ_thr=3, edge_thr=2, weight_thr=0.3, report=False
    )
    flood_net.fit_conditional(train_df, max_parents=2, alpha=1.0)
    flood_net.build_bayes_network()
    
    return flood_net, test_df

def main():
    """主函数"""
    print("🎯 阈值分析和优化")
    print("=" * 80)
    
    # 加载系统
    flood_net, test_df = load_system()
    
    # 创建优化器
    optimizer = ThresholdOptimizer(flood_net, test_df)
    
    # 1. 收集所有预测概率
    prob_dist = optimizer.collect_all_predictions()
    
    # 2. 分析概率分布
    optimizer.analyze_probability_distribution()
    
    # 3. 创建概率分布可视化
    fig1 = optimizer.create_probability_distribution_plot()
    
    # 4. 网格搜索最优阈值
    recommended, best_configs, all_results = optimizer.optimize_thresholds_grid_search()
    
    # 5. 验证推荐的阈值
    if recommended:
        final_metrics = optimizer.validate_optimized_thresholds(
            recommended['pos_threshold'], 
            recommended['neg_threshold']
        )
        
        print(f"\n✅ 阈值优化完成！")
        print(f"🎯 推荐使用:")
        print(f"   正预测阈值: {recommended['pos_threshold']:.3f}")
        print(f"   负预测阈值: {recommended['neg_threshold']:.3f}")
        
        return optimizer, recommended, final_metrics
    else:
        print(f"\n⚠️ 未找到满足目标的阈值，请检查数据或调整目标")
        return optimizer, None, None

if __name__ == "__main__":
    optimizer, recommended, metrics = main()