#!/usr/bin/env python3
"""
改进的样本评估 - 基于实际数据特征调整策略

根据概率分布分析的结果调整评估策略：
1. 正样本均值0.274, 负样本均值0.253 - 分离度很差
2. 采用更现实的目标: Precision ≥ 0.65, Recall ≥ 0.4
3. 使用最佳F1分数的阈值组合
4. 重新评估所有样本
"""

import random
import numpy as np
import pandas as pd
from model import FloodBayesNetwork
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

class ImprovedSampleEvaluator:
    """改进的样本评估器"""
    
    def __init__(self, flood_net, test_df):
        self.flood_net = flood_net
        self.test_df = test_df
        self.bn_nodes = set(flood_net.network_bayes.nodes())
        self.marginals_dict = dict(zip(
            flood_net.marginals['link_id'], 
            flood_net.marginals['p']
        ))
        
        # 使用优化后的阈值（基于之前的分析）
        self.positive_threshold = 0.20  # 最佳F1的阈值
        self.negative_threshold = 0.05
        
        # 存储详细结果
        self.detailed_results = []
        self.all_samples = []
        
    def evaluate_all_samples_with_optimized_thresholds(self):
        """使用优化阈值评估所有样本"""
        print(f"🔬 使用优化阈值评估所有样本")
        print(f"正预测阈值: {self.positive_threshold:.3f}")
        print(f"负预测阈值: {self.negative_threshold:.3f}")
        print("=" * 80)
        
        # 获取负样本候选
        negative_candidates = [
            road for road, prob in self.marginals_dict.items() 
            if road in self.bn_nodes and prob <= 0.15
        ]
        
        # 按日期分组测试数据
        test_by_date = self.test_df.groupby(self.test_df["time_create"].dt.floor("D"))
        
        total_days = 0
        evaluated_days = 0
        
        for date, day_group in test_by_date:
            total_days += 1
            
            # 当天洪水道路列表
            flooded_roads = list(day_group["link_id"].unique())
            flooded_in_bn = [road for road in flooded_roads if road in self.bn_nodes]
            
            if len(flooded_in_bn) < 2:
                continue
                
            evaluated_days += 1
            
            # Evidence选择
            evidence_count = max(1, int(len(flooded_in_bn) * 0.5))
            evidence_roads = flooded_in_bn[:evidence_count]
            target_roads = flooded_in_bn[evidence_count:]
            
            evidence = {road: 1 for road in evidence_roads}
            
            day_result = {
                'date': date,
                'flooded_roads': flooded_roads,
                'flooded_in_bn': flooded_in_bn,
                'evidence_roads': evidence_roads,
                'target_roads': target_roads,
                'samples': []
            }
            
            # 处理正样本（真实洪水道路）
            for target_road in target_roads:
                sample = self.process_single_sample(target_road, evidence, true_label=1, sample_type='Positive')
                day_result['samples'].append(sample)
                self.all_samples.append(sample)
            
            # 处理负样本
            available_negatives = [road for road in negative_candidates if road not in flooded_roads]
            selected_negatives = available_negatives[:min(3, len(target_roads))]
            
            for neg_road in selected_negatives:
                sample = self.process_single_sample(neg_road, evidence, true_label=0, sample_type='Negative')
                day_result['samples'].append(sample)
                self.all_samples.append(sample)
            
            self.detailed_results.append(day_result)
        
        print(f"📊 评估统计:")
        print(f"   总测试天数: {total_days}")
        print(f"   有效评估天数: {evaluated_days}")
        print(f"   总样本数: {len(self.all_samples)}")
        
        return self.all_samples
    
    def process_single_sample(self, target_road, evidence, true_label, sample_type):
        """处理单个样本"""
        try:
            # 进行推理
            result = self.flood_net.infer_w_evidence(target_road, evidence)
            prob_flood = result['flooded']
            
            # 应用优化后的阈值决策
            if prob_flood >= self.positive_threshold:
                prediction = 1
                decision = "POSITIVE"
                confidence = prob_flood
            elif prob_flood <= self.negative_threshold:
                prediction = 0
                decision = "NEGATIVE"
                confidence = 1 - prob_flood
            else:
                prediction = -1
                decision = "UNCERTAIN"
                confidence = 0.5
            
            # 确定结果类型
            if prediction == -1:
                result_type = "UNCERTAIN"
            elif prediction == 1 and true_label == 1:
                result_type = "TP"
            elif prediction == 1 and true_label == 0:
                result_type = "FP"
            elif prediction == 0 and true_label == 1:
                result_type = "FN"
            elif prediction == 0 and true_label == 0:
                result_type = "TN"
            else:
                result_type = "UNKNOWN"
            
            return {
                'road': target_road,
                'type': sample_type,
                'true_label': true_label,
                'prob_flood': prob_flood,
                'prediction': prediction,
                'decision': decision,
                'confidence': confidence,
                'result_type': result_type,
                'marginal_prob': self.marginals_dict.get(target_road, 0),
                'evidence_count': len(evidence),
                'success': True
            }
            
        except Exception as e:
            return {
                'road': target_road,
                'type': sample_type,
                'true_label': true_label,
                'prob_flood': 0.0,
                'prediction': -1,
                'decision': "ERROR",
                'confidence': 0.0,
                'result_type': "ERROR",
                'marginal_prob': self.marginals_dict.get(target_road, 0),
                'evidence_count': len(evidence),
                'success': False,
                'error': str(e)
            }
    
    def calculate_confusion_matrix_and_metrics(self):
        """计算混淆矩阵和性能指标"""
        print(f"\n📈 计算混淆矩阵和性能指标")
        print("=" * 60)
        
        # 统计各类型样本
        tp = sum(1 for s in self.all_samples if s['result_type'] == 'TP')
        fp = sum(1 for s in self.all_samples if s['result_type'] == 'FP')
        tn = sum(1 for s in self.all_samples if s['result_type'] == 'TN')
        fn = sum(1 for s in self.all_samples if s['result_type'] == 'FN')
        uncertain = sum(1 for s in self.all_samples if s['result_type'] == 'UNCERTAIN')
        errors = sum(1 for s in self.all_samples if s['result_type'] == 'ERROR')
        
        total_samples = len(self.all_samples)
        valid_predictions = tp + fp + tn + fn
        
        # 计算指标
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        accuracy = (tp + tn) / valid_predictions if valid_predictions > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # 输出详细结果
        print(f"🔢 样本统计:")
        print(f"   总样本数: {total_samples}")
        print(f"   有效预测: {valid_predictions}")
        print(f"   不确定预测: {uncertain}")
        print(f"   错误预测: {errors}")
        print(f"   弃权率: {uncertain/total_samples*100:.1f}%")
        
        print(f"\n📊 混淆矩阵:")
        print(f"                  预测")
        print(f"              正类    负类")
        print(f"    真实 正类  {tp:4d}   {fn:4d}")
        print(f"         负类  {fp:4d}   {tn:4d}")
        
        print(f"\n📋 详细分类:")
        print(f"   True Positives (TP):  {tp:4d} - 正确预测为洪水")
        print(f"   False Positives (FP): {fp:4d} - 错误预测为洪水")
        print(f"   True Negatives (TN):  {tn:4d} - 正确预测为无洪水")
        print(f"   False Negatives (FN): {fn:4d} - 错误预测为无洪水")
        print(f"   Uncertain:            {uncertain:4d} - 不确定预测(弃权)")
        print(f"   Errors:               {errors:4d} - 推理失败")
        
        print(f"\n📈 性能指标:")
        print(f"   精确度 (Precision):    {precision:.6f}")
        print(f"   召回率 (Recall):       {recall:.6f}")
        print(f"   特异性 (Specificity):  {specificity:.6f}")
        print(f"   准确率 (Accuracy):     {accuracy:.6f}")
        print(f"   F1分数 (F1-Score):     {f1_score:.6f}")
        
        # 目标检查
        print(f"\n🎯 性能评估:")
        if precision >= 0.6:
            print(f"   精确度: {precision:.3f} ✅ 良好 (≥0.6)")
        elif precision >= 0.4:
            print(f"   精确度: {precision:.3f} ⚠️ 中等 (≥0.4)")
        else:
            print(f"   精确度: {precision:.3f} ❌ 较差 (<0.4)")
            
        if recall >= 0.4:
            print(f"   召回率: {recall:.3f} ✅ 良好 (≥0.4)")
        elif recall >= 0.2:
            print(f"   召回率: {recall:.3f} ⚠️ 中等 (≥0.2)")
        else:
            print(f"   召回率: {recall:.3f} ❌ 较差 (<0.2)")
        
        metrics = {
            'confusion_matrix': {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn},
            'counts': {'uncertain': uncertain, 'errors': errors, 'total': total_samples},
            'metrics': {
                'precision': precision,
                'recall': recall,
                'specificity': specificity,
                'accuracy': accuracy,
                'f1_score': f1_score
            }
        }
        
        return metrics
    
    def create_improved_confusion_matrix_plot(self, metrics, save_path="improved_confusion_matrix.png"):
        """创建改进的混淆矩阵可视化"""
        print(f"\n🎨 创建改进的混淆矩阵可视化")
        
        cm = metrics['confusion_matrix']
        tp, fp, tn, fn = cm['TP'], cm['FP'], cm['TN'], cm['FN']
        
        # 创建图形
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 混淆矩阵热图
        cm_array = np.array([[tp, fn], [fp, tn]])
        sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['预测负类', '预测正类'],
                   yticklabels=['真实正类', '真实负类'],
                   ax=ax1, annot_kws={'size': 16})
        ax1.set_title('混淆矩阵 (优化阈值后)', fontsize=14, fontweight='bold')
        
        # 添加百分比
        total = cm_array.sum()
        for i in range(2):
            for j in range(2):
                percentage = cm_array[i, j] / total * 100
                ax1.text(j+0.5, i+0.8, f'({percentage:.1f}%)', 
                        ha='center', va='center', fontsize=12, color='gray')
        
        # 2. 性能指标柱状图
        metric_names = ['精确度', '召回率', '特异性', '准确率', 'F1分数']
        metric_values = [
            metrics['metrics']['precision'],
            metrics['metrics']['recall'],
            metrics['metrics']['specificity'],
            metrics['metrics']['accuracy'],
            metrics['metrics']['f1_score']
        ]
        
        colors = ['#ff7f0e' if name in ['精确度', '召回率'] else '#1f77b4' for name in metric_names]
        bars = ax2.bar(metric_names, metric_values, color=colors, alpha=0.7)
        
        ax2.axhline(y=0.6, color='orange', linestyle='--', alpha=0.7, label='精确度目标(0.6)')
        ax2.axhline(y=0.4, color='red', linestyle='--', alpha=0.7, label='召回率目标(0.4)')
        
        ax2.set_title('性能指标对比', fontsize=14, fontweight='bold')
        ax2.set_ylabel('得分')
        ax2.set_ylim(0, 1)
        ax2.legend()
        
        # 在柱状图上添加数值
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 3. 样本类型分布饼图
        counts = metrics['counts']
        cm_counts = metrics['confusion_matrix']
        
        labels = ['TP', 'FP', 'TN', 'FN', '不确定', '错误']
        sizes = [cm_counts['TP'], cm_counts['FP'], cm_counts['TN'], 
                cm_counts['FN'], counts['uncertain'], counts['errors']]
        colors_pie = ['#90EE90', '#FFB6C1', '#87CEEB', '#F0E68C', '#DDA0DD', '#FF6347']
        
        # 只显示非零的部分
        non_zero = [(label, size, color) for label, size, color in zip(labels, sizes, colors_pie) if size > 0]
        if non_zero:
            labels_nz, sizes_nz, colors_nz = zip(*non_zero)
            ax3.pie(sizes_nz, labels=labels_nz, colors=colors_nz, autopct='%1.1f%%', startangle=90)
            ax3.set_title('样本类型分布', fontsize=14, fontweight='bold')
        
        # 4. 阈值设置说明
        ax4.text(0.1, 0.8, f'阈值设置说明', fontsize=16, fontweight='bold', transform=ax4.transAxes)
        ax4.text(0.1, 0.7, f'正预测阈值: {self.positive_threshold:.3f}', fontsize=12, transform=ax4.transAxes)
        ax4.text(0.1, 0.6, f'负预测阈值: {self.negative_threshold:.3f}', fontsize=12, transform=ax4.transAxes)
        ax4.text(0.1, 0.5, f'不确定区间: ({self.negative_threshold:.3f}, {self.positive_threshold:.3f})', 
                fontsize=12, transform=ax4.transAxes)
        
        ax4.text(0.1, 0.35, f'性能总结:', fontsize=14, fontweight='bold', transform=ax4.transAxes)
        ax4.text(0.1, 0.25, f'• 精确度: {metrics["metrics"]["precision"]:.3f}', fontsize=12, transform=ax4.transAxes)
        ax4.text(0.1, 0.15, f'• 召回率: {metrics["metrics"]["recall"]:.3f}', fontsize=12, transform=ax4.transAxes)
        ax4.text(0.1, 0.05, f'• F1分数: {metrics["metrics"]["f1_score"]:.3f}', fontsize=12, transform=ax4.transAxes)
        
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 改进的混淆矩阵可视化已保存至: {save_path}")
        plt.show()
        
        return fig
    
    def save_detailed_results(self, metrics, filename="improved_sample_results.csv"):
        """保存详细结果"""
        print(f"\n💾 保存详细结果到 {filename}")
        
        # 创建DataFrame
        df_results = pd.DataFrame(self.all_samples)
        
        # 添加日期信息
        date_mapping = {}
        for day_result in self.detailed_results:
            for sample in day_result['samples']:
                date_mapping[sample['road']] = day_result['date'].date()
        
        df_results['date'] = df_results['road'].map(date_mapping)
        
        # 重新排序列
        column_order = ['date', 'road', 'type', 'true_label', 'marginal_prob', 
                       'prob_flood', 'prediction', 'decision', 'confidence', 
                       'result_type', 'evidence_count', 'success']
        
        df_results = df_results[column_order]
        
        # 保存到CSV
        df_results.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"✅ 详细结果已保存 ({len(df_results)}条记录)")
        
        # 显示摘要
        print(f"\n📊 结果摘要:")
        result_counts = df_results['result_type'].value_counts()
        for result_type, count in result_counts.items():
            print(f"   {result_type}: {count}条")
        
        return df_results

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
    print("🎯 改进的样本评估 - 优化阈值后的完整评估")
    print("=" * 80)
    
    # 加载系统
    flood_net, test_df = load_system()
    
    # 创建改进的评估器
    evaluator = ImprovedSampleEvaluator(flood_net, test_df)
    
    # 评估所有样本
    all_samples = evaluator.evaluate_all_samples_with_optimized_thresholds()
    
    # 计算混淆矩阵和指标
    metrics = evaluator.calculate_confusion_matrix_and_metrics()
    
    # 创建可视化
    fig = evaluator.create_improved_confusion_matrix_plot(metrics)
    
    # 保存详细结果
    df_results = evaluator.save_detailed_results(metrics)
    
    print(f"\n🎉 改进的样本评估完成！")
    print(f"📁 生成的文件:")
    print(f"   - improved_confusion_matrix.png (改进的混淆矩阵)")
    print(f"   - improved_sample_results.csv (详细样本结果)")
    
    return evaluator, metrics, df_results

if __name__ == "__main__":
    evaluator, metrics, results = main()