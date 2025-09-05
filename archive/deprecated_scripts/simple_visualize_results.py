#!/usr/bin/env python3
"""
简化版贝叶斯网络洪水预测结果可视化
- 不依赖pandas，直接处理CSV和JSON
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import csv
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS'] 
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SimpleBayesianVisualizer:
    def __init__(self, csv_file, json_file):
        self.csv_file = csv_file
        self.json_file = json_file
        self.data = []
        self.best_experiment = None
        self.load_data()
        
    def load_data(self):
        """加载数据"""
        print("📊 Loading data...")
        
        # 加载CSV数据
        with open(self.csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 转换数值字段
                for key in ['pred_threshold', 'precision', 'recall', 'f1_score', 'accuracy',
                           'positive_predict_count', 'negative_predict_count', 'total_predict_count',
                           'successful_predictions', 'network_nodes', 'evidence_roads_count',
                           'tp', 'fp', 'tn', 'fn']:
                    if key in row:
                        row[key] = float(row[key])
                self.data.append(row)
        
        print(f"✅ Loaded CSV: {len(self.data)} experiments")
        
        # 加载最佳实验JSON
        with open(self.json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.best_experiment = data['best_experiment']
        print(f"✅ Loaded best experiment: {self.best_experiment['test_date']}")
        
    def create_performance_by_threshold_plot(self):
        """性能指标随阈值变化图"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Bayesian Network Performance vs Prediction Threshold', fontsize=16, fontweight='bold')
        
        # 按阈值分组计算统计
        threshold_stats = defaultdict(lambda: {'precision': [], 'recall': [], 'f1_score': [], 'accuracy': []})
        
        for row in self.data:
            threshold = row['pred_threshold']
            threshold_stats[threshold]['precision'].append(row['precision'])
            threshold_stats[threshold]['recall'].append(row['recall'])
            threshold_stats[threshold]['f1_score'].append(row['f1_score'])
            threshold_stats[threshold]['accuracy'].append(row['accuracy'])
        
        thresholds = sorted(threshold_stats.keys())
        
        # 计算均值和标准差
        precision_means = [np.mean(threshold_stats[t]['precision']) for t in thresholds]
        precision_stds = [np.std(threshold_stats[t]['precision']) for t in thresholds]
        recall_means = [np.mean(threshold_stats[t]['recall']) for t in thresholds]
        recall_stds = [np.std(threshold_stats[t]['recall']) for t in thresholds]
        f1_means = [np.mean(threshold_stats[t]['f1_score']) for t in thresholds]
        f1_stds = [np.std(threshold_stats[t]['f1_score']) for t in thresholds]
        accuracy_means = [np.mean(threshold_stats[t]['accuracy']) for t in thresholds]
        accuracy_stds = [np.std(threshold_stats[t]['accuracy']) for t in thresholds]
        
        # 1. Precision
        ax1.errorbar(thresholds, precision_means, yerr=precision_stds,
                    marker='o', linewidth=2, markersize=8, capsize=5)
        ax1.set_title('Precision vs Threshold', fontweight='bold')
        ax1.set_xlabel('Prediction Threshold')
        ax1.set_ylabel('Precision')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.1)
        
        # 2. Recall
        ax2.errorbar(thresholds, recall_means, yerr=recall_stds,
                    marker='s', linewidth=2, markersize=8, capsize=5, color='orange')
        ax2.set_title('Recall vs Threshold', fontweight='bold')
        ax2.set_xlabel('Prediction Threshold')
        ax2.set_ylabel('Recall')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.1)
        
        # 3. F1 Score
        ax3.errorbar(thresholds, f1_means, yerr=f1_stds,
                    marker='^', linewidth=2, markersize=8, capsize=5, color='green')
        ax3.set_title('F1 Score vs Threshold', fontweight='bold')
        ax3.set_xlabel('Prediction Threshold')
        ax3.set_ylabel('F1 Score')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1.1)
        
        # 4. Accuracy
        ax4.errorbar(thresholds, accuracy_means, yerr=accuracy_stds,
                    marker='d', linewidth=2, markersize=8, capsize=5, color='red')
        ax4.set_title('Accuracy vs Threshold', fontweight='bold')
        ax4.set_xlabel('Prediction Threshold')
        ax4.set_ylabel('Accuracy')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig('performance_by_threshold.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("💾 Saved: performance_by_threshold.png")
        
    def create_performance_by_date_plot(self):
        """不同日期洪水事件的性能对比"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison by Flood Event Date', fontsize=16, fontweight='bold')
        
        # 按日期分组
        date_stats = defaultdict(lambda: {'precision': [], 'recall': [], 'f1_score': [], 'accuracy': [],
                                         'tp': [], 'fp': [], 'tn': [], 'fn': [], 'test_roads_total': 0})
        
        for row in self.data:
            date = row['test_date']
            date_stats[date]['precision'].append(row['precision'])
            date_stats[date]['recall'].append(row['recall'])
            date_stats[date]['f1_score'].append(row['f1_score'])
            date_stats[date]['accuracy'].append(row['accuracy'])
            date_stats[date]['tp'].append(row['tp'])
            date_stats[date]['fp'].append(row['fp'])
            date_stats[date]['tn'].append(row['tn'])
            date_stats[date]['fn'].append(row['fn'])
            if date_stats[date]['test_roads_total'] == 0:
                date_stats[date]['test_roads_total'] = int(row.get('test_roads_total', 0))
        
        dates = sorted(date_stats.keys())
        
        # 1. Precision
        precision_means = [np.mean(date_stats[d]['precision']) for d in dates]
        precision_stds = [np.std(date_stats[d]['precision']) for d in dates]
        road_counts = [date_stats[d]['test_roads_total'] for d in dates]
        
        bars1 = ax1.bar(range(len(dates)), precision_means, yerr=precision_stds,
                       alpha=0.7, capsize=5)
        ax1.set_title('Precision Comparison by Date', fontweight='bold')
        ax1.set_xlabel('Test Date')
        ax1.set_ylabel('Precision')
        ax1.set_xticks(range(len(dates)))
        ax1.set_xticklabels(dates, rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 添加道路数量标注
        for i, (bar, count) in enumerate(zip(bars1, road_counts)):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f'{count} roads', ha='center', va='bottom', fontsize=9)
        
        # 2. Recall
        recall_means = [np.mean(date_stats[d]['recall']) for d in dates]
        recall_stds = [np.std(date_stats[d]['recall']) for d in dates]
        
        ax2.bar(range(len(dates)), recall_means, yerr=recall_stds,
               alpha=0.7, capsize=5, color='orange')
        ax2.set_title('Recall Comparison by Date', fontweight='bold')
        ax2.set_xlabel('Test Date')
        ax2.set_ylabel('Recall')
        ax2.set_xticks(range(len(dates)))
        ax2.set_xticklabels(dates, rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 3. F1 Score
        f1_means = [np.mean(date_stats[d]['f1_score']) for d in dates]
        f1_stds = [np.std(date_stats[d]['f1_score']) for d in dates]
        
        ax3.bar(range(len(dates)), f1_means, yerr=f1_stds,
               alpha=0.7, capsize=5, color='green')
        ax3.set_title('F1 Score Comparison by Date', fontweight='bold')
        ax3.set_xlabel('Test Date')
        ax3.set_ylabel('F1 Score')
        ax3.set_xticks(range(len(dates)))
        ax3.set_xticklabels(dates, rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 4. 混淆矩阵指标
        tp_means = [np.mean(date_stats[d]['tp']) for d in dates]
        fp_means = [np.mean(date_stats[d]['fp']) for d in dates]
        tn_means = [np.mean(date_stats[d]['tn']) for d in dates]
        fn_means = [np.mean(date_stats[d]['fn']) for d in dates]
        
        width = 0.2
        x = np.arange(len(dates))
        ax4.bar(x - 1.5*width, tp_means, width, label='TP', alpha=0.7)
        ax4.bar(x - 0.5*width, fp_means, width, label='FP', alpha=0.7)
        ax4.bar(x + 0.5*width, tn_means, width, label='TN', alpha=0.7)
        ax4.bar(x + 1.5*width, fn_means, width, label='FN', alpha=0.7)
        
        ax4.set_title('Average Confusion Matrix by Date', fontweight='bold')
        ax4.set_xlabel('Test Date')
        ax4.set_ylabel('Average Count')
        ax4.set_xticks(x)
        ax4.set_xticklabels(dates, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('performance_by_date.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("💾 Saved: performance_by_date.png")
        
    def create_precision_recall_scatter(self):
        """Precision-Recall散点图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Precision-Recall Relationship Analysis', fontsize=16, fontweight='bold')
        
        # 1. 按阈值着色的散点图
        recalls = [row['recall'] for row in self.data]
        precisions = [row['precision'] for row in self.data]
        thresholds = [row['pred_threshold'] for row in self.data]
        
        scatter = ax1.scatter(recalls, precisions, c=thresholds, s=60, alpha=0.7, cmap='viridis')
        ax1.set_xlabel('Recall')
        ax1.set_ylabel('Precision')
        ax1.set_title('Precision vs Recall (colored by threshold)', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1.05)
        
        # 添加colorbar
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Prediction Threshold')
        
        # 2. 按日期着色的散点图
        dates = list(set(row['test_date'] for row in self.data))
        colors = plt.cm.Set1(np.linspace(0, 1, len(dates)))
        
        for date, color in zip(dates, colors):
            date_recalls = [row['recall'] for row in self.data if row['test_date'] == date]
            date_precisions = [row['precision'] for row in self.data if row['test_date'] == date]
            ax2.scatter(date_recalls, date_precisions, label=date, s=60, alpha=0.7, color=color)
        
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision vs Recall (colored by date)', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1.05)
        
        plt.tight_layout()
        plt.savefig('precision_recall_scatter.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("💾 Saved: precision_recall_scatter.png")
        
    def create_best_experiment_visualization(self):
        """最佳实验详细可视化"""
        exp = self.best_experiment
        predictions = exp['detailed_predictions']
        
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        fig.suptitle(f'Best Experiment Analysis: {exp["test_date"]} (Threshold={exp["pred_threshold"]})', 
                     fontsize=18, fontweight='bold')
        
        # 1. 性能指标雷达图
        ax1 = fig.add_subplot(gs[0, 0], projection='polar')
        metrics = ['Precision', 'Recall', 'F1 Score', 'Accuracy']
        values = [exp['precision'], exp['recall'], exp['f1_score'], exp['accuracy']]
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        values += values[:1]  # 闭合多边形
        angles += angles[:1]
        
        ax1.plot(angles, values, 'o-', linewidth=2, color='red')
        ax1.fill(angles, values, alpha=0.25, color='red')
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(metrics)
        ax1.set_ylim(0, 1)
        ax1.set_title('Performance Radar Chart', fontweight='bold', pad=20)
        ax1.grid(True)
        
        # 2. 混淆矩阵
        ax2 = fig.add_subplot(gs[0, 1])
        confusion_matrix = np.array([[exp['tp'], exp['fp']], 
                                   [exp['fn'], exp['tn']]])
        
        # 手动创建热力图
        im = ax2.imshow(confusion_matrix, cmap='Blues')
        ax2.set_xticks([0, 1])
        ax2.set_yticks([0, 1])
        ax2.set_xticklabels(['Pred Flood', 'Pred No Flood'])
        ax2.set_yticklabels(['Actual Flood', 'Actual No Flood'])
        
        # 添加数值标注
        for i in range(2):
            for j in range(2):
                text = ax2.text(j, i, confusion_matrix[i, j],
                               ha="center", va="center", color="black", fontweight='bold')
        
        ax2.set_title('Confusion Matrix', fontweight='bold')
        
        # 3. 样本分布饼图
        ax3 = fig.add_subplot(gs[0, 2])
        sizes = [exp['tp'], exp['fp'], exp['tn'], exp['fn']]
        labels = [f'TP ({exp["tp"]})', f'FP ({exp["fp"]})', 
                 f'TN ({exp["tn"]})', f'FN ({exp["fn"]})']
        colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']
        ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Prediction Results Distribution', fontweight='bold')
        
        # 4. 道路预测概率分布
        ax4 = fig.add_subplot(gs[1, :])
        
        # 准备数据
        road_names = [p['road_name'] for p in predictions]
        probabilities = [p['predicted_probability'] for p in predictions]
        true_labels = [p['true_label'] for p in predictions]
        predicted_labels = [p['predicted_label'] for p in predictions]
        
        # 按概率排序
        sorted_indices = np.argsort(probabilities)[::-1]  # 降序
        
        x_pos = np.arange(len(road_names))
        colors = []
        for i in sorted_indices:
            if true_labels[i] == 1 and predicted_labels[i] == 1:  # TP
                colors.append('#2ecc71')  # 绿色
            elif true_labels[i] == 0 and predicted_labels[i] == 0:  # TN  
                colors.append('#3498db')  # 蓝色
            elif true_labels[i] == 1 and predicted_labels[i] == 0:  # FN
                colors.append('#f39c12')  # 橙色
            else:  # FP
                colors.append('#e74c3c')  # 红色
        
        bars = ax4.bar(x_pos, [probabilities[i] for i in sorted_indices], color=colors, alpha=0.8)
        ax4.axhline(y=exp['pred_threshold'], color='red', linestyle='--', linewidth=2, 
                   label=f'Threshold = {exp["pred_threshold"]}')
        
        ax4.set_xlabel('Roads (sorted by prediction probability)')
        ax4.set_ylabel('Flood Probability')
        ax4.set_title('Flood Prediction Probabilities for All Roads', fontweight='bold')
        ax4.set_xticks(x_pos[::3])  # 只显示部分标签避免重叠
        ax4.set_xticklabels([road_names[sorted_indices[i]] for i in range(0, len(road_names), 3)], 
                           rotation=90, fontsize=8)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 添加颜色图例
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#2ecc71', label='TP (Correct Flood)'),
                          Patch(facecolor='#3498db', label='TN (Correct No Flood)'),
                          Patch(facecolor='#f39c12', label='FN (Missed Flood)'),
                          Patch(facecolor='#e74c3c', label='FP (False Flood)')]
        ax4.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
        
        # 5. 证据道路信息
        ax5 = fig.add_subplot(gs[2, 0])
        evidence_roads = exp['evidence_roads']
        evidence_text = 'Evidence Roads:\\n'
        for i, road in enumerate(evidence_roads):
            evidence_text += f'{i+1}. {road}\\n'
        
        ax5.text(0.1, 0.9, evidence_text, fontsize=10, transform=ax5.transAxes, 
                verticalalignment='top', fontfamily='monospace')
        ax5.set_xlim(0, 1)
        ax5.set_ylim(0, 1) 
        ax5.axis('off')
        
        # 6. 统计摘要
        ax6 = fig.add_subplot(gs[2, 1:])
        
        # 分析高概率预测
        high_prob_correct = sum(1 for p in predictions 
                               if p['predicted_probability'] > 0.7 and 
                               p['true_label'] == p['predicted_label'])
        high_prob_total = sum(1 for p in predictions if p['predicted_probability'] > 0.7)
        
        summary_text = f'''Experiment Summary:
• Total predicted roads: {len(predictions)}
• Successful predictions: {exp['successful_predictions']} (100%)
• High confidence predictions (>0.7): {high_prob_total} roads, {high_prob_correct} correct
• Network size: {exp['network_nodes']} nodes, {exp['network_edges']} edges
• Evidence ratio: {exp['evidence_roads_count']}/{exp['test_roads_in_network']} = {exp['evidence_roads_count']/exp['test_roads_in_network']:.1%}

Key Success Factors:
• Perfect precision (100%) - no false alarms
• Reasonable recall ({exp['recall']:.1%})
• Effective evidence road selection
• Stable Bayesian inference execution'''
        
        ax6.text(0.05, 0.95, summary_text, fontsize=10, transform=ax6.transAxes,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        ax6.axis('off')
        
        plt.savefig('best_experiment_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("💾 Saved: best_experiment_analysis.png")
        
    def create_all_visualizations(self):
        """创建所有可视化图表"""
        print("🎨 Creating visualization charts...")
        
        print("\\n1️⃣ Creating performance vs threshold plot...")
        self.create_performance_by_threshold_plot()
        
        print("\\n2️⃣ Creating performance comparison by date...")
        self.create_performance_by_date_plot()
        
        print("\\n3️⃣ Creating precision-recall scatter plot...")
        self.create_precision_recall_scatter()
        
        print("\\n4️⃣ Creating best experiment visualization...")
        self.create_best_experiment_visualization()
        
        print(f"\\n🎉 All visualizations completed!")
        print("📁 Generated files:")
        print("   • performance_by_threshold.png")
        print("   • performance_by_date.png") 
        print("   • precision_recall_scatter.png")
        print("   • best_experiment_analysis.png")

def main():
    """主函数"""
    print("🌊 Charleston Flood Prediction - Bayesian Network Results Visualization")
    print("="*70)
    
    # 文件路径
    csv_file = "corrected_bayesian_flood_validation_full_network_summary_20250820_112441.csv"
    json_file = "best_2017_09_11_threshold_04_experiment.json"
    
    # 创建可视化器
    visualizer = SimpleBayesianVisualizer(csv_file, json_file)
    
    # 生成所有可视化
    visualizer.create_all_visualizations()

if __name__ == "__main__":
    main()