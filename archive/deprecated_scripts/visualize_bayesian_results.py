#!/usr/bin/env python3
"""
贝叶斯网络洪水预测结果可视化
- 交叉验证性能分析
- 最佳实验详细可视化
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class BayesianResultsVisualizer:
    def __init__(self, csv_file, json_file):
        self.csv_file = csv_file
        self.json_file = json_file
        self.df = None
        self.best_experiment = None
        self.load_data()
        
    def load_data(self):
        """加载数据"""
        print("📊 加载数据...")
        
        # 加载CSV交叉验证结果
        self.df = pd.read_csv(self.csv_file)
        print(f"✅ 加载CSV: {len(self.df)} 条实验记录")
        
        # 加载最佳实验JSON
        with open(self.json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.best_experiment = data['best_experiment']
        print(f"✅ 加载最佳实验: {self.best_experiment['test_date']}")
        
    def create_performance_by_threshold_plot(self):
        """性能指标随阈值变化图"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('🎯 贝叶斯网络性能指标随预测阈值变化', fontsize=16, fontweight='bold')
        
        # 按阈值分组计算统计
        threshold_stats = self.df.groupby('pred_threshold').agg({
            'precision': ['mean', 'std'],
            'recall': ['mean', 'std'], 
            'f1_score': ['mean', 'std'],
            'accuracy': ['mean', 'std']
        }).round(4)
        
        thresholds = threshold_stats.index
        
        # 1. Precision
        ax1.errorbar(thresholds, threshold_stats['precision']['mean'], 
                    yerr=threshold_stats['precision']['std'], 
                    marker='o', linewidth=2, markersize=8, capsize=5)
        ax1.set_title('Precision vs 阈值', fontweight='bold')
        ax1.set_xlabel('预测阈值')
        ax1.set_ylabel('Precision')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.1)
        
        # 2. Recall  
        ax2.errorbar(thresholds, threshold_stats['recall']['mean'],
                    yerr=threshold_stats['recall']['std'],
                    marker='s', linewidth=2, markersize=8, capsize=5, color='orange')
        ax2.set_title('Recall vs 阈值', fontweight='bold')
        ax2.set_xlabel('预测阈值') 
        ax2.set_ylabel('Recall')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.1)
        
        # 3. F1 Score
        ax3.errorbar(thresholds, threshold_stats['f1_score']['mean'],
                    yerr=threshold_stats['f1_score']['std'], 
                    marker='^', linewidth=2, markersize=8, capsize=5, color='green')
        ax3.set_title('F1 Score vs 阈值', fontweight='bold')
        ax3.set_xlabel('预测阈值')
        ax3.set_ylabel('F1 Score') 
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1.1)
        
        # 4. Accuracy
        ax4.errorbar(thresholds, threshold_stats['accuracy']['mean'],
                    yerr=threshold_stats['accuracy']['std'],
                    marker='d', linewidth=2, markersize=8, capsize=5, color='red')
        ax4.set_title('Accuracy vs 阈值', fontweight='bold')
        ax4.set_xlabel('预测阈值')
        ax4.set_ylabel('Accuracy')
        ax4.grid(True, alpha=0.3) 
        ax4.set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig('performance_by_threshold.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("💾 保存: performance_by_threshold.png")
        
    def create_performance_by_date_plot(self):
        """不同日期洪水事件的性能对比"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('📅 不同洪水事件日期的模型性能对比', fontsize=16, fontweight='bold')
        
        # 按日期分组
        date_stats = self.df.groupby('test_date').agg({
            'precision': ['mean', 'std'],
            'recall': ['mean', 'std'],
            'f1_score': ['mean', 'std'],
            'accuracy': ['mean', 'std'],
            'test_roads_total': 'first'
        }).round(4)
        
        dates = date_stats.index
        road_counts = date_stats['test_roads_total']['first']
        
        # 创建带道路数量标注的图表
        x_pos = np.arange(len(dates))
        
        # 1. Precision
        bars1 = ax1.bar(x_pos, date_stats['precision']['mean'], 
                       yerr=date_stats['precision']['std'],
                       alpha=0.7, capsize=5)
        ax1.set_title('各日期 Precision 对比', fontweight='bold')
        ax1.set_xlabel('测试日期')
        ax1.set_ylabel('Precision')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(dates, rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 添加道路数量标注
        for i, (bar, count) in enumerate(zip(bars1, road_counts)):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f'{count}条道路', ha='center', va='bottom', fontsize=9)
        
        # 2. Recall
        bars2 = ax2.bar(x_pos, date_stats['recall']['mean'],
                       yerr=date_stats['recall']['std'], 
                       alpha=0.7, capsize=5, color='orange')
        ax2.set_title('各日期 Recall 对比', fontweight='bold')
        ax2.set_xlabel('测试日期')
        ax2.set_ylabel('Recall')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(dates, rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 3. F1 Score
        bars3 = ax3.bar(x_pos, date_stats['f1_score']['mean'],
                       yerr=date_stats['f1_score']['std'],
                       alpha=0.7, capsize=5, color='green')
        ax3.set_title('各日期 F1 Score 对比', fontweight='bold')
        ax3.set_xlabel('测试日期')
        ax3.set_ylabel('F1 Score')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(dates, rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 4. 混淆矩阵指标
        confusion_stats = self.df.groupby('test_date').agg({
            'tp': 'mean', 'fp': 'mean', 'tn': 'mean', 'fn': 'mean'
        }).round(1)
        
        width = 0.2
        x_pos = np.arange(len(dates))
        ax4.bar(x_pos - 1.5*width, confusion_stats['tp'], width, label='TP', alpha=0.7)
        ax4.bar(x_pos - 0.5*width, confusion_stats['fp'], width, label='FP', alpha=0.7) 
        ax4.bar(x_pos + 0.5*width, confusion_stats['tn'], width, label='TN', alpha=0.7)
        ax4.bar(x_pos + 1.5*width, confusion_stats['fn'], width, label='FN', alpha=0.7)
        
        ax4.set_title('各日期混淆矩阵平均值', fontweight='bold')
        ax4.set_xlabel('测试日期')
        ax4.set_ylabel('平均数量')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(dates, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('performance_by_date.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("💾 保存: performance_by_date.png")
        
    def create_precision_recall_scatter(self):
        """Precision-Recall散点图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('🎯 Precision-Recall 关系分析', fontsize=16, fontweight='bold')
        
        # 1. 按阈值着色的散点图
        scatter = ax1.scatter(self.df['recall'], self.df['precision'], 
                             c=self.df['pred_threshold'], s=60, alpha=0.7, 
                             cmap='viridis')
        ax1.set_xlabel('Recall')
        ax1.set_ylabel('Precision')
        ax1.set_title('Precision vs Recall (按阈值着色)', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1.05)
        
        # 添加colorbar
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('预测阈值')
        
        # 2. 按日期着色的散点图
        dates = self.df['test_date'].unique()
        colors = plt.cm.Set1(np.linspace(0, 1, len(dates)))
        
        for date, color in zip(dates, colors):
            mask = self.df['test_date'] == date
            ax2.scatter(self.df[mask]['recall'], self.df[mask]['precision'], 
                       label=date, s=60, alpha=0.7, color=color)
        
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision') 
        ax2.set_title('Precision vs Recall (按日期着色)', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1.05)
        
        plt.tight_layout()
        plt.savefig('precision_recall_scatter.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("💾 保存: precision_recall_scatter.png")
        
    def create_sample_distribution_plot(self):
        """样本分布分析图"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('📊 样本分布与网络规模分析', fontsize=16, fontweight='bold')
        
        # 1. 正负样本分布
        ax1.scatter(self.df['positive_predict_count'], self.df['negative_predict_count'], 
                   c=self.df['f1_score'], s=80, alpha=0.7, cmap='RdYlGn')
        ax1.set_xlabel('正样本数量')
        ax1.set_ylabel('负样本数量')
        ax1.set_title('正负样本分布 (F1分数着色)', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. 网络规模vs性能
        ax2.scatter(self.df['network_nodes'], self.df['f1_score'], 
                   c=self.df['pred_threshold'], s=80, alpha=0.7, cmap='plasma')
        ax2.set_xlabel('网络节点数')
        ax2.set_ylabel('F1 Score')
        ax2.set_title('网络规模 vs F1性能', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. 成功预测比例
        success_rate = self.df['successful_predictions'] / self.df['total_predict_count']
        ax3.hist(success_rate, bins=20, alpha=0.7, edgecolor='black')
        ax3.axvline(success_rate.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'平均成功率: {success_rate.mean():.3f}')
        ax3.set_xlabel('成功预测比例')
        ax3.set_ylabel('实验数量')
        ax3.set_title('贝叶斯推理成功率分布', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 证据道路数量影响
        evidence_stats = self.df.groupby('evidence_roads_count').agg({
            'precision': 'mean',
            'recall': 'mean', 
            'f1_score': 'mean'
        })
        
        x = evidence_stats.index
        ax4.plot(x, evidence_stats['precision'], 'o-', label='Precision', linewidth=2)
        ax4.plot(x, evidence_stats['recall'], 's-', label='Recall', linewidth=2)  
        ax4.plot(x, evidence_stats['f1_score'], '^-', label='F1 Score', linewidth=2)
        ax4.set_xlabel('证据道路数量')
        ax4.set_ylabel('性能指标')
        ax4.set_title('证据道路数量对性能的影响', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('sample_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("💾 保存: sample_distribution_analysis.png")
        
    def create_best_experiment_visualization(self):
        """最佳实验详细可视化"""
        exp = self.best_experiment
        predictions = exp['detailed_predictions']
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        fig.suptitle(f'🏆 最佳实验详细分析: {exp["test_date"]} (阈值={exp["pred_threshold"]})', 
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
        ax1.set_title('性能指标雷达图', fontweight='bold', pad=20)
        ax1.grid(True)
        
        # 2. 混淆矩阵
        ax2 = fig.add_subplot(gs[0, 1])
        confusion_matrix = np.array([[exp['tp'], exp['fp']], 
                                   [exp['fn'], exp['tn']]])
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', ax=ax2,
                   xticklabels=['预测洪水', '预测无洪水'],
                   yticklabels=['实际洪水', '实际无洪水'])
        ax2.set_title('混淆矩阵', fontweight='bold')
        
        # 3. 样本分布饼图
        ax3 = fig.add_subplot(gs[0, 2])
        sizes = [exp['tp'], exp['fp'], exp['tn'], exp['fn']]
        labels = [f'TP ({exp["tp"]})', f'FP ({exp["fp"]})', 
                 f'TN ({exp["tn"]})', f'FN ({exp["fn"]})']
        colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']
        ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax3.set_title('预测结果分布', fontweight='bold')
        
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
                   label=f'阈值 = {exp["pred_threshold"]}')
        
        ax4.set_xlabel('道路 (按预测概率排序)')
        ax4.set_ylabel('洪水概率')
        ax4.set_title('所有道路的洪水预测概率', fontweight='bold')
        ax4.set_xticks(x_pos[::2])  # 只显示部分标签避免重叠
        ax4.set_xticklabels([road_names[sorted_indices[i]] for i in range(0, len(road_names), 2)], 
                           rotation=90, fontsize=8)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. 证据道路信息
        ax5 = fig.add_subplot(gs[2, 0])
        evidence_roads = exp['evidence_roads']
        ax5.text(0.1, 0.9, '🔑 证据道路:', fontsize=12, fontweight='bold', transform=ax5.transAxes)
        
        evidence_text = ''
        for i, road in enumerate(evidence_roads):
            evidence_text += f'{i+1}. {road}\n'
        
        ax5.text(0.1, 0.8, evidence_text, fontsize=10, transform=ax5.transAxes, 
                verticalalignment='top')
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
        
        summary_text = f"""
📊 实验统计摘要:
• 总预测道路: {len(predictions)}条
• 成功预测: {exp['successful_predictions']}条 (100%)
• 高置信度预测(>0.7): {high_prob_total}条，其中正确{high_prob_correct}条
• 网络规模: {exp['network_nodes']}个节点, {exp['network_edges']}条边
• 证据比例: {exp['evidence_roads_count']}/{exp['test_roads_in_network']} = {exp['evidence_roads_count']/exp['test_roads_in_network']:.1%}

🎯 关键成功因素:
• 完美精确率 (100%) - 无误报
• 合理召回率 ({exp['recall']:.1%}) 
• 证据道路选择有效
• 贝叶斯推理稳定运行
        """
        
        ax6.text(0.05, 0.95, summary_text, fontsize=11, transform=ax6.transAxes,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        ax6.axis('off')
        
        plt.savefig('best_experiment_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("💾 保存: best_experiment_analysis.png")
        
    def create_all_visualizations(self):
        """创建所有可视化图表"""
        print("🎨 开始创建可视化图表...")
        
        print("\n1️⃣ 创建性能随阈值变化图...")
        self.create_performance_by_threshold_plot()
        
        print("\n2️⃣ 创建不同日期性能对比图...")
        self.create_performance_by_date_plot()
        
        print("\n3️⃣ 创建Precision-Recall散点图...")
        self.create_precision_recall_scatter()
        
        print("\n4️⃣ 创建样本分布分析图...")
        self.create_sample_distribution_plot()
        
        print("\n5️⃣ 创建最佳实验详细可视化...")
        self.create_best_experiment_visualization()
        
        print(f"\n🎉 所有可视化图表创建完成！")
        print("📁 生成的文件:")
        print("   • performance_by_threshold.png")
        print("   • performance_by_date.png") 
        print("   • precision_recall_scatter.png")
        print("   • sample_distribution_analysis.png")
        print("   • best_experiment_analysis.png")

def main():
    """主函数"""
    print("🌊 Charleston洪水预测 - 贝叶斯网络结果可视化")
    print("="*60)
    
    # 文件路径
    csv_file = "corrected_bayesian_flood_validation_full_network_summary_20250820_112441.csv"
    json_file = "best_2017_09_11_threshold_04_experiment.json"
    
    # 创建可视化器
    visualizer = BayesianResultsVisualizer(csv_file, json_file)
    
    # 生成所有可视化
    visualizer.create_all_visualizations()

if __name__ == "__main__":
    main()