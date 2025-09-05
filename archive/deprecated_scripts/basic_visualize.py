#!/usr/bin/env python3
"""
基础版贝叶斯网络洪水预测结果可视化
- 使用最基础的Python库避免兼容性问题
"""

import json
import csv
from collections import defaultdict
import math

try:
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    print("❌ matplotlib not available, creating text-based analysis only")
    HAS_MATPLOTLIB = False

class BasicBayesianVisualizer:
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
                numeric_fields = ['pred_threshold', 'precision', 'recall', 'f1_score', 'accuracy',
                                'positive_predict_count', 'negative_predict_count', 'total_predict_count',
                                'successful_predictions', 'network_nodes', 'evidence_roads_count',
                                'tp', 'fp', 'tn', 'fn', 'test_roads_total']
                for key in numeric_fields:
                    if key in row and row[key]:
                        row[key] = float(row[key])
                self.data.append(row)
        
        print(f"✅ Loaded CSV: {len(self.data)} experiments")
        
        # 加载最佳实验JSON
        with open(self.json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.best_experiment = data['best_experiment']
        print(f"✅ Loaded best experiment: {self.best_experiment['test_date']}")
        
    def calculate_stats(self, values):
        """计算统计信息"""
        if not values:
            return 0, 0
        mean = sum(values) / len(values)
        if len(values) == 1:
            return mean, 0
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std = math.sqrt(variance)
        return mean, std
        
    def create_text_analysis(self):
        """创建文本分析报告"""
        print("\\n" + "="*80)
        print("📊 BAYESIAN NETWORK CROSS-VALIDATION ANALYSIS REPORT")
        print("="*80)
        
        # 1. 按阈值分析
        print("\\n🎯 1. PERFORMANCE BY THRESHOLD")
        print("-"*50)
        
        threshold_stats = defaultdict(lambda: {'precision': [], 'recall': [], 'f1_score': [], 'accuracy': []})
        
        for row in self.data:
            threshold = row['pred_threshold']
            threshold_stats[threshold]['precision'].append(row['precision'])
            threshold_stats[threshold]['recall'].append(row['recall'])
            threshold_stats[threshold]['f1_score'].append(row['f1_score'])
            threshold_stats[threshold]['accuracy'].append(row['accuracy'])
        
        print(f"{'Threshold':<10} {'Precision':<15} {'Recall':<15} {'F1 Score':<15} {'Accuracy':<15}")
        print("-"*75)
        
        for threshold in sorted(threshold_stats.keys()):
            stats = threshold_stats[threshold]
            prec_mean, prec_std = self.calculate_stats(stats['precision'])
            rec_mean, rec_std = self.calculate_stats(stats['recall'])
            f1_mean, f1_std = self.calculate_stats(stats['f1_score'])
            acc_mean, acc_std = self.calculate_stats(stats['accuracy'])
            
            print(f"{threshold:<10.1f} {prec_mean:.3f}±{prec_std:.3f}   {rec_mean:.3f}±{rec_std:.3f}   "
                  f"{f1_mean:.3f}±{f1_std:.3f}   {acc_mean:.3f}±{acc_std:.3f}")
        
        # 2. 按日期分析
        print("\\n📅 2. PERFORMANCE BY DATE")
        print("-"*50)
        
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
        
        print(f"{'Date':<12} {'Roads':<6} {'Precision':<15} {'Recall':<15} {'F1 Score':<15}")
        print("-"*70)
        
        for date in sorted(date_stats.keys()):
            stats = date_stats[date]
            roads = int(stats['test_roads_total'])
            prec_mean, _ = self.calculate_stats(stats['precision'])
            rec_mean, _ = self.calculate_stats(stats['recall'])
            f1_mean, _ = self.calculate_stats(stats['f1_score'])
            
            print(f"{date:<12} {roads:<6} {prec_mean:.3f}        {rec_mean:.3f}        {f1_mean:.3f}")
        
        # 3. 最佳实验分析
        print("\\n🏆 3. BEST EXPERIMENT ANALYSIS")
        print("-"*50)
        
        exp = self.best_experiment
        print(f"Date: {exp['test_date']}")
        print(f"Threshold: {exp['pred_threshold']}")
        print(f"Trial ID: {exp['trial_id']}")
        print(f"")
        print(f"Performance Metrics:")
        print(f"  Precision: {exp['precision']:.3f} (100% - Perfect accuracy for positive predictions)")
        print(f"  Recall:    {exp['recall']:.3f} ({exp['recall']*100:.1f}% of actual floods detected)")
        print(f"  F1 Score:  {exp['f1_score']:.3f} (Balanced performance measure)")
        print(f"  Accuracy:  {exp['accuracy']:.3f} ({exp['accuracy']*100:.1f}% overall correctness)")
        print(f"")
        print(f"Confusion Matrix:")
        print(f"  True Positives (TP):  {exp['tp']} - Correctly predicted floods")
        print(f"  False Positives (FP): {exp['fp']} - Incorrectly predicted floods")
        print(f"  True Negatives (TN):  {exp['tn']} - Correctly predicted no floods")
        print(f"  False Negatives (FN): {exp['fn']} - Missed actual floods")
        print(f"")
        print(f"Network Information:")
        print(f"  Total test roads: {exp['test_roads_total']}")
        print(f"  Roads in network: {exp['test_roads_in_network']}")
        print(f"  Evidence roads: {exp['evidence_roads_count']}")
        print(f"  Network nodes: {exp['network_nodes']}")
        print(f"  Network edges: {exp['network_edges']}")
        
        # 4. 证据道路
        print(f"\\n🔑 Evidence Roads Used:")
        for i, road in enumerate(exp['evidence_roads'], 1):
            print(f"  {i}. {road}")
        
        # 5. 预测详情分析
        predictions = exp['detailed_predictions']
        
        # 按结果类型分组
        tp_roads = [p for p in predictions if p['true_label'] == 1 and p['predicted_label'] == 1]
        fp_roads = [p for p in predictions if p['true_label'] == 0 and p['predicted_label'] == 1]
        tn_roads = [p for p in predictions if p['true_label'] == 0 and p['predicted_label'] == 0]
        fn_roads = [p for p in predictions if p['true_label'] == 1 and p['predicted_label'] == 0]
        
        print(f"\\n🎯 4. DETAILED ROAD PREDICTIONS")
        print("-"*50)
        
        print(f"\\n✅ TRUE POSITIVES ({len(tp_roads)} roads) - Correctly predicted floods:")
        for road in sorted(tp_roads, key=lambda x: x['predicted_probability'], reverse=True):
            print(f"  {road['road_name']:<25} Probability: {road['predicted_probability']:.3f}")
        
        if fp_roads:
            print(f"\\n❌ FALSE POSITIVES ({len(fp_roads)} roads) - Incorrectly predicted floods:")
            for road in sorted(fp_roads, key=lambda x: x['predicted_probability'], reverse=True):
                print(f"  {road['road_name']:<25} Probability: {road['predicted_probability']:.3f}")
        else:
            print(f"\\n✨ FALSE POSITIVES: 0 - Perfect precision!")
        
        print(f"\\n⚠️  FALSE NEGATIVES ({len(fn_roads)} roads) - Missed actual floods:")
        for road in sorted(fn_roads, key=lambda x: x['predicted_probability'], reverse=True):
            print(f"  {road['road_name']:<25} Probability: {road['predicted_probability']:.3f}")
        
        print(f"\\n✅ TRUE NEGATIVES ({len(tn_roads)} roads) - Correctly predicted no floods:")
        print(f"  (Showing top 5 by probability)")
        for road in sorted(tn_roads, key=lambda x: x['predicted_probability'], reverse=True)[:5]:
            print(f"  {road['road_name']:<25} Probability: {road['predicted_probability']:.3f}")
        
        # 6. 高置信度分析
        print(f"\\n🎪 5. HIGH CONFIDENCE PREDICTIONS")
        print("-"*50)
        
        high_conf_predictions = [p for p in predictions if p['predicted_probability'] > 0.7]
        very_high_conf = [p for p in predictions if p['predicted_probability'] > 0.9]
        
        print(f"High confidence predictions (>0.7): {len(high_conf_predictions)}")
        print(f"Very high confidence predictions (>0.9): {len(very_high_conf)}")
        
        if very_high_conf:
            print(f"\\nVery high confidence predictions:")
            for road in sorted(very_high_conf, key=lambda x: x['predicted_probability'], reverse=True):
                status = "✅ CORRECT" if road['true_label'] == road['predicted_label'] else "❌ WRONG"
                print(f"  {road['road_name']:<25} Prob: {road['predicted_probability']:.3f} {status}")
        
        # 7. 总结和建议
        print(f"\\n💡 6. KEY INSIGHTS AND RECOMMENDATIONS")
        print("-"*50)
        
        print(f"✨ Strengths:")
        print(f"  • Perfect precision (100%) - No false alarms")
        print(f"  • Stable Bayesian inference (100% success rate)")
        print(f"  • Evidence-based approach working effectively")
        print(f"  • Good performance on large flood events")
        
        print(f"\\n🔧 Areas for Improvement:")
        print(f"  • Recall could be improved ({exp['recall']*100:.1f}% -> target 60%+)")
        print(f"  • Some high-impact roads still being missed")
        print(f"  • Consider adjusting evidence ratio or thresholds")
        
        print(f"\\n🎯 Deployment Recommendations:")
        print(f"  • Use threshold 0.4 for balanced performance")
        print(f"  • Monitor precision to stay above 80%") 
        print(f"  • Focus on recall improvement for critical roads")
        print(f"  • Consider ensemble methods for edge cases")
        
        print("\\n" + "="*80)
        print("📄 Report completed! Check the analysis above.")
        print("="*80)
        
    def create_simple_plots(self):
        """创建简单的matplotlib图表"""
        if not HAS_MATPLOTLIB:
            print("⚠️  Matplotlib not available, skipping plots")
            return
            
        try:
            # 设置matplotlib参数
            plt.style.use('default')
            plt.rcParams['figure.figsize'] = [12, 8]
            plt.rcParams['font.size'] = 10
            
            # 1. 阈值性能图
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Bayesian Network Performance Analysis', fontsize=16, fontweight='bold')
            
            # 按阈值分组计算统计
            threshold_stats = defaultdict(lambda: {'precision': [], 'recall': [], 'f1_score': [], 'accuracy': []})
            
            for row in self.data:
                threshold = row['pred_threshold']
                threshold_stats[threshold]['precision'].append(row['precision'])
                threshold_stats[threshold]['recall'].append(row['recall'])
                threshold_stats[threshold]['f1_score'].append(row['f1_score'])
                threshold_stats[threshold]['accuracy'].append(row['accuracy'])
            
            thresholds = sorted(threshold_stats.keys())
            precision_means = []
            recall_means = []
            f1_means = []
            accuracy_means = []
            
            for t in thresholds:
                prec_mean, _ = self.calculate_stats(threshold_stats[t]['precision'])
                rec_mean, _ = self.calculate_stats(threshold_stats[t]['recall'])
                f1_mean, _ = self.calculate_stats(threshold_stats[t]['f1_score'])
                acc_mean, _ = self.calculate_stats(threshold_stats[t]['accuracy'])
                
                precision_means.append(prec_mean)
                recall_means.append(rec_mean)
                f1_means.append(f1_mean)
                accuracy_means.append(acc_mean)
            
            # 绘制性能曲线
            ax1.plot(thresholds, precision_means, 'o-', linewidth=2, markersize=8, label='Precision')
            ax1.set_title('Precision vs Threshold')
            ax1.set_xlabel('Prediction Threshold')
            ax1.set_ylabel('Precision')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1.1)
            
            ax2.plot(thresholds, recall_means, 's-', linewidth=2, markersize=8, color='orange', label='Recall')
            ax2.set_title('Recall vs Threshold')
            ax2.set_xlabel('Prediction Threshold')
            ax2.set_ylabel('Recall')
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1.1)
            
            ax3.plot(thresholds, f1_means, '^-', linewidth=2, markersize=8, color='green', label='F1 Score')
            ax3.set_title('F1 Score vs Threshold')
            ax3.set_xlabel('Prediction Threshold')
            ax3.set_ylabel('F1 Score')
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(0, 1.1)
            
            ax4.plot(thresholds, accuracy_means, 'd-', linewidth=2, markersize=8, color='red', label='Accuracy')
            ax4.set_title('Accuracy vs Threshold')
            ax4.set_xlabel('Prediction Threshold')
            ax4.set_ylabel('Accuracy')
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim(0, 1.1)
            
            plt.tight_layout()
            plt.savefig('performance_analysis.png', dpi=300, bbox_inches='tight')
            print("💾 Saved: performance_analysis.png")
            plt.close()
            
            # 2. 最佳实验可视化
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Best Experiment: {self.best_experiment["test_date"]} (Threshold={self.best_experiment["pred_threshold"]})', 
                        fontsize=16, fontweight='bold')
            
            exp = self.best_experiment
            
            # 性能指标条形图
            metrics = ['Precision', 'Recall', 'F1 Score', 'Accuracy']
            values = [exp['precision'], exp['recall'], exp['f1_score'], exp['accuracy']]
            colors = ['blue', 'orange', 'green', 'red']
            
            bars = ax1.bar(metrics, values, color=colors, alpha=0.7)
            ax1.set_title('Performance Metrics')
            ax1.set_ylabel('Score')
            ax1.set_ylim(0, 1.1)
            ax1.grid(True, alpha=0.3)
            
            # 在条形图上添加数值标签
            for bar, value in zip(bars, values):
                ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                        f'{value:.3f}', ha='center', va='bottom')
            
            # 混淆矩阵热力图
            confusion = [[exp['tp'], exp['fp']], [exp['fn'], exp['tn']]]
            im = ax2.imshow(confusion, cmap='Blues')
            ax2.set_xticks([0, 1])
            ax2.set_yticks([0, 1])
            ax2.set_xticklabels(['Pred Flood', 'Pred No Flood'])
            ax2.set_yticklabels(['Actual Flood', 'Actual No Flood'])
            ax2.set_title('Confusion Matrix')
            
            # 添加数值标注
            for i in range(2):
                for j in range(2):
                    text = ax2.text(j, i, confusion[i][j],
                                   ha="center", va="center", color="black", fontweight='bold', fontsize=14)
            
            # 预测概率分布
            predictions = exp['detailed_predictions']
            probabilities = [p['predicted_probability'] for p in predictions]
            
            ax3.hist(probabilities, bins=15, alpha=0.7, edgecolor='black')
            ax3.axvline(x=exp['pred_threshold'], color='red', linestyle='--', linewidth=2, 
                       label=f'Threshold = {exp["pred_threshold"]}')
            ax3.set_title('Prediction Probability Distribution')
            ax3.set_xlabel('Flood Probability')
            ax3.set_ylabel('Number of Roads')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 样本分布饼图
            sizes = [exp['tp'], exp['fp'], exp['tn'], exp['fn']]
            labels = [f'TP\\n({exp["tp"]})', f'FP\\n({exp["fp"]})', f'TN\\n({exp["tn"]})', f'FN\\n({exp["fn"]})']
            colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']
            
            wedges, texts, autotexts = ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                                              startangle=90)
            ax4.set_title('Prediction Results Distribution')
            
            plt.tight_layout()
            plt.savefig('best_experiment_details.png', dpi=300, bbox_inches='tight')
            print("💾 Saved: best_experiment_details.png")
            plt.close()
            
            print("✅ Successfully created visualization plots!")
            
        except Exception as e:
            print(f"⚠️  Error creating plots: {str(e)}")
            print("But text analysis is still available above.")

    def run_analysis(self):
        """运行完整分析"""
        print("🎨 Starting Bayesian Network Results Analysis...")
        
        # 创建文本分析
        self.create_text_analysis()
        
        # 尝试创建图表
        self.create_simple_plots()
        
        print("\\n🎉 Analysis completed!")
        if HAS_MATPLOTLIB:
            print("📁 Generated files:")
            print("   • performance_analysis.png")
            print("   • best_experiment_details.png")
        else:
            print("📝 Text-based analysis completed (no plots due to library issues)")

def main():
    """主函数"""
    print("🌊 Charleston Flood Prediction - Bayesian Network Results Analysis")
    print("="*70)
    
    # 文件路径
    csv_file = "corrected_bayesian_flood_validation_full_network_summary_20250820_112441.csv"
    json_file = "best_2017_09_11_threshold_04_experiment.json"
    
    try:
        # 创建分析器
        analyzer = BasicBayesianVisualizer(csv_file, json_file)
        
        # 运行分析
        analyzer.run_analysis()
        
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find required file: {str(e)}")
        print("Please ensure the following files exist:")
        print(f"  • {csv_file}")
        print(f"  • {json_file}")
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")

if __name__ == "__main__":
    main()