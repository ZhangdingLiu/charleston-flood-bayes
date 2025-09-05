#!/usr/bin/env python3
"""
逐样本详细评估输出

展示测试集中每个样本的完整评估过程：
1. 每个测试日的详细推理过程
2. Evidence设置和目标选择
3. 每个样本的预测概率和决策
4. TP/FP/TN/FN的具体归类
5. 最终混淆矩阵和指标计算
6. 可视化结果
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

class SampleBySeamleEvaluator:
    """逐样本详细评估器"""
    
    def __init__(self, flood_net, test_df):
        self.flood_net = flood_net
        self.test_df = test_df
        self.bn_nodes = set(flood_net.network_bayes.nodes())
        self.marginals_dict = dict(zip(
            flood_net.marginals['link_id'], 
            flood_net.marginals['p']
        ))
        
        # 评估参数
        self.positive_threshold = 0.6
        self.negative_threshold = 0.3
        self.min_marginal_for_negative = 0.15
        
        # 存储详细结果
        self.detailed_results = []
        self.confusion_matrix = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
        self.uncertain_predictions = 0
        
    def get_negative_candidates(self):
        """获取负样本候选"""
        return [
            road for road, prob in self.marginals_dict.items() 
            if road in self.bn_nodes and prob <= self.min_marginal_for_negative
        ]
    
    def make_prediction_with_details(self, target_road, evidence):
        """进行预测并返回详细信息"""
        try:
            result = self.flood_net.infer_w_evidence(target_road, evidence)
            prob_flood = result['flooded']
            
            # 获取父节点信息
            parents = list(self.flood_net.network.predecessors(target_road))
            relevant_evidence = {k: v for k, v in evidence.items() if k in parents}
            
            # 阈值决策
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
            
            return {
                'prob_flood': prob_flood,
                'prediction': prediction,
                'decision': decision,
                'confidence': confidence,
                'parents': parents,
                'relevant_evidence': relevant_evidence,
                'marginal_prob': self.marginals_dict.get(target_road, 0),
                'success': True,
                'error': None
            }
            
        except Exception as e:
            return {
                'prob_flood': 0.0,
                'prediction': -1,
                'decision': "ERROR",
                'confidence': 0.0,
                'parents': [],
                'relevant_evidence': {},
                'marginal_prob': self.marginals_dict.get(target_road, 0),
                'success': False,
                'error': str(e)
            }
    
    def evaluate_single_day(self, date, day_group, day_index):
        """评估单日数据"""
        print(f"\n📅 【第{day_index}天】{date.date()}")
        print("=" * 60)
        
        # 获取当天洪水道路
        flooded_roads = list(day_group["link_id"].unique())
        flooded_in_bn = [road for road in flooded_roads if road in self.bn_nodes]
        
        print(f"🌊 当天洪水情况:")
        print(f"   原始洪水道路: {len(flooded_roads)}条")
        print(f"   网络中洪水道路: {len(flooded_in_bn)}条")
        print(f"   洪水道路列表: {flooded_roads}")
        print(f"   网络道路列表: {flooded_in_bn}")
        
        day_results = {
            'date': date,
            'flooded_roads': flooded_roads,
            'flooded_in_bn': flooded_in_bn,
            'positive_samples': [],
            'negative_samples': [],
            'day_tp': 0,
            'day_fp': 0,
            'day_tn': 0,
            'day_fn': 0,
            'day_uncertain': 0
        }
        
        if len(flooded_in_bn) < 2:
            print("⚠️  可用网络道路不足2条，跳过此日期")
            return day_results
        
        # Evidence选择策略
        evidence_count = max(1, int(len(flooded_in_bn) * 0.5))
        evidence_roads = flooded_in_bn[:evidence_count]
        target_roads = flooded_in_bn[evidence_count:]
        
        evidence = {road: 1 for road in evidence_roads}
        
        print(f"\n🎯 Evidence设置:")
        print(f"   Evidence道路 ({len(evidence_roads)}条): {evidence_roads}")
        for road in evidence_roads:
            marginal_p = self.marginals_dict.get(road, 0)
            print(f"     {road}: P(洪水)={marginal_p:.3f}")
        
        print(f"\n🔍 正样本推理 ({len(target_roads)}条):")
        
        # 处理正样本（真实洪水道路）
        for i, target_road in enumerate(target_roads, 1):
            pred_details = self.make_prediction_with_details(target_road, evidence)
            
            sample_result = {
                'type': 'POSITIVE',
                'road': target_road,
                'true_label': 1,
                'evidence': evidence.copy(),
                'prediction_details': pred_details
            }
            
            day_results['positive_samples'].append(sample_result)
            
            # 显示详细推理过程
            print(f"\n   [{i}] 目标道路: {target_road}")
            print(f"       边际概率: P(洪水)={pred_details['marginal_prob']:.6f}")
            print(f"       父节点: {pred_details['parents']}")
            print(f"       相关Evidence: {pred_details['relevant_evidence']}")
            
            if pred_details['success']:
                print(f"       推理概率: P(洪水|Evidence)={pred_details['prob_flood']:.6f}")
                print(f"       决策: {pred_details['decision']} (置信度={pred_details['confidence']:.3f})")
                
                # 计算混淆矩阵贡献
                if pred_details['prediction'] == 1:
                    day_results['day_tp'] += 1
                    result_type = "TP ✅"
                elif pred_details['prediction'] == 0:
                    day_results['day_fn'] += 1
                    result_type = "FN ❌"
                else:
                    day_results['day_uncertain'] += 1
                    result_type = "UNCERTAIN ❓"
                
                print(f"       结果: 预测={pred_details['prediction']}, 真实=1 → {result_type}")
            else:
                print(f"       ❌ 推理失败: {pred_details['error']}")
                day_results['day_uncertain'] += 1
        
        # 处理负样本
        negative_candidates = self.get_negative_candidates()
        available_negatives = [road for road in negative_candidates if road not in flooded_roads]
        selected_negatives = available_negatives[:min(3, len(target_roads))]
        
        if selected_negatives:
            print(f"\n🚫 负样本推理 ({len(selected_negatives)}条):")
            
            for i, neg_road in enumerate(selected_negatives, 1):
                pred_details = self.make_prediction_with_details(neg_road, evidence)
                
                sample_result = {
                    'type': 'NEGATIVE',
                    'road': neg_road,
                    'true_label': 0,
                    'evidence': evidence.copy(),
                    'prediction_details': pred_details
                }
                
                day_results['negative_samples'].append(sample_result)
                
                print(f"\n   [{i}] 负样本道路: {neg_road}")
                print(f"       边际概率: P(洪水)={pred_details['marginal_prob']:.6f}")
                print(f"       选择理由: 低概率且当天无洪水记录")
                
                if pred_details['success']:
                    print(f"       推理概率: P(洪水|Evidence)={pred_details['prob_flood']:.6f}")
                    print(f"       决策: {pred_details['decision']} (置信度={pred_details['confidence']:.3f})")
                    
                    # 计算混淆矩阵贡献
                    if pred_details['prediction'] == 0:
                        day_results['day_tn'] += 1
                        result_type = "TN ✅"
                    elif pred_details['prediction'] == 1:
                        day_results['day_fp'] += 1
                        result_type = "FP ❌"
                    else:
                        day_results['day_uncertain'] += 1
                        result_type = "UNCERTAIN ❓"
                    
                    print(f"       结果: 预测={pred_details['prediction']}, 真实=0 → {result_type}")
                else:
                    print(f"       ❌ 推理失败: {pred_details['error']}")
                    day_results['day_uncertain'] += 1
        
        # 当日统计
        total_samples = len(day_results['positive_samples']) + len(day_results['negative_samples'])
        
        print(f"\n📊 当日统计:")
        print(f"   总样本数: {total_samples}")
        print(f"   TP: {day_results['day_tp']}, FP: {day_results['day_fp']}")
        print(f"   TN: {day_results['day_tn']}, FN: {day_results['day_fn']}")
        print(f"   不确定: {day_results['day_uncertain']}")
        
        if total_samples > 0:
            day_precision = day_results['day_tp'] / (day_results['day_tp'] + day_results['day_fp']) if (day_results['day_tp'] + day_results['day_fp']) > 0 else 0
            day_recall = day_results['day_tp'] / (day_results['day_tp'] + day_results['day_fn']) if (day_results['day_tp'] + day_results['day_fn']) > 0 else 0
            day_accuracy = (day_results['day_tp'] + day_results['day_tn']) / (total_samples - day_results['day_uncertain']) if (total_samples - day_results['day_uncertain']) > 0 else 0
            
            print(f"   当日精确度: {day_precision:.3f}")
            print(f"   当日召回率: {day_recall:.3f}")
            print(f"   当日准确率: {day_accuracy:.3f}")
        
        return day_results
    
    def run_detailed_evaluation(self, max_days=None):
        """运行详细评估"""
        print("🔬 逐样本详细评估开始")
        print("=" * 80)
        
        # 按日期分组测试数据
        test_by_date = self.test_df.groupby(self.test_df["time_create"].dt.floor("D"))
        
        day_groups = list(test_by_date)
        if max_days:
            day_groups = day_groups[:max_days]
        
        print(f"📅 将评估 {len(day_groups)} 个测试日")
        
        # 逐日评估
        for day_index, (date, day_group) in enumerate(day_groups, 1):
            day_result = self.evaluate_single_day(date, day_group, day_index)
            
            if day_result['positive_samples'] or day_result['negative_samples']:
                self.detailed_results.append(day_result)
                
                # 累计混淆矩阵
                self.confusion_matrix['TP'] += day_result['day_tp']
                self.confusion_matrix['FP'] += day_result['day_fp']
                self.confusion_matrix['TN'] += day_result['day_tn']
                self.confusion_matrix['FN'] += day_result['day_fn']
                self.uncertain_predictions += day_result['day_uncertain']
        
        return self.detailed_results
    
    def generate_confusion_matrix_analysis(self):
        """生成混淆矩阵分析"""
        print(f"\n\n📈 混淆矩阵详细分析")
        print("=" * 80)
        
        # 基础统计
        tp = self.confusion_matrix['TP']
        fp = self.confusion_matrix['FP']
        tn = self.confusion_matrix['TN']
        fn = self.confusion_matrix['FN']
        uncertain = self.uncertain_predictions
        
        total_predictions = tp + fp + tn + fn
        total_samples = total_predictions + uncertain
        
        print(f"🔢 样本统计:")
        print(f"   总样本数: {total_samples}")
        print(f"   有效预测: {total_predictions}")
        print(f"   不确定预测: {uncertain}")
        print(f"   弃权率: {uncertain/total_samples*100:.1f}%")
        
        # 混淆矩阵
        print(f"\n📊 混淆矩阵:")
        print(f"                  预测")
        print(f"              正类    负类")
        print(f"    真实 正类  {tp:4d}   {fn:4d}")
        print(f"         负类  {fp:4d}   {tn:4d}")
        
        # 详细分类统计
        print(f"\n📋 详细分类:")
        print(f"   True Positives (TP):  {tp:4d} - 正确预测为洪水")
        print(f"   False Positives (FP): {fp:4d} - 错误预测为洪水")
        print(f"   True Negatives (TN):  {tn:4d} - 正确预测为无洪水")
        print(f"   False Negatives (FN): {fn:4d} - 错误预测为无洪水")
        print(f"   Uncertain:            {uncertain:4d} - 不确定预测(弃权)")
        
        # 计算指标
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        accuracy = (tp + tn) / total_predictions if total_predictions > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"\n📈 性能指标:")
        print(f"   精确度 (Precision):    {precision:.6f}")
        print(f"   召回率 (Recall):       {recall:.6f}")
        print(f"   特异性 (Specificity):  {specificity:.6f}")
        print(f"   准确率 (Accuracy):     {accuracy:.6f}")
        print(f"   F1分数 (F1-Score):     {f1_score:.6f}")
        
        # 目标达成情况
        print(f"\n🎯 目标达成情况:")
        precision_target = 0.8
        recall_target = 0.3
        
        precision_status = "✅ 达成" if precision >= precision_target else "❌ 未达成"
        recall_status = "✅ 达成" if recall >= recall_target else "❌ 未达成"
        
        print(f"   精确度目标 (≥{precision_target}): {precision:.3f} {precision_status}")
        print(f"   召回率目标 (≥{recall_target}):  {recall:.3f} {recall_status}")
        
        return {
            'confusion_matrix': self.confusion_matrix,
            'metrics': {
                'precision': precision,
                'recall': recall,
                'specificity': specificity,
                'accuracy': accuracy,
                'f1_score': f1_score
            },
            'targets': {
                'precision_target': precision_target,
                'recall_target': recall_target,
                'precision_achieved': precision >= precision_target,
                'recall_achieved': recall >= recall_target
            }
        }
    
    def create_confusion_matrix_visualization(self, save_path="confusion_matrix.png"):
        """创建混淆矩阵可视化"""
        print(f"\n🎨 生成混淆矩阵可视化")
        
        # 准备数据
        tp = self.confusion_matrix['TP']
        fp = self.confusion_matrix['FP']
        tn = self.confusion_matrix['TN']
        fn = self.confusion_matrix['FN']
        
        # 创建混淆矩阵数组
        cm_array = np.array([[tp, fn], [fp, tn]])
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 混淆矩阵热图
        sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['预测负类', '预测正类'],
                   yticklabels=['真实正类', '真实负类'],
                   ax=ax1, annot_kws={'size': 14})
        ax1.set_title('混淆矩阵', fontsize=16, fontweight='bold')
        ax1.set_xlabel('预测标签', fontsize=12)
        ax1.set_ylabel('真实标签', fontsize=12)
        
        # 在热图中添加百分比
        total = cm_array.sum()
        for i in range(2):
            for j in range(2):
                percentage = cm_array[i, j] / total * 100
                ax1.text(j+0.5, i+0.7, f'({percentage:.1f}%)', 
                        ha='center', va='center', fontsize=10, color='gray')
        
        # 指标柱状图
        metrics = self.generate_confusion_matrix_analysis()['metrics']
        metric_names = ['精确度', '召回率', '特异性', '准确率', 'F1分数']
        metric_values = [metrics['precision'], metrics['recall'], 
                        metrics['specificity'], metrics['accuracy'], metrics['f1_score']]
        
        colors = ['#ff7f0e' if name in ['精确度', '召回率'] else '#1f77b4' for name in metric_names]
        bars = ax2.bar(metric_names, metric_values, color=colors, alpha=0.7)
        
        # 添加目标线
        ax2.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='精确度目标(0.8)')
        ax2.axhline(y=0.3, color='orange', linestyle='--', alpha=0.7, label='召回率目标(0.3)')
        
        ax2.set_title('性能指标', fontsize=16, fontweight='bold')
        ax2.set_ylabel('得分', fontsize=12)
        ax2.set_ylim(0, 1)
        ax2.legend()
        
        # 在柱状图上添加数值
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 混淆矩阵可视化已保存至: {save_path}")
        plt.show()
        
        return fig
    
    def generate_sample_summary_table(self):
        """生成样本汇总表"""
        print(f"\n📋 生成样本汇总表")
        
        all_samples = []
        
        for day_result in self.detailed_results:
            date = day_result['date']
            
            # 处理正样本
            for sample in day_result['positive_samples']:
                pred_details = sample['prediction_details']
                all_samples.append({
                    'Date': date.date(),
                    'Road': sample['road'],
                    'Type': 'Positive',
                    'True_Label': sample['true_label'],
                    'Marginal_Prob': pred_details['marginal_prob'],
                    'Inferred_Prob': pred_details['prob_flood'],
                    'Prediction': pred_details['prediction'],
                    'Decision': pred_details['decision'],
                    'Confidence': pred_details['confidence'],
                    'Result_Type': self._get_result_type(pred_details['prediction'], sample['true_label']),
                    'Evidence_Count': len(sample['evidence']),
                    'Parents': len(pred_details['parents']),
                    'Relevant_Evidence': len(pred_details['relevant_evidence'])
                })
            
            # 处理负样本
            for sample in day_result['negative_samples']:
                pred_details = sample['prediction_details']
                all_samples.append({
                    'Date': date.date(),
                    'Road': sample['road'],
                    'Type': 'Negative',
                    'True_Label': sample['true_label'],
                    'Marginal_Prob': pred_details['marginal_prob'],
                    'Inferred_Prob': pred_details['prob_flood'],
                    'Prediction': pred_details['prediction'],
                    'Decision': pred_details['decision'],
                    'Confidence': pred_details['confidence'],
                    'Result_Type': self._get_result_type(pred_details['prediction'], sample['true_label']),
                    'Evidence_Count': len(sample['evidence']),
                    'Parents': len(pred_details['parents']),
                    'Relevant_Evidence': len(pred_details['relevant_evidence'])
                })
        
        # 创建DataFrame
        df_summary = pd.DataFrame(all_samples)
        
        # 保存到CSV
        csv_path = "sample_by_sample_results.csv"
        df_summary.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"✅ 样本汇总表已保存至: {csv_path}")
        
        # 显示前10行
        print(f"\n📊 样本汇总表预览 (前10行):")
        print(df_summary.head(10).to_string(index=False))
        
        # 统计汇总
        print(f"\n📈 样本分布统计:")
        print(f"   总样本数: {len(df_summary)}")
        print(f"   正样本数: {len(df_summary[df_summary['Type'] == 'Positive'])}")
        print(f"   负样本数: {len(df_summary[df_summary['Type'] == 'Negative'])}")
        
        result_counts = df_summary['Result_Type'].value_counts()
        for result_type, count in result_counts.items():
            print(f"   {result_type}: {count}")
        
        return df_summary
    
    def _get_result_type(self, prediction, true_label):
        """获取结果类型"""
        if prediction == -1:
            return "UNCERTAIN"
        elif prediction == 1 and true_label == 1:
            return "TP"
        elif prediction == 1 and true_label == 0:
            return "FP"
        elif prediction == 0 and true_label == 1:
            return "FN"
        elif prediction == 0 and true_label == 0:
            return "TN"
        else:
            return "UNKNOWN"

def load_test_system():
    """加载测试系统"""
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
    print("🔬 Charleston洪水预测 - 逐样本详细评估")
    print("=" * 80)
    
    # 加载系统
    flood_net, test_df = load_test_system()
    
    # 创建评估器
    evaluator = SampleBySeamleEvaluator(flood_net, test_df)
    
    print(f"📊 系统配置:")
    print(f"   网络节点数: {len(evaluator.bn_nodes)}")
    print(f"   测试天数: {test_df['time_create'].dt.floor('D').nunique()}")
    print(f"   正预测阈值: {evaluator.positive_threshold}")
    print(f"   负预测阈值: {evaluator.negative_threshold}")
    
    # 运行详细评估（限制天数以避免输出过长）
    detailed_results = evaluator.run_detailed_evaluation(max_days=10)  # 可以调整天数
    
    # 生成混淆矩阵分析
    analysis_results = evaluator.generate_confusion_matrix_analysis()
    
    # 创建可视化
    fig = evaluator.create_confusion_matrix_visualization()
    
    # 生成样本汇总表
    df_summary = evaluator.generate_sample_summary_table()
    
    print(f"\n🎉 逐样本详细评估完成！")
    print(f"📁 生成的文件:")
    print(f"   - confusion_matrix.png (混淆矩阵可视化)")
    print(f"   - sample_by_sample_results.csv (详细样本结果)")
    
    return evaluator, analysis_results, df_summary

if __name__ == "__main__":
    evaluator, results, summary_df = main()