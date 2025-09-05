#!/usr/bin/env python3
"""
Fix and Visualize Results
修复JSON序列化问题并生成完整的可视化结果

这个脚本会处理已有的CSV结果，修复JSON序列化问题，并生成完整的可视化分析。
"""

import sys
import os
import pandas as pd
import numpy as np
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'visualization'))

def fix_json_serialization(obj):
    """修复JSON序列化问题的辅助函数"""
    if isinstance(obj, dict):
        return {k: fix_json_serialization(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [fix_json_serialization(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def analyze_and_visualize_results(result_dir):
    """分析结果并生成可视化"""
    print("🔧 修复结果并生成可视化分析")
    print("=" * 60)
    
    # 检查结果文件
    csv_file = os.path.join(result_dir, "complete_results.csv")
    if not os.path.exists(csv_file):
        print(f"❌ 结果文件不存在: {csv_file}")
        return False
    
    print(f"📊 加载结果数据: {csv_file}")
    try:
        df = pd.read_csv(csv_file)
        print(f"✅ 成功加载 {len(df)} 个参数组合结果")
    except Exception as e:
        print(f"❌ 加载数据失败: {e}")
        return False
    
    # 数据基本统计
    print(f"\n📈 数据统计:")
    print(f"总组合数: {len(df)}")
    print(f"F1分数范围: {df['f1_score'].min():.3f} - {df['f1_score'].max():.3f}")
    print(f"精确度范围: {df['precision'].min():.3f} - {df['precision'].max():.3f}")
    print(f"召回率范围: {df['recall'].min():.3f} - {df['recall'].max():.3f}")
    
    # 生成推荐配置（修复版本）
    print(f"\n💡 生成推荐配置...")
    try:
        recommendations = generate_recommendations(df)
        
        # 修复JSON序列化问题
        fixed_recommendations = fix_json_serialization(recommendations)
        
        # 保存推荐结果
        rec_file = os.path.join(result_dir, "parameter_recommendations.json")
        with open(rec_file, 'w', encoding='utf-8') as f:
            json.dump(fixed_recommendations, f, indent=2, ensure_ascii=False)
        print(f"✅ 推荐配置已保存: {rec_file}")
        
    except Exception as e:
        print(f"⚠️ 推荐配置生成失败: {e}")
        recommendations = {}
    
    # 生成可视化
    print(f"\n🎨 生成可视化分析...")
    try:
        from visualization.parameter_analysis_visualizer import ParameterVisualizer
        
        viz_dir = os.path.join(result_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        visualizer = ParameterVisualizer(df, viz_dir)
        
        # 定义约束条件用于可视化
        constraints = {
            'min_precision': 0.8,
            'min_recall': 0.8,
            'min_f1_score': 0.7,
            'min_samples': 50
        }
        
        print("  生成3D性能分布图...")
        visualizer.create_3d_performance_scatter()
        
        print("  生成参数热图...")
        visualizer.create_parameter_heatmaps()
        
        print("  生成参数敏感性分析...")
        visualizer.create_parameter_sensitivity_analysis()
        
        print("  生成Pareto前沿分析...")
        visualizer.create_pareto_frontier_analysis()
        
        print("  生成约束筛选分析...")
        filtered_df = visualizer.create_constraint_filtering_visualization(constraints)
        
        print(f"✅ 可视化生成完成!")
        print(f"✅ 约束条件下筛选出 {len(filtered_df)} 个满足条件的配置")
        
    except Exception as e:
        print(f"❌ 可视化生成失败: {e}")
        import traceback
        traceback.print_exc()
        filtered_df = df
    
    # 生成完整报告
    print(f"\n📝 生成分析报告...")
    try:
        generate_complete_report(df, filtered_df, result_dir, recommendations)
        print(f"✅ 分析报告生成完成")
    except Exception as e:
        print(f"⚠️ 报告生成失败: {e}")
    
    return True

def generate_recommendations(df):
    """生成推荐配置"""
    recommendations = {}
    
    # 基于所有结果的推荐
    best_f1_idx = df['f1_score'].idxmax()
    best_precision_idx = df['precision'].idxmax()
    best_recall_idx = df['recall'].idxmax()
    
    recommendations['overall'] = {
        'best_f1': df.iloc[best_f1_idx].to_dict(),
        'best_precision': df.iloc[best_precision_idx].to_dict(),
        'best_recall': df.iloc[best_recall_idx].to_dict()
    }
    
    # 高标准约束条件筛选
    high_std_mask = (df['precision'] >= 0.8) & (df['recall'] >= 0.6) & (df['f1_score'] >= 0.7)
    high_std_df = df[high_std_mask]
    
    if len(high_std_df) > 0:
        recommendations['high_standard'] = {
            'count': len(high_std_df),
            'best_f1': high_std_df.iloc[high_std_df['f1_score'].idxmax()].to_dict()
        }
    
    # 平衡约束条件筛选
    balanced_mask = (df['precision'] >= 0.7) & (df['recall'] >= 0.7)
    balanced_df = df[balanced_mask]
    
    if len(balanced_df) > 0:
        recommendations['balanced'] = {
            'count': len(balanced_df),
            'best_f1': balanced_df.iloc[balanced_df['f1_score'].idxmax()].to_dict()
        }
    
    return recommendations

def generate_complete_report(df, filtered_df, result_dir, recommendations):
    """生成完整分析报告"""
    report_file = os.path.join(result_dir, "complete_analysis_report.md")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 贝叶斯网络参数优化完整分析报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**分析脚本**: fix_and_visualize_results.py\n\n")
        
        # 实验概述
        f.write("## 📊 实验概述\n\n")
        f.write(f"- **总参数组合数**: 8,640 (4×3×4×4×5×3×3)\n")
        f.write(f"- **成功评估组合数**: {len(df)}\n")
        f.write(f"- **成功率**: {len(df)/8640*100:.1f}%\n")
        f.write(f"- **失败组合数**: {8640-len(df)} (主要原因：负样本候选不足)\n\n")
        
        # 性能统计
        f.write("## 📈 性能统计\n\n")
        f.write("| 指标 | 最小值 | 最大值 | 平均值 | 标准差 | 中位数 |\n")
        f.write("|------|--------|--------|--------|--------|--------|\n")
        
        metrics = ['precision', 'recall', 'f1_score', 'accuracy', 'total_samples']
        for metric in metrics:
            if metric in df.columns:
                min_val = df[metric].min()
                max_val = df[metric].max()
                mean_val = df[metric].mean()
                std_val = df[metric].std()
                median_val = df[metric].median()
                
                f.write(f"| {metric.replace('_', ' ').title()} | {min_val:.3f} | {max_val:.3f} | {mean_val:.3f} | {std_val:.3f} | {median_val:.3f} |\n")
        f.write("\n")
        
        # 最佳配置
        f.write("## 🏆 最佳配置\n\n")
        
        best_f1_idx = df['f1_score'].idxmax()
        best_config = df.iloc[best_f1_idx]
        
        f.write(f"### 最佳F1分数配置 (F1 = {best_config['f1_score']:.3f})\n\n")
        f.write(f"**网络参数**:\n")
        f.write(f"- occ_thr (最小出现次数): {best_config['occ_thr']}\n")
        f.write(f"- edge_thr (边最小共现): {best_config['edge_thr']}\n")
        f.write(f"- weight_thr (边权重阈值): {best_config['weight_thr']}\n\n")
        
        f.write(f"**评估参数**:\n")
        f.write(f"- evidence_count (证据数量): {best_config['evidence_count']}\n")
        f.write(f"- pred_threshold (预测阈值): {best_config['pred_threshold']}\n")
        f.write(f"- neg_pos_ratio (负正样本比): {best_config['neg_pos_ratio']}\n")
        f.write(f"- marginal_prob_threshold (边际概率阈值): {best_config['marginal_prob_threshold']}\n\n")
        
        f.write(f"**性能表现**:\n")
        f.write(f"- 精确度 (Precision): {best_config['precision']:.3f}\n")
        f.write(f"- 召回率 (Recall): {best_config['recall']:.3f}\n")
        f.write(f"- F1分数: {best_config['f1_score']:.3f}\n")
        f.write(f"- 准确率: {best_config['accuracy']:.3f}\n")
        f.write(f"- 测试样本数: {best_config['total_samples']}\n")
        f.write(f"- 网络节点数: {best_config['network_nodes']}\n\n")
        
        # 高标准配置分析
        high_std_mask = (df['precision'] >= 0.8) & (df['recall'] >= 0.6) & (df['f1_score'] >= 0.7)
        high_std_df = df[high_std_mask]
        
        f.write(f"## 🎯 高标准配置分析 (P≥0.8, R≥0.6, F1≥0.7)\n\n")
        f.write(f"- **满足条件的配置数**: {len(high_std_df)}\n")
        f.write(f"- **比例**: {len(high_std_df)/len(df)*100:.1f}%\n\n")
        
        if len(high_std_df) > 0:
            high_best = high_std_df.iloc[high_std_df['f1_score'].idxmax()]
            f.write(f"**高标准中的最佳配置** (F1 = {high_best['f1_score']:.3f}):\n")
            f.write(f"- 参数组合: occ_thr={high_best['occ_thr']}, edge_thr={high_best['edge_thr']}, weight_thr={high_best['weight_thr']}\n")
            f.write(f"- 评估参数: evidence_count={high_best['evidence_count']}, pred_threshold={high_best['pred_threshold']}\n")
            f.write(f"- 性能: P={high_best['precision']:.3f}, R={high_best['recall']:.3f}, F1={high_best['f1_score']:.3f}\n\n")
        
        # 参数重要性分析
        f.write("## 🔍 参数重要性分析\n\n")
        
        param_impact = {}
        categorical_params = ['occ_thr', 'edge_thr', 'evidence_count']
        
        for param in categorical_params:
            param_performance = df.groupby(param)['f1_score'].agg(['mean', 'std', 'count']).round(4)
            best_value = param_performance['mean'].idxmax()
            worst_value = param_performance['mean'].idxmin()
            impact = param_performance['mean'].max() - param_performance['mean'].min()
            
            param_impact[param] = {
                'impact': impact,
                'best_value': best_value,
                'worst_value': worst_value
            }
            
            f.write(f"### {param.replace('_', ' ').title()}\n")
            f.write(f"- **性能影响范围**: {impact:.3f}\n")
            f.write(f"- **最佳值**: {best_value} (F1均值: {param_performance.loc[best_value, 'mean']:.3f})\n")
            f.write(f"- **最差值**: {worst_value} (F1均值: {param_performance.loc[worst_value, 'mean']:.3f})\n\n")
        
        # 使用建议
        f.write("## 💡 使用建议\n\n")
        f.write("### 根据应用场景选择配置\n\n")
        f.write("1. **高精度场景** (减少误报):\n")
        high_precision_config = df.iloc[df['precision'].idxmax()]
        f.write(f"   - 推荐配置: occ_thr={high_precision_config['occ_thr']}, evidence_count={high_precision_config['evidence_count']}, pred_threshold={high_precision_config['pred_threshold']}\n")
        f.write(f"   - 预期性能: P={high_precision_config['precision']:.3f}, R={high_precision_config['recall']:.3f}, F1={high_precision_config['f1_score']:.3f}\n\n")
        
        f.write("2. **高召回场景** (避免遗漏):\n")
        high_recall_config = df.iloc[df['recall'].idxmax()]
        f.write(f"   - 推荐配置: occ_thr={high_recall_config['occ_thr']}, evidence_count={high_recall_config['evidence_count']}, pred_threshold={high_recall_config['pred_threshold']}\n")
        f.write(f"   - 预期性能: P={high_recall_config['precision']:.3f}, R={high_recall_config['recall']:.3f}, F1={high_recall_config['f1_score']:.3f}\n\n")
        
        f.write("3. **平衡应用场景**:\n")
        f.write(f"   - 推荐配置: occ_thr={best_config['occ_thr']}, evidence_count={best_config['evidence_count']}, pred_threshold={best_config['pred_threshold']}\n")
        f.write(f"   - 预期性能: P={best_config['precision']:.3f}, R={best_config['recall']:.3f}, F1={best_config['f1_score']:.3f}\n\n")
        
        # 文件说明
        f.write("## 📁 输出文件说明\n\n")
        f.write("- `complete_results.csv`: 所有5,760个成功配置的详细结果\n")
        f.write("- `parameter_recommendations.json`: 推荐配置的JSON格式\n")
        f.write("- `visualizations/`: 可视化图表文件夹\n")
        f.write("  - `precision_recall_f1_3d.png`: 3D性能分布图\n")
        f.write("  - `parameter_heatmaps.png`: 参数组合热图\n")
        f.write("  - `parameter_sensitivity.png`: 参数敏感性分析\n")
        f.write("  - `pareto_frontier.png`: Precision-Recall权衡分析\n")
        f.write("  - `constraint_filtering.png`: 约束条件筛选结果\n\n")

def main():
    """主函数"""
    print("🔧 参数优化结果修复和可视化工具")
    print("=" * 60)
    
    # 查找最新的结果目录
    result_dir = "/mnt/d/Data/coda_PycharmProjects/PIN_bayesian/results/parameter_optimization_20250721_140722"
    
    if not os.path.exists(result_dir):
        print(f"❌ 结果目录不存在: {result_dir}")
        print("请检查目录路径是否正确")
        return
    
    print(f"📁 处理结果目录: {result_dir}")
    
    # 分析和可视化
    success = analyze_and_visualize_results(result_dir)
    
    if success:
        print(f"\n🎉 结果修复和可视化完成!")
        print(f"📁 结果目录: {result_dir}")
        print(f"📊 数据文件: complete_results.csv")
        print(f"💡 推荐配置: parameter_recommendations.json")
        print(f"📋 分析报告: complete_analysis_report.md")
        print(f"🎨 可视化图表: visualizations/")
        print(f"\n✨ 你现在可以用这些结果进行答辩了!")
    else:
        print(f"\n❌ 处理失败，请检查错误信息")

if __name__ == "__main__":
    main()