#!/usr/bin/env python3
"""
Run Parameter Optimization
参数优化主控脚本

一键运行贝叶斯网络洪水预测模型的完整参数优化和可视化流程

功能：
- 执行全参数空间网格搜索
- 基于用户约束条件筛选最佳参数
- 生成全面的可视化分析报告
- 提供多种优化策略的参数推荐

使用方法：
    python run_parameter_optimization.py

自定义约束条件：
    在main函数中修改constraints字典

作者：Claude AI
日期：2025-01-21
"""

import sys
import os
import pandas as pd
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径以便导入模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'analysis'))
sys.path.append(os.path.join(current_dir, 'visualization'))

try:
    from analysis.comprehensive_parameter_grid_search import ParameterGridSearcher
    from visualization.parameter_analysis_visualizer import ParameterVisualizer
except ImportError as e:
    print(f"❌ 模块导入失败: {e}")
    print("请确保analysis和visualization目录存在且包含相应的.py文件")
    sys.exit(1)

class ParameterOptimizer:
    """参数优化器主类"""
    
    def __init__(self, constraints=None, param_grid=None):
        """
        初始化参数优化器
        
        Args:
            constraints (dict): 约束条件字典
            param_grid (dict): 参数网格，None则使用默认
        """
        self.constraints = constraints or {}
        self.param_grid = param_grid
        self.results_df = None
        self.filtered_df = None
        self.result_dir = None
        
    def run_optimization(self):
        """运行完整的参数优化流程"""
        print("🚀 启动贝叶斯网络参数优化流程")
        print("=" * 80)
        
        start_time = datetime.now()
        
        # 步骤1: 执行网格搜索
        print("\n📊 步骤1: 执行参数网格搜索")
        print("-" * 50)
        
        searcher = ParameterGridSearcher(param_grid=self.param_grid)
        results, result_dir = searcher.run_grid_search()
        
        self.results_df = pd.DataFrame(results)
        self.result_dir = result_dir
        
        if len(results) == 0:
            print("❌ 网格搜索失败，没有成功的参数组合")
            return
        
        print(f"✅ 网格搜索完成，成功评估 {len(results)} 个参数组合")
        
        # 步骤2: 应用约束条件筛选
        print(f"\n🎯 步骤2: 应用约束条件筛选")
        print("-" * 50)
        
        self.filtered_df = self.apply_constraints()
        
        # 步骤3: 生成参数推荐
        print(f"\n💡 步骤3: 生成参数推荐")
        print("-" * 50)
        
        recommendations = self.generate_recommendations()
        
        # 步骤4: 创建可视化
        print(f"\n🎨 步骤4: 生成可视化分析")
        print("-" * 50)
        
        self.create_visualizations()
        
        # 步骤5: 生成综合报告
        print(f"\n📝 步骤5: 生成优化报告")
        print("-" * 50)
        
        self.generate_optimization_report(recommendations)
        
        # 总结
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "=" * 80)
        print("🎉 参数优化流程完成！")
        print(f"总耗时: {duration}")
        print(f"结果目录: {self.result_dir}")
        print(f"成功评估: {len(results)} 个参数组合")
        print(f"满足约束: {len(self.filtered_df) if self.filtered_df is not None else 0} 个组合")
        print("=" * 80)
        
        return self.result_dir, recommendations
    
    def apply_constraints(self):
        """应用约束条件筛选参数组合"""
        if not self.constraints:
            print("⚠️ 未设置约束条件，返回所有结果")
            return self.results_df
        
        print("应用的约束条件:")
        for key, value in self.constraints.items():
            print(f"  - {key}: {value}")
        
        # 初始化筛选掩码
        mask = pd.Series([True] * len(self.results_df), index=self.results_df.index)
        
        # 应用各种约束条件
        if 'min_precision' in self.constraints:
            precision_mask = self.results_df['precision'] >= self.constraints['min_precision']
            mask &= precision_mask
            print(f"精确度约束 (≥{self.constraints['min_precision']}): {precision_mask.sum()}/{len(self.results_df)} 满足")
        
        if 'min_recall' in self.constraints:
            recall_mask = self.results_df['recall'] >= self.constraints['min_recall']
            mask &= recall_mask
            print(f"召回率约束 (≥{self.constraints['min_recall']}): {recall_mask.sum()}/{len(self.results_df)} 满足")
        
        if 'min_f1_score' in self.constraints:
            f1_mask = self.results_df['f1_score'] >= self.constraints['min_f1_score']
            mask &= f1_mask
            print(f"F1分数约束 (≥{self.constraints['min_f1_score']}): {f1_mask.sum()}/{len(self.results_df)} 满足")
        
        if 'min_samples' in self.constraints:
            samples_mask = self.results_df['total_samples'] >= self.constraints['min_samples']
            mask &= samples_mask
            print(f"样本数约束 (≥{self.constraints['min_samples']}): {samples_mask.sum()}/{len(self.results_df)} 满足")
        
        if 'min_accuracy' in self.constraints:
            accuracy_mask = self.results_df['accuracy'] >= self.constraints['min_accuracy']
            mask &= accuracy_mask
            print(f"准确率约束 (≥{self.constraints['min_accuracy']}): {accuracy_mask.sum()}/{len(self.results_df)} 满足")
        
        filtered_df = self.results_df[mask].copy()
        
        print(f"\n筛选结果:")
        print(f"  原始组合数: {len(self.results_df)}")
        print(f"  满足约束组合数: {len(filtered_df)}")
        print(f"  筛选率: {len(filtered_df)/len(self.results_df)*100:.1f}%")
        
        if len(filtered_df) == 0:
            print("⚠️ 没有参数组合满足所有约束条件")
            print("建议：")
            print("  1. 放宽约束条件")
            print("  2. 检查数据质量")
            print("  3. 调整参数搜索范围")
        
        return filtered_df
    
    def generate_recommendations(self):
        """生成参数推荐"""
        recommendations = {}
        
        # 基于所有结果的推荐
        if len(self.results_df) > 0:
            all_best_f1 = self.results_df.loc[self.results_df['f1_score'].idxmax()]
            all_best_precision = self.results_df.loc[self.results_df['precision'].idxmax()]
            all_best_recall = self.results_df.loc[self.results_df['recall'].idxmax()]
            
            recommendations['overall'] = {
                'best_f1': all_best_f1.to_dict(),
                'best_precision': all_best_precision.to_dict(),
                'best_recall': all_best_recall.to_dict()
            }
        
        # 基于约束条件筛选结果的推荐
        if self.filtered_df is not None and len(self.filtered_df) > 0:
            # 最佳F1分数
            best_f1_config = self.filtered_df.loc[self.filtered_df['f1_score'].idxmax()]
            
            # 最高精确度
            best_precision_config = self.filtered_df.loc[self.filtered_df['precision'].idxmax()]
            
            # 最高召回率
            best_recall_config = self.filtered_df.loc[self.filtered_df['recall'].idxmax()]
            
            # 最平衡的配置 (精确度和召回率最接近)
            self.filtered_df['balance_score'] = 1 - abs(self.filtered_df['precision'] - self.filtered_df['recall'])
            best_balanced_config = self.filtered_df.loc[self.filtered_df['balance_score'].idxmax()]
            
            # 计算鲁棒性分数 (基于参数的标准化距离)
            param_cols = ['occ_thr', 'edge_thr', 'weight_thr', 'evidence_count', 'pred_threshold']
            try:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                param_scaled = scaler.fit_transform(self.filtered_df[param_cols])
            except ImportError:
                print("⚠️ sklearn未安装，使用简化的鲁棒性计算")
                # 简化的标准化方法
                param_data = self.filtered_df[param_cols].values
                param_scaled = (param_data - param_data.mean(axis=0)) / (param_data.std(axis=0) + 1e-8)
            
            # 找到参数空间中心附近的高性能配置
            center = param_scaled.mean(axis=0)
            distances = ((param_scaled - center) ** 2).sum(axis=1)
            
            # 结合性能和参数稳定性的综合分数
            performance_weight = 0.7
            stability_weight = 0.3
            
            normalized_f1 = (self.filtered_df['f1_score'] - self.filtered_df['f1_score'].min()) / (
                self.filtered_df['f1_score'].max() - self.filtered_df['f1_score'].min() + 1e-8)
            normalized_distance = 1 - (distances - distances.min()) / (distances.max() - distances.min() + 1e-8)
            
            robust_scores = performance_weight * normalized_f1 + stability_weight * normalized_distance
            best_robust_idx = robust_scores.idxmax()
            best_robust_config = self.filtered_df.loc[best_robust_idx]
            
            recommendations['constrained'] = {
                'best_f1': best_f1_config.to_dict(),
                'best_precision': best_precision_config.to_dict(),
                'best_recall': best_recall_config.to_dict(),
                'best_balanced': best_balanced_config.to_dict(),
                'best_robust': best_robust_config.to_dict()
            }
            
            print("🏆 基于约束条件的推荐配置:")
            print(f"最佳F1分数配置 (F1={best_f1_config['f1_score']:.3f}):")
            self._print_config(best_f1_config)
            
            print(f"\n最高精确度配置 (P={best_precision_config['precision']:.3f}):")
            self._print_config(best_precision_config)
            
            print(f"\n最高召回率配置 (R={best_recall_config['recall']:.3f}):")
            self._print_config(best_recall_config)
            
            print(f"\n最平衡配置 (P={best_balanced_config['precision']:.3f}, R={best_balanced_config['recall']:.3f}):")
            self._print_config(best_balanced_config)
            
            print(f"\n最鲁棒配置 (综合分数={robust_scores.max():.3f}):")
            self._print_config(best_robust_config)
        
        else:
            print("⚠️ 没有满足约束条件的配置，无法生成约束推荐")
            recommendations['constrained'] = None
        
        # 保存推荐结果
        recommendations_file = os.path.join(self.result_dir, "parameter_recommendations.json")
        with open(recommendations_file, 'w') as f:
            # 转换numpy类型为Python原生类型以便JSON序列化
            import numpy as np
            def convert_numpy(obj):
                if isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_numpy(item) for item in obj]
                elif isinstance(obj, (np.integer, np.floating)):
                    return obj.item()
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return obj
            
            json.dump(convert_numpy(recommendations), f, indent=2, ensure_ascii=False)
        
        print(f"💾 参数推荐已保存: {recommendations_file}")
        
        return recommendations
    
    def _print_config(self, config):
        """打印参数配置的辅助函数"""
        param_keys = ['occ_thr', 'edge_thr', 'weight_thr', 'evidence_count', 
                     'pred_threshold', 'neg_pos_ratio', 'marginal_prob_threshold']
        
        config_str = ", ".join([f"{key}={config[key]}" for key in param_keys if key in config])
        print(f"  参数: {config_str}")
        print(f"  性能: P={config['precision']:.3f}, R={config['recall']:.3f}, F1={config['f1_score']:.3f}")
    
    def create_visualizations(self):
        """创建可视化"""
        if self.results_df is None or len(self.results_df) == 0:
            print("❌ 没有结果数据，跳过可视化")
            return
        
        # 创建可视化目录
        viz_dir = os.path.join(self.result_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # 创建可视化器
        visualizer = ParameterVisualizer(self.results_df, viz_dir)
        
        # 生成所有可视化
        visualizer.generate_all_visualizations(constraints=self.constraints)
    
    def generate_optimization_report(self, recommendations):
        """生成优化报告"""
        report_file = os.path.join(self.result_dir, "optimization_report.md")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 贝叶斯网络洪水预测模型参数优化报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 实验概述
            f.write("## 实验概述\n\n")
            f.write(f"- **总参数组合数**: {len(self.results_df)}\n")
            f.write(f"- **满足约束组合数**: {len(self.filtered_df) if self.filtered_df is not None else 0}\n")
            f.write(f"- **筛选成功率**: {len(self.filtered_df)/len(self.results_df)*100:.1f}%\n\n" if self.filtered_df is not None else "")
            
            # 约束条件
            f.write("## 约束条件\n\n")
            if self.constraints:
                for key, value in self.constraints.items():
                    constraint_name = {
                        'min_precision': '最小精确度',
                        'min_recall': '最小召回率', 
                        'min_f1_score': '最小F1分数',
                        'min_samples': '最小样本数',
                        'min_accuracy': '最小准确率'
                    }.get(key, key)
                    f.write(f"- **{constraint_name}**: {value}\n")
            else:
                f.write("未设置约束条件\n")
            f.write("\n")
            
            # 整体性能统计
            f.write("## 整体性能统计\n\n")
            f.write("| 指标 | 最小值 | 最大值 | 平均值 | 标准差 |\n")
            f.write("|------|--------|--------|--------|--------|\n")
            
            metrics = ['precision', 'recall', 'f1_score', 'accuracy']
            for metric in metrics:
                if metric in self.results_df.columns:
                    min_val = self.results_df[metric].min()
                    max_val = self.results_df[metric].max()
                    mean_val = self.results_df[metric].mean()
                    std_val = self.results_df[metric].std()
                    
                    f.write(f"| {metric.replace('_', ' ').title()} | {min_val:.3f} | {max_val:.3f} | {mean_val:.3f} | {std_val:.3f} |\n")
            f.write("\n")
            
            # 推荐配置
            if 'constrained' in recommendations and recommendations['constrained']:
                f.write("## 推荐参数配置\n\n")
                
                configs = [
                    ('最佳F1分数配置', 'best_f1'),
                    ('最高精确度配置', 'best_precision'),
                    ('最高召回率配置', 'best_recall'),
                    ('最平衡配置', 'best_balanced'),
                    ('最鲁棒配置', 'best_robust')
                ]
                
                for config_name, config_key in configs:
                    if config_key in recommendations['constrained']:
                        config = recommendations['constrained'][config_key]
                        f.write(f"### {config_name}\n\n")
                        f.write(f"**性能指标**:\n")
                        f.write(f"- 精确度 (Precision): {config['precision']:.3f}\n")
                        f.write(f"- 召回率 (Recall): {config['recall']:.3f}\n")
                        f.write(f"- F1分数: {config['f1_score']:.3f}\n")
                        f.write(f"- 准确率: {config['accuracy']:.3f}\n\n")
                        
                        f.write(f"**网络参数**:\n")
                        f.write(f"- occ_thr (道路最小出现次数): {config['occ_thr']}\n")
                        f.write(f"- edge_thr (边最小共现次数): {config['edge_thr']}\n")
                        f.write(f"- weight_thr (边权重阈值): {config['weight_thr']}\n\n")
                        
                        f.write(f"**评估参数**:\n")
                        f.write(f"- evidence_count (证据道路数): {config['evidence_count']}\n")
                        f.write(f"- pred_threshold (预测阈值): {config['pred_threshold']}\n")
                        f.write(f"- neg_pos_ratio (负正样本比): {config['neg_pos_ratio']}\n")
                        f.write(f"- marginal_prob_threshold (边际概率阈值): {config['marginal_prob_threshold']}\n\n")
            
            # 使用建议
            f.write("## 使用建议\n\n")
            f.write("1. **高精度场景**: 如果需要减少误报，优先使用'最高精确度配置'\n")
            f.write("2. **高召回场景**: 如果不能遗漏真实洪水事件，使用'最高召回率配置'\n")
            f.write("3. **平衡应用**: 对于一般应用场景，推荐使用'最佳F1分数配置'\n")
            f.write("4. **生产部署**: 考虑到系统稳定性，推荐使用'最鲁棒配置'\n\n")
            
            # 文件说明
            f.write("## 输出文件说明\n\n")
            f.write("- `complete_results.csv`: 所有参数组合的详细结果\n")
            f.write("- `parameter_recommendations.json`: 推荐参数配置的JSON格式\n")
            f.write("- `visualizations/`: 可视化图表文件夹\n")
            f.write("  - `precision_recall_f1_3d.png`: 3D性能分布图\n")
            f.write("  - `parameter_heatmaps.png`: 参数组合热图\n")
            f.write("  - `parameter_sensitivity.png`: 参数敏感性分析\n")
            f.write("  - `pareto_frontier.png`: Precision-Recall权衡分析\n")
            f.write("  - `constraint_filtering.png`: 约束条件筛选结果\n\n")
        
        print(f"📋 优化报告已生成: {report_file}")

def main():
    """主函数"""
    print("🎯 贝叶斯网络洪水预测模型参数优化系统")
    print("=" * 80)
    print("本系统将执行完整的参数网格搜索和性能分析")
    print("预计搜索 4,320 个参数组合 (4×3×4×4×5×3×3)")
    print("=" * 80)
    
    # 用户可自定义的约束条件
    # 💡 这里可以根据需要修改约束条件
    constraints = {
        'min_precision': 0.8,    # 精确度要求 ≥ 0.8
        'min_recall': 0.8,       # 召回率要求 ≥ 0.8  
        'min_f1_score': 0.7,     # F1分数要求 ≥ 0.7
        'min_samples': 30        # 最小测试样本数 ≥ 30
    }
    
    # 显示约束条件
    print("🎯 设定的约束条件:")
    for key, value in constraints.items():
        constraint_desc = {
            'min_precision': '精确度',
            'min_recall': '召回率',
            'min_f1_score': 'F1分数', 
            'min_samples': '测试样本数',
            'min_accuracy': '准确率'
        }.get(key, key)
        print(f"  - {constraint_desc} ≥ {value}")
    print()
    
    # 询问用户是否继续
    user_input = input("是否继续执行参数优化？这可能需要较长时间 (y/n): ").lower()
    if user_input not in ['y', 'yes', '是', '']:
        print("操作已取消")
        return
    
    # 创建优化器并运行
    optimizer = ParameterOptimizer(constraints=constraints)
    
    try:
        result_dir, recommendations = optimizer.run_optimization()
        
        # 显示最终结果路径
        print(f"\n🎉 优化完成！请查看以下文件:")
        print(f"📁 结果目录: {result_dir}")
        print(f"📊 完整结果: {result_dir}/complete_results.csv")
        print(f"💡 参数推荐: {result_dir}/parameter_recommendations.json") 
        print(f"📋 分析报告: {result_dir}/optimization_report.md")
        print(f"🎨 可视化图表: {result_dir}/visualizations/")
        
        return optimizer, result_dir, recommendations
        
    except Exception as e:
        print(f"❌ 优化过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = main()