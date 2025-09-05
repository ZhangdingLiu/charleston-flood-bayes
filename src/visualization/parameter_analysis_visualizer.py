#!/usr/bin/env python3
"""
Parameter Analysis Visualizer
参数分析可视化模块

为贝叶斯网络洪水预测模型的参数优化结果创建各种可视化图表

作者：Claude AI
日期：2025-01-21
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
import os
import sys
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
warnings.filterwarnings('ignore')

# 确保sklearn可用
try:
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
except ImportError:
    print("⚠️ sklearn未安装，部分功能可能受限")
    MinMaxScaler = None
    StandardScaler = None

# 设置matplotlib中文字体和样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

class ParameterVisualizer:
    """参数分析可视化器"""
    
    def __init__(self, results_df, output_dir="visualizations"):
        """
        初始化可视化器
        
        Args:
            results_df (pd.DataFrame): 参数搜索结果DataFrame
            output_dir (str): 输出目录
        """
        self.df = results_df.copy()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置配色方案
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72', 
            'accent': '#F18F01',
            'success': '#C73E1D',
            'light': '#F5F5F5',
            'dark': '#333333'
        }
        
    def create_3d_performance_scatter(self, save_name="precision_recall_f1_3d.png"):
        """创建3D性能散点图 (Precision vs Recall vs F1)"""
        print("🎨 创建3D性能散点图...")
        
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # 创建颜色映射 - 根据F1分数上色
        scatter = ax.scatter(
            self.df['precision'], 
            self.df['recall'], 
            self.df['f1_score'],
            c=self.df['f1_score'], 
            cmap='viridis',
            alpha=0.7,
            s=50
        )
        
        # 设置标签和标题
        ax.set_xlabel('Precision', fontsize=12, labelpad=10)
        ax.set_ylabel('Recall', fontsize=12, labelpad=10)
        ax.set_zlabel('F1 Score', fontsize=12, labelpad=10)
        ax.set_title('Parameter Performance 3D Distribution\n(Precision vs Recall vs F1)', 
                     fontsize=14, pad=20)
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, aspect=30, pad=0.1)
        cbar.set_label('F1 Score', fontsize=10)
        
        # 添加网格
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息文本框
        stats_text = f"Total Combinations: {len(self.df)}\n"
        stats_text += f"Best F1: {self.df['f1_score'].max():.3f}\n"
        stats_text += f"Best Precision: {self.df['precision'].max():.3f}\n"
        stats_text += f"Best Recall: {self.df['recall'].max():.3f}"
        
        ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle='round', 
                 facecolor='white', alpha=0.8), fontsize=9)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"✅ 3D散点图已保存: {save_path}")
        
    def create_parameter_heatmaps(self, save_name="parameter_heatmaps.png"):
        """创建参数组合性能热图"""
        print("🎨 创建参数热图...")
        
        # 选择主要的网络构建参数进行热图显示
        param_pairs = [
            ('occ_thr', 'edge_thr'),
            ('occ_thr', 'weight_thr'),
            ('evidence_count', 'pred_threshold'),
            ('neg_pos_ratio', 'marginal_prob_threshold')
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()
        
        for idx, (param1, param2) in enumerate(param_pairs):
            ax = axes[idx]
            
            # 创建数据透视表
            pivot_data = self.df.pivot_table(
                values='f1_score', 
                index=param1, 
                columns=param2, 
                aggfunc='mean'
            )
            
            # 创建热图
            sns.heatmap(
                pivot_data, 
                annot=True, 
                fmt='.3f', 
                cmap='YlOrRd',
                cbar_kws={'label': 'F1 Score'},
                ax=ax,
                square=True
            )
            
            ax.set_title(f'F1 Score Heatmap: {param1} vs {param2}', fontsize=12)
            ax.set_xlabel(param2.replace('_', ' ').title(), fontsize=10)
            ax.set_ylabel(param1.replace('_', ' ').title(), fontsize=10)
        
        plt.suptitle('Parameter Combination Performance Heatmaps', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"✅ 参数热图已保存: {save_path}")
        
    def create_parameter_sensitivity_analysis(self, save_name="parameter_sensitivity.png"):
        """创建参数敏感性分析图"""
        print("🎨 创建参数敏感性分析图...")
        
        # 识别需要分析的参数
        numeric_params = ['occ_thr', 'edge_thr', 'weight_thr', 'evidence_count', 
                         'pred_threshold', 'neg_pos_ratio', 'marginal_prob_threshold']
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        axes = axes.ravel()
        
        for idx, param in enumerate(numeric_params):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            # 按参数值分组，计算平均性能
            param_performance = self.df.groupby(param).agg({
                'f1_score': ['mean', 'std'],
                'precision': 'mean',
                'recall': 'mean'
            }).round(4)
            
            param_performance.columns = ['f1_mean', 'f1_std', 'precision_mean', 'recall_mean']
            param_performance = param_performance.reset_index()
            
            # 绘制F1分数及其标准差
            ax.errorbar(
                param_performance[param], 
                param_performance['f1_mean'],
                yerr=param_performance['f1_std'],
                marker='o', 
                linewidth=2,
                capsize=5,
                capthick=2,
                label='F1 Score ± Std',
                color=self.colors['primary']
            )
            
            # 绘制精确度和召回率
            ax.plot(param_performance[param], param_performance['precision_mean'], 
                   'o--', label='Precision', color=self.colors['accent'], alpha=0.8)
            ax.plot(param_performance[param], param_performance['recall_mean'], 
                   's--', label='Recall', color=self.colors['secondary'], alpha=0.8)
            
            ax.set_xlabel(param.replace('_', ' ').title(), fontsize=10)
            ax.set_ylabel('Performance Score', fontsize=10)
            ax.set_title(f'Sensitivity Analysis: {param.replace("_", " ").title()}', fontsize=11)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
        # 移除多余的子图
        for idx in range(len(numeric_params), len(axes)):
            axes[idx].remove()
        
        plt.suptitle('Parameter Sensitivity Analysis', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"✅ 参数敏感性分析图已保存: {save_path}")
        
    def create_pareto_frontier_analysis(self, save_name="pareto_frontier.png"):
        """创建Pareto前沿分析 (Precision vs Recall权衡)"""
        print("🎨 创建Pareto前沿分析图...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # 左图: Precision vs Recall散点图
        scatter = ax1.scatter(
            self.df['recall'], 
            self.df['precision'],
            c=self.df['f1_score'], 
            cmap='viridis',
            alpha=0.7,
            s=50
        )
        
        ax1.set_xlabel('Recall', fontsize=12)
        ax1.set_ylabel('Precision', fontsize=12)
        ax1.set_title('Precision vs Recall Trade-off', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # 添加颜色条
        cbar1 = plt.colorbar(scatter, ax=ax1)
        cbar1.set_label('F1 Score', fontsize=10)
        
        # 添加等F1线
        recall_range = np.linspace(0, 1, 100)
        f1_levels = [0.5, 0.6, 0.7, 0.8, 0.9]
        
        for f1 in f1_levels:
            if f1 > 0:
                precision_line = f1 * recall_range / (2 * recall_range - f1)
                # 只绘制有效范围内的线
                valid_mask = (precision_line >= 0) & (precision_line <= 1) & (recall_range > f1/2)
                if np.any(valid_mask):
                    ax1.plot(recall_range[valid_mask], precision_line[valid_mask], 
                            '--', alpha=0.5, color='gray', linewidth=1)
                    
                    # 添加F1标签
                    if np.any(valid_mask):
                        mid_idx = len(recall_range[valid_mask]) // 2
                        if mid_idx < len(recall_range[valid_mask]) and mid_idx < len(precision_line[valid_mask]):
                            ax1.text(recall_range[valid_mask][mid_idx], 
                                   precision_line[valid_mask][mid_idx] + 0.02, 
                                   f'F1={f1}', fontsize=8, alpha=0.7)
        
        # 右图: 按约束条件筛选的结果
        # 定义几个常见的约束条件
        constraints = [
            {'name': 'High Precision (≥0.8)', 'condition': self.df['precision'] >= 0.8, 'color': self.colors['success']},
            {'name': 'High Recall (≥0.8)', 'condition': self.df['recall'] >= 0.8, 'color': self.colors['accent']},
            {'name': 'Balanced (P≥0.7, R≥0.7)', 'condition': (self.df['precision'] >= 0.7) & (self.df['recall'] >= 0.7), 'color': self.colors['primary']},
            {'name': 'High F1 (≥0.8)', 'condition': self.df['f1_score'] >= 0.8, 'color': self.colors['secondary']}
        ]
        
        for constraint in constraints:
            filtered_df = self.df[constraint['condition']]
            if len(filtered_df) > 0:
                ax2.scatter(
                    filtered_df['recall'], 
                    filtered_df['precision'],
                    label=f"{constraint['name']} (n={len(filtered_df)})",
                    alpha=0.7,
                    color=constraint['color'],
                    s=60
                )
        
        ax2.set_xlabel('Recall', fontsize=12)
        ax2.set_ylabel('Precision', fontsize=12)
        ax2.set_title('Configurations by Performance Constraints', fontsize=14)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"✅ Pareto前沿分析图已保存: {save_path}")
        
    def create_constraint_filtering_visualization(self, constraints, save_name="constraint_filtering.png"):
        """创建约束条件筛选可视化"""
        print("🎨 创建约束条件筛选可视化...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 应用约束条件
        constraint_mask = pd.Series([True] * len(self.df), index=self.df.index)
        
        if 'min_precision' in constraints:
            constraint_mask &= (self.df['precision'] >= constraints['min_precision'])
        if 'min_recall' in constraints:
            constraint_mask &= (self.df['recall'] >= constraints['min_recall'])
        if 'min_f1_score' in constraints:
            constraint_mask &= (self.df['f1_score'] >= constraints['min_f1_score'])
        if 'min_samples' in constraints:
            constraint_mask &= (self.df['total_samples'] >= constraints['min_samples'])
        
        filtered_df = self.df[constraint_mask]
        excluded_df = self.df[~constraint_mask]
        
        # 图1: 约束前后对比散点图
        ax1.scatter(excluded_df['recall'], excluded_df['precision'], 
                   alpha=0.3, color='lightgray', s=30, label=f'Excluded (n={len(excluded_df)})')
        ax1.scatter(filtered_df['recall'], filtered_df['precision'], 
                   alpha=0.8, color=self.colors['primary'], s=60, label=f'Satisfying Constraints (n={len(filtered_df)})')
        
        ax1.set_xlabel('Recall', fontsize=12)
        ax1.set_ylabel('Precision', fontsize=12)
        ax1.set_title('Constraint Filtering Results', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 添加约束线
        if 'min_precision' in constraints:
            ax1.axhline(y=constraints['min_precision'], color='red', linestyle='--', 
                       alpha=0.7, label=f"Min Precision = {constraints['min_precision']}")
        if 'min_recall' in constraints:
            ax1.axvline(x=constraints['min_recall'], color='red', linestyle='--', 
                       alpha=0.7, label=f"Min Recall = {constraints['min_recall']}")
        
        # 图2: F1分数分布对比
        bins = np.linspace(0, 1, 20)
        ax2.hist(self.df['f1_score'], bins=bins, alpha=0.5, color='lightgray', 
                label=f'All Configurations (n={len(self.df)})', density=True)
        if len(filtered_df) > 0:
            ax2.hist(filtered_df['f1_score'], bins=bins, alpha=0.8, 
                    color=self.colors['primary'], label=f'Satisfying Constraints (n={len(filtered_df)})', density=True)
        
        ax2.set_xlabel('F1 Score', fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)
        ax2.set_title('F1 Score Distribution', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 图3: 参数分布比较 (选择几个关键参数)
        key_params = ['occ_thr', 'pred_threshold', 'evidence_count']
        
        for i, param in enumerate(key_params):
            if i >= 1:  # 只显示一个参数的分布
                break
                
            param_counts_all = self.df[param].value_counts().sort_index()
            param_counts_filtered = filtered_df[param].value_counts().sort_index() if len(filtered_df) > 0 else pd.Series()
            
            x_pos = np.arange(len(param_counts_all))
            ax3.bar(x_pos - 0.2, param_counts_all.values, width=0.4, 
                   alpha=0.6, color='lightgray', label='All Configurations')
            
            if len(param_counts_filtered) > 0:
                # 确保索引对齐
                filtered_values = [param_counts_filtered.get(idx, 0) for idx in param_counts_all.index]
                ax3.bar(x_pos + 0.2, filtered_values, width=0.4, 
                       alpha=0.8, color=self.colors['primary'], label='Satisfying Constraints')
            
            ax3.set_xlabel(f'{param.replace("_", " ").title()}', fontsize=12)
            ax3.set_ylabel('Count', fontsize=12)
            ax3.set_title(f'{param.replace("_", " ").title()} Distribution', fontsize=14)
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(param_counts_all.index)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 图4: 约束满足情况总结
        constraint_info = []
        if 'min_precision' in constraints:
            precision_satisfied = (self.df['precision'] >= constraints['min_precision']).sum()
            constraint_info.append(f"Precision ≥ {constraints['min_precision']}: {precision_satisfied}/{len(self.df)} ({precision_satisfied/len(self.df)*100:.1f}%)")
        
        if 'min_recall' in constraints:
            recall_satisfied = (self.df['recall'] >= constraints['min_recall']).sum()
            constraint_info.append(f"Recall ≥ {constraints['min_recall']}: {recall_satisfied}/{len(self.df)} ({recall_satisfied/len(self.df)*100:.1f}%)")
        
        if 'min_f1_score' in constraints:
            f1_satisfied = (self.df['f1_score'] >= constraints['min_f1_score']).sum()
            constraint_info.append(f"F1 Score ≥ {constraints['min_f1_score']}: {f1_satisfied}/{len(self.df)} ({f1_satisfied/len(self.df)*100:.1f}%)")
        
        constraint_info.append(f"\nAll constraints satisfied: {len(filtered_df)}/{len(self.df)} ({len(filtered_df)/len(self.df)*100:.1f}%)")
        
        ax4.text(0.1, 0.7, '\n'.join(constraint_info), transform=ax4.transAxes, 
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        ax4.set_title('Constraint Satisfaction Summary', fontsize=14)
        ax4.axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"✅ 约束条件筛选可视化已保存: {save_path}")
        
        return filtered_df
    
    def generate_all_visualizations(self, constraints=None):
        """生成所有可视化图表"""
        print("🎨 开始生成所有可视化图表...")
        print("=" * 60)
        
        # 基础性能可视化
        self.create_3d_performance_scatter()
        self.create_parameter_heatmaps()
        self.create_parameter_sensitivity_analysis()
        self.create_pareto_frontier_analysis()
        
        # 如果提供了约束条件，创建约束筛选可视化
        filtered_df = None
        if constraints:
            filtered_df = self.create_constraint_filtering_visualization(constraints)
        
        print("=" * 60)
        print("🎉 所有可视化图表生成完成！")
        print(f"输出目录: {self.output_dir}")
        
        return filtered_df

def main():
    """主函数 - 演示用法"""
    # 这里应该加载真实的结果数据
    print("⚠️ 这是可视化模块的演示函数")
    print("实际使用时，请通过 run_parameter_optimization.py 调用")
    
    # 演示约束条件
    demo_constraints = {
        'min_precision': 0.8,
        'min_recall': 0.6,
        'min_f1_score': 0.7,
        'min_samples': 30
    }
    
    print(f"演示约束条件: {demo_constraints}")

if __name__ == "__main__":
    main()