#!/usr/bin/env python3
"""
Filtered Parameter Optimization Visualization Script

This script creates visualizations from parameter optimization results with filtering:
- Excludes configurations where precision = 1.0 or recall = 1.0 (overfitting indicators)
- Only includes configurations with negative_candidates_count >= 9 (sufficient samples)
- Generates the same 5 visualization types as the original system
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_and_filter_results(csv_path):
    """
    Load results and apply filtering criteria
    
    Filters:
    - negative_candidates_count >= 9 (sufficient negative samples)
    - precision < 1.0 (exclude perfect precision, likely overfitting)
    - recall < 1.0 (exclude perfect recall, likely overfitting)
    """
    print("Loading and filtering results...")
    df = pd.read_csv(csv_path)
    
    print(f"Original data shape: {df.shape}")
    
    # Apply filters
    filtered_df = df[
        (df['negative_candidates_count'] >= 9) &
        (df['precision'] < 1.0) &
        (df['recall'] < 1.0)
    ].copy()
    
    print(f"Filtered data shape: {filtered_df.shape}")
    print(f"Removed {len(df) - len(filtered_df)} configurations due to filtering")
    
    return filtered_df

def create_3d_performance_plot(df, output_dir):
    """Create 3D scatter plot of Precision vs Recall vs F1 Score"""
    print("Creating 3D performance visualization...")
    
    fig = plt.figure(figsize=(15, 6))
    
    # 3D scatter plot
    ax1 = fig.add_subplot(121, projection='3d')
    scatter = ax1.scatter(df['precision'], df['recall'], df['f1_score'], 
                         c=df['f1_score'], cmap='viridis', s=30, alpha=0.7)
    ax1.set_xlabel('Precision')
    ax1.set_ylabel('Recall')
    ax1.set_zlabel('F1 Score')
    ax1.set_title('3D Performance Distribution\n(Filtered Data)')
    plt.colorbar(scatter, ax=ax1, shrink=0.6)
    
    # 2D projection with color-coded F1
    ax2 = fig.add_subplot(122)
    scatter2 = ax2.scatter(df['precision'], df['recall'], 
                          c=df['f1_score'], cmap='viridis', s=30, alpha=0.7)
    ax2.set_xlabel('Precision')
    ax2.set_ylabel('Recall')
    ax2.set_title('Precision vs Recall\n(Color: F1 Score)')
    plt.colorbar(scatter2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'precision_recall_f1_3d_filtered.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_parameter_heatmaps(df, output_dir):
    """Create heatmaps showing parameter combination performance"""
    print("Creating parameter heatmaps...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Parameter Combination Performance Heatmaps\n(Filtered Data)', fontsize=16)
    
    # Heatmap 1: occ_thr vs edge_thr
    pivot1 = df.groupby(['occ_thr', 'edge_thr'])['f1_score'].mean().unstack()
    sns.heatmap(pivot1, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[0,0])
    axes[0,0].set_title('F1 Score Heatmap: occ_thr vs edge_thr')
    
    # Heatmap 2: occ_thr vs weight_thr
    pivot2 = df.groupby(['occ_thr', 'weight_thr'])['f1_score'].mean().unstack()
    sns.heatmap(pivot2, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[0,1])
    axes[0,1].set_title('F1 Score Heatmap: occ_thr vs weight_thr')
    
    # Heatmap 3: evidence_count vs pred_threshold
    pivot3 = df.groupby(['evidence_count', 'pred_threshold'])['f1_score'].mean().unstack()
    sns.heatmap(pivot3, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[1,0])
    axes[1,0].set_title('F1 Score Heatmap: evidence_count vs pred_threshold')
    
    # Heatmap 4: neg_pos_ratio vs marginal_prob_threshold
    pivot4 = df.groupby(['neg_pos_ratio', 'marginal_prob_threshold'])['f1_score'].mean().unstack()
    sns.heatmap(pivot4, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[1,1])
    axes[1,1].set_title('F1 Score Heatmap: neg_pos_ratio vs marginal_prob_threshold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'parameter_heatmaps_filtered.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_parameter_sensitivity_analysis(df, output_dir):
    """Create parameter sensitivity analysis plots"""
    print("Creating parameter sensitivity analysis...")
    
    params = ['occ_thr', 'edge_thr', 'weight_thr', 'evidence_count', 
              'pred_threshold', 'neg_pos_ratio', 'marginal_prob_threshold']
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Parameter Sensitivity Analysis\n(Filtered Data)', fontsize=16)
    axes = axes.flatten()
    
    for i, param in enumerate(params):
        if i >= len(axes):
            break
            
        # Group by parameter and calculate statistics
        grouped = df.groupby(param).agg({
            'precision': ['mean', 'std'],
            'recall': ['mean', 'std'],
            'f1_score': ['mean', 'std']
        }).reset_index()
        
        # Flatten column names
        grouped.columns = [param, 'precision_mean', 'precision_std',
                          'recall_mean', 'recall_std', 'f1_mean', 'f1_std']
        
        # Plot with error bars
        x = grouped[param]
        axes[i].errorbar(x, grouped['precision_mean'], yerr=grouped['precision_std'], 
                        label='Precision', marker='o', capsize=5)
        axes[i].errorbar(x, grouped['recall_mean'], yerr=grouped['recall_std'], 
                        label='Recall', marker='s', capsize=5)
        axes[i].errorbar(x, grouped['f1_mean'], yerr=grouped['f1_std'], 
                        label='F1 Score', marker='^', capsize=5)
        
        axes[i].set_xlabel(f'{param}')
        axes[i].set_ylabel('Performance Score')
        axes[i].set_title(f'Sensitivity Analysis: {param}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    # Remove unused subplot
    if len(params) < len(axes):
        fig.delaxes(axes[-1])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'parameter_sensitivity_filtered.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_pareto_frontier_analysis(df, output_dir):
    """Create Pareto frontier analysis for precision-recall trade-off"""
    print("Creating Pareto frontier analysis...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left plot: Precision vs Recall colored by F1
    scatter1 = ax1.scatter(df['recall'], df['precision'], 
                          c=df['f1_score'], cmap='viridis', s=30, alpha=0.7)
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_title('Precision vs Recall Trade-off\n(Filtered Data)')
    plt.colorbar(scatter1, ax=ax1, label='F1 Score')
    
    # Right plot: Configurations by performance constraints
    # Define constraint categories
    high_precision = df['precision'] >= 0.8
    high_recall = df['recall'] >= 0.8
    high_f1 = df['f1_score'] >= 0.7
    
    # Create color mapping
    colors = []
    labels = []
    for _, row in df.iterrows():
        if row['precision'] >= 0.8 and row['recall'] >= 0.8 and row['f1_score'] >= 0.7:
            colors.append('red')
            if 'All High (P≥0.8, R≥0.8, F1≥0.7)' not in labels:
                labels.append('All High (P≥0.8, R≥0.8, F1≥0.7)')
        elif row['precision'] >= 0.8:
            colors.append('orange')
            if 'High Precision (≥0.8)' not in labels:
                labels.append('High Precision (≥0.8)')
        elif row['recall'] >= 0.8:
            colors.append('purple')
            if 'High Recall (≥0.8)' not in labels:
                labels.append('High Recall (≥0.8)')
        elif row['f1_score'] >= 0.7:
            colors.append('blue')
            if 'High F1 (≥0.7)' not in labels:
                labels.append('High F1 (≥0.7)')
        else:
            colors.append('gray')
            if 'Others' not in labels:
                labels.append('Others')
    
    ax2.scatter(df['recall'], df['precision'], c=colors, s=30, alpha=0.7)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Configurations by Performance Constraints')
    
    # Add legend
    unique_colors = list(set(colors))
    for color in unique_colors:
        mask = [c == color for c in colors]
        if color == 'red':
            label = 'All High (P≥0.8, R≥0.8, F1≥0.7)'
        elif color == 'orange':
            label = 'High Precision (≥0.8)'
        elif color == 'purple':
            label = 'High Recall (≥0.8)'
        elif color == 'blue':
            label = 'High F1 (≥0.7)'
        else:
            label = 'Others'
        ax2.scatter([], [], c=color, label=label, s=50)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pareto_frontier_filtered.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_constraint_filtering_analysis(df, output_dir):
    """Create constraint filtering analysis visualization"""
    print("Creating constraint filtering analysis...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Constraint Filtering Results\n(Filtered Data)', fontsize=16)
    
    # Define constraints
    precision_constraint = df['precision'] >= 0.8
    recall_constraint = df['recall'] >= 0.8
    f1_constraint = df['f1_score'] >= 0.7
    all_constraints = precision_constraint & recall_constraint & f1_constraint
    
    # Plot 1: Scatter plot showing constraint satisfaction
    colors = np.where(all_constraints, 'blue', 'lightgray')
    ax1.scatter(df['recall'], df['precision'], c=colors, s=30, alpha=0.7)
    ax1.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Precision ≥ 0.8')
    ax1.axvline(x=0.8, color='red', linestyle='--', alpha=0.7, label='Recall ≥ 0.8')
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_title('Constraint Filtering Results')
    ax1.legend()
    
    # Plot 2: F1 Score Distribution
    ax2.hist(df['f1_score'], bins=30, alpha=0.7, color='skyblue', 
             label=f'All Configurations (n={len(df)})')
    ax2.hist(df[all_constraints]['f1_score'], bins=15, alpha=0.7, color='darkblue', 
             label=f'Satisfying Constraints (n={sum(all_constraints)})')
    ax2.axvline(x=0.7, color='red', linestyle='--', alpha=0.7, label='F1 ≥ 0.7')
    ax2.set_xlabel('F1 Score')
    ax2.set_ylabel('Count')
    ax2.set_title('F1 Score Distribution')
    ax2.legend()
    
    # Plot 3: Negative Candidates Distribution
    ax3.hist(df['negative_candidates_count'], bins=20, alpha=0.7, color='lightblue', 
             label='All Configurations')
    ax3.hist(df[all_constraints]['negative_candidates_count'], bins=10, alpha=0.7, 
             color='darkblue', label='Satisfying Constraints')
    ax3.axvline(x=9, color='red', linestyle='--', alpha=0.7, 
                label='Min Negative Candidates = 9')
    ax3.set_xlabel('Negative Candidates Count')
    ax3.set_ylabel('Count')
    ax3.set_title('Negative Candidates Distribution')
    ax3.legend()
    
    # Plot 4: Constraint Satisfaction Summary
    precision_count = sum(precision_constraint)
    recall_count = sum(recall_constraint)  
    f1_count = sum(f1_constraint)
    all_count = sum(all_constraints)
    
    summary_text = f"""
    Precision ≥ 0.8: {precision_count}/{len(df)} ({precision_count/len(df)*100:.1f}%)
    Recall ≥ 0.8: {recall_count}/{len(df)} ({recall_count/len(df)*100:.1f}%)
    F1 Score ≥ 0.7: {f1_count}/{len(df)} ({f1_count/len(df)*100:.1f}%)
    All constraints satisfied: {all_count}/{len(df)} ({all_count/len(df)*100:.1f}%)
    """
    
    ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_title('Constraint Satisfaction Summary')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'constraint_filtering_filtered.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_summary_report(df, output_dir):
    """Generate summary report of filtered results"""
    
    # Basic statistics
    total_configs = len(df)
    best_f1_idx = df['f1_score'].idxmax()
    best_config = df.loc[best_f1_idx]
    
    # Constraint analysis
    precision_constraint = df['precision'] >= 0.8
    recall_constraint = df['recall'] >= 0.8
    f1_constraint = df['f1_score'] >= 0.7
    all_constraints = precision_constraint & recall_constraint & f1_constraint
    
    report = f"""# Filtered Parameter Optimization Results Summary

## Filtering Criteria Applied
- negative_candidates_count >= 9 (sufficient negative samples)
- precision < 1.0 (exclude perfect precision, likely overfitting)
- recall < 1.0 (exclude perfect recall, likely overfitting)

## Overall Statistics
- **Total filtered configurations**: {total_configs}
- **F1 Score range**: {df['f1_score'].min():.3f} - {df['f1_score'].max():.3f}
- **Precision range**: {df['precision'].min():.3f} - {df['precision'].max():.3f}
- **Recall range**: {df['recall'].min():.3f} - {df['recall'].max():.3f}

## Best Configuration (Highest F1)
- **F1 Score**: {best_config['f1_score']:.3f}
- **Precision**: {best_config['precision']:.3f}
- **Recall**: {best_config['recall']:.3f}
- **Parameters**: 
  - occ_thr: {best_config['occ_thr']}
  - edge_thr: {best_config['edge_thr']}
  - weight_thr: {best_config['weight_thr']}
  - evidence_count: {best_config['evidence_count']}
  - pred_threshold: {best_config['pred_threshold']}
  - neg_pos_ratio: {best_config['neg_pos_ratio']}
  - marginal_prob_threshold: {best_config['marginal_prob_threshold']}
- **Negative candidates**: {best_config['negative_candidates_count']}

## Constraint Satisfaction Analysis
- **Precision ≥ 0.8**: {sum(precision_constraint)} / {total_configs} ({sum(precision_constraint)/total_configs*100:.1f}%)
- **Recall ≥ 0.8**: {sum(recall_constraint)} / {total_configs} ({sum(recall_constraint)/total_configs*100:.1f}%)
- **F1 Score ≥ 0.7**: {sum(f1_constraint)} / {total_configs} ({sum(f1_constraint)/total_configs*100:.1f}%)
- **All constraints satisfied**: {sum(all_constraints)} / {total_configs} ({sum(all_constraints)/total_configs*100:.1f}%)

## Generated Visualizations
1. **precision_recall_f1_3d_filtered.png** - 3D performance distribution
2. **parameter_heatmaps_filtered.png** - Parameter combination performance heatmaps
3. **parameter_sensitivity_filtered.png** - Parameter sensitivity analysis
4. **pareto_frontier_filtered.png** - Precision-recall trade-off analysis
5. **constraint_filtering_filtered.png** - Constraint filtering results

This filtered analysis provides a more robust view of parameter performance by excluding 
configurations that may have achieved perfect scores due to insufficient data or 
overly restrictive parameter settings.
"""
    
    with open(output_dir / 'FILTERED_ANALYSIS_SUMMARY.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Summary report saved to: {output_dir / 'FILTERED_ANALYSIS_SUMMARY.md'}")

def main():
    """Main function to generate filtered visualizations"""
    
    # Paths
    results_dir = Path("results/parameter_optimization_20250721_140722")
    csv_path = results_dir / "complete_results.csv"
    output_dir = results_dir / "filtered_visualizations"
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading results from: {csv_path}")
    print(f"Output directory: {output_dir}")
    
    # Load and filter data
    df = load_and_filter_results(csv_path)
    
    if len(df) == 0:
        print("❌ No data remaining after filtering!")
        return
    
    # Generate visualizations
    create_3d_performance_plot(df, output_dir)
    create_parameter_heatmaps(df, output_dir)
    create_parameter_sensitivity_analysis(df, output_dir)
    create_pareto_frontier_analysis(df, output_dir)
    create_constraint_filtering_analysis(df, output_dir)
    
    # Generate summary report
    generate_summary_report(df, output_dir)
    
    print("\n✅ Filtered visualizations completed successfully!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"📊 Processed {len(df)} filtered configurations")
    
    # Show best configuration
    best_idx = df['f1_score'].idxmax()
    best_config = df.loc[best_idx]
    print(f"\n🏆 Best filtered configuration (F1={best_config['f1_score']:.3f}):")
    print(f"   - Precision: {best_config['precision']:.3f}")
    print(f"   - Recall: {best_config['recall']:.3f}")
    print(f"   - Negative candidates: {best_config['negative_candidates_count']}")

if __name__ == "__main__":
    main()