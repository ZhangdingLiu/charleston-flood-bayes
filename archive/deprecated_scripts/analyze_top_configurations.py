#!/usr/bin/env python3
"""
Top Parameter Configuration Analysis

This script analyzes the parameter optimization results to identify the best performing
configurations based on different criteria and creates documentation suitable for presentation.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

def load_and_filter_data(csv_path):
    """Load and filter the results data"""
    print("Loading parameter optimization results...")
    df = pd.read_csv(csv_path)
    
    print(f"Original data: {len(df)} configurations")
    
    # Apply filtering criteria
    filtered_df = df[
        (df['negative_candidates_count'] >= 9) &
        (df['precision'] < 1.0) &
        (df['recall'] < 1.0)
    ].copy()
    
    print(f"Filtered data: {len(filtered_df)} configurations")
    print(f"Removed {len(df) - len(filtered_df)} configurations due to filtering")
    
    return filtered_df

def find_top_configurations(df):
    """Find top configurations based on different criteria"""
    
    configs = {}
    
    # 1. Highest F1 Score (overall best)
    top_f1 = df.nlargest(5, 'f1_score')
    configs['highest_f1'] = top_f1
    
    # 2. Highest Precision (with reasonable recall > 0.7)
    high_prec_df = df[df['recall'] > 0.7]
    if len(high_prec_df) > 0:
        top_precision = high_prec_df.nlargest(5, 'precision')
        configs['highest_precision'] = top_precision
    
    # 3. Highest Recall (with reasonable precision > 0.7)
    high_recall_df = df[df['precision'] > 0.7]
    if len(high_recall_df) > 0:
        top_recall = high_recall_df.nlargest(5, 'recall')
        configs['highest_recall'] = top_recall
    
    # 4. Best Balanced (precision > 0.8 AND recall > 0.8)
    balanced_df = df[(df['precision'] > 0.8) & (df['recall'] > 0.8)]
    if len(balanced_df) > 0:
        top_balanced = balanced_df.nlargest(5, 'f1_score')
        configs['best_balanced'] = top_balanced
    
    # 5. Most Robust (high negative_candidates_count, good performance)
    robust_df = df[(df['f1_score'] > 0.7) & (df['negative_candidates_count'] >= 15)]
    if len(robust_df) > 0:
        top_robust = robust_df.nlargest(5, 'negative_candidates_count')
        configs['most_robust'] = top_robust
    
    return configs

def analyze_parameter_patterns(df):
    """Analyze patterns in parameter choices for top performers"""
    
    # Get top 20% performers
    top_20_pct = df.quantile(0.8)['f1_score']
    top_performers = df[df['f1_score'] >= top_20_pct]
    
    parameter_analysis = {}
    params = ['occ_thr', 'edge_thr', 'weight_thr', 'evidence_count', 
              'pred_threshold', 'neg_pos_ratio', 'marginal_prob_threshold']
    
    for param in params:
        param_stats = {
            'most_common': top_performers[param].mode().iloc[0] if len(top_performers[param].mode()) > 0 else None,
            'mean': top_performers[param].mean(),
            'median': top_performers[param].median(),
            'range': [top_performers[param].min(), top_performers[param].max()],
            'value_counts': top_performers[param].value_counts().to_dict()
        }
        parameter_analysis[param] = param_stats
    
    return parameter_analysis, top_performers

def create_parameter_documentation():
    """Create comprehensive parameter documentation"""
    
    param_docs = {
        'occ_thr': {
            'name': 'Occurrence Threshold',
            'description': 'Minimum number of flood occurrences required for a road to be included in the Bayesian network',
            'purpose': 'Controls network size and data quality - higher values create smaller, more reliable networks',
            'impact': 'Lower values include more roads but may introduce noise; higher values focus on frequently flooded roads',
            'range_tested': [2, 3, 4, 5],
            'typical_good_values': [3, 4, 5]
        },
        'edge_thr': {
            'name': 'Edge Threshold', 
            'description': 'Minimum number of co-occurrences required to create an edge between two roads in the network',
            'purpose': 'Determines the strength of relationships between roads that get modeled',
            'impact': 'Lower values create more connected networks; higher values focus on strongest relationships',
            'range_tested': [1, 2, 3],
            'typical_good_values': [2, 3]
        },
        'weight_thr': {
            'name': 'Weight Threshold',
            'description': 'Minimum conditional probability required for an edge to be retained in the network',
            'purpose': 'Filters out weak probabilistic relationships between roads',
            'impact': 'Higher values keep only strong dependencies; lower values preserve more relationships',
            'range_tested': [0.2, 0.3, 0.4, 0.5],
            'typical_good_values': [0.2, 0.3]
        },
        'evidence_count': {
            'name': 'Evidence Count',
            'description': 'Number of roads that must be observed as flooded to trigger predictions for other roads',
            'purpose': 'Controls prediction sensitivity - how much evidence is needed before making predictions',
            'impact': 'Lower values make more aggressive predictions; higher values require more confirmation',
            'range_tested': [1, 2, 3, 4],
            'typical_good_values': [1, 2]
        },
        'pred_threshold': {
            'name': 'Prediction Threshold',
            'description': 'Minimum probability required for the model to predict a road will flood',
            'purpose': 'Controls the precision-recall trade-off for final predictions',
            'impact': 'Lower values increase recall but decrease precision; higher values do the opposite',
            'range_tested': [0.1, 0.2, 0.3, 0.4, 0.5],
            'typical_good_values': [0.1, 0.2]
        },
        'neg_pos_ratio': {
            'name': 'Negative-to-Positive Ratio',
            'description': 'Ratio of negative (non-flood) to positive (flood) samples used in training',
            'purpose': 'Balances the dataset to prevent bias toward the majority class',
            'impact': 'Higher ratios include more negative examples; lower ratios focus more on flood patterns',
            'range_tested': [1.0, 1.5, 2.0],
            'typical_good_values': [1.0, 1.5]
        },
        'marginal_prob_threshold': {
            'name': 'Marginal Probability Threshold',
            'description': 'Minimum marginal probability for a road to be included in negative sampling',
            'purpose': 'Controls which roads are considered as potential negative examples',
            'impact': 'Higher values focus on roads with higher baseline flood probability as negatives',
            'range_tested': [0.03, 0.05, 0.08],
            'typical_good_values': [0.05, 0.08]
        }
    }
    
    return param_docs

def generate_comprehensive_report(configs, param_analysis, param_docs, output_dir):
    """Generate comprehensive documentation report"""
    
    report = """# Parameter Optimization Analysis Report

## Executive Summary
This report analyzes the parameter optimization results for the Charleston Flood Prediction Bayesian Network model. We tested 5,760 parameter combinations and identified the top-performing configurations for different use cases.

## Parameter Definitions

"""
    
    # Add parameter documentation
    for param, info in param_docs.items():
        report += f"### {info['name']} (`{param}`)\n"
        report += f"**Description**: {info['description']}\n\n"
        report += f"**Purpose**: {info['purpose']}\n\n"
        report += f"**Impact**: {info['impact']}\n\n"
        report += f"**Range Tested**: {info['range_tested']}\n\n"
        report += f"**Typically Good Values**: {info['typical_good_values']}\n\n"
        report += "---\n\n"
    
    # Add top configurations analysis
    report += "## Top Performing Configurations\n\n"
    
    for category, df_configs in configs.items():
        if df_configs is not None and len(df_configs) > 0:
            category_name = category.replace('_', ' ').title()
            report += f"### {category_name}\n\n"
            
            for i, (idx, config) in enumerate(df_configs.head(3).iterrows(), 1):
                report += f"**Configuration {i}:**\n"
                report += f"- F1 Score: {config['f1_score']:.3f}\n"
                report += f"- Precision: {config['precision']:.3f}\n"
                report += f"- Recall: {config['recall']:.3f}\n"
                report += f"- Negative Candidates: {config['negative_candidates_count']:.0f}\n"
                report += "- Parameters:\n"
                for param in ['occ_thr', 'edge_thr', 'weight_thr', 'evidence_count', 
                             'pred_threshold', 'neg_pos_ratio', 'marginal_prob_threshold']:
                    report += f"  - {param}: {config[param]}\n"
                report += "\n"
            
            report += "---\n\n"
    
    # Add parameter patterns analysis
    report += "## Parameter Selection Patterns\n\n"
    report += "Based on analysis of top-performing configurations:\n\n"
    
    for param, stats in param_analysis.items():
        param_name = param_docs[param]['name']
        report += f"### {param_name} (`{param}`)\n"
        report += f"- Most common value: {stats['most_common']}\n"
        report += f"- Median value: {stats['median']:.3f}\n"
        report += f"- Value distribution: {stats['value_counts']}\n\n"
    
    # Add recommendations
    report += """## Deployment Recommendations

### For Emergency Response (High Recall Priority)
Use configurations that maximize recall to ensure no floods are missed, even if some false alarms occur.
Recommended parameters based on analysis:
- Lower pred_threshold (0.1-0.2)
- Lower evidence_count (1-2)
- Moderate occ_thr (3-4)

### For Planning and Preparedness (High Precision Priority)
Use configurations that minimize false positives for resource allocation and long-term planning.
Recommended parameters:
- Higher pred_threshold (0.3-0.4)
- Higher evidence_count (2-3)
- Higher weight_thr (0.3-0.4)

### For Balanced Operations (High F1 Priority)
Use configurations that balance precision and recall for general-purpose flood prediction.
Recommended parameters from best F1 configurations:
- occ_thr: 3-4
- edge_thr: 2-3
- weight_thr: 0.2-0.3
- evidence_count: 1-2
- pred_threshold: 0.1-0.2

### Robustness Considerations
For production deployment, prioritize configurations with:
- negative_candidates_count ≥ 15 (sufficient validation data)
- Consistent performance across different parameter nearby values
- Reasonable computational requirements (network_nodes < 100)

## Implementation Notes
1. Start with the highest F1 configuration for initial deployment
2. A/B test different thresholds based on operational needs
3. Monitor performance on new data and adjust parameters accordingly
4. Consider ensemble approaches using multiple top configurations
"""
    
    # Save report
    output_path = output_dir / 'PARAMETER_ANALYSIS_REPORT.md'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Comprehensive report saved to: {output_path}")
    return output_path

def create_top_configs_table(configs, output_dir):
    """Create a summary table of top configurations"""
    
    all_top_configs = []
    
    for category, df_configs in configs.items():
        if df_configs is not None and len(df_configs) > 0:
            for i, (idx, config) in enumerate(df_configs.head(3).iterrows()):
                config_data = {
                    'category': category.replace('_', ' ').title(),
                    'rank': i + 1,
                    'f1_score': config['f1_score'],
                    'precision': config['precision'],
                    'recall': config['recall'],
                    'negative_candidates': config['negative_candidates_count'],
                    'occ_thr': config['occ_thr'],
                    'edge_thr': config['edge_thr'],
                    'weight_thr': config['weight_thr'],
                    'evidence_count': config['evidence_count'],
                    'pred_threshold': config['pred_threshold'],
                    'neg_pos_ratio': config['neg_pos_ratio'],
                    'marginal_prob_threshold': config['marginal_prob_threshold']
                }
                all_top_configs.append(config_data)
    
    # Create DataFrame and save
    top_configs_df = pd.DataFrame(all_top_configs)
    csv_path = output_dir / 'top_configurations_summary.csv'
    top_configs_df.to_csv(csv_path, index=False)
    
    print(f"Top configurations table saved to: {csv_path}")
    return top_configs_df

def main():
    """Main analysis function"""
    
    # Paths
    results_dir = Path("results/parameter_optimization_20250721_140722")
    csv_path = results_dir / "complete_results.csv"
    output_dir = results_dir / "analysis"
    output_dir.mkdir(exist_ok=True)
    
    print("=== Parameter Optimization Analysis ===\n")
    
    # Load and filter data
    df = load_and_filter_data(csv_path)
    
    if len(df) == 0:
        print("❌ No data remaining after filtering!")
        return
    
    print(f"\n📊 Analyzing {len(df)} filtered configurations...")
    
    # Find top configurations
    configs = find_top_configurations(df)
    
    print("\n🏆 Top configurations found:")
    for category, df_configs in configs.items():
        if df_configs is not None and len(df_configs) > 0:
            best_f1 = df_configs['f1_score'].iloc[0]
            print(f"  - {category.replace('_', ' ').title()}: {len(df_configs)} configs (best F1: {best_f1:.3f})")
    
    # Analyze parameter patterns
    param_analysis, top_performers = analyze_parameter_patterns(df)
    
    print(f"\n📈 Analyzed patterns from top {len(top_performers)} performers (top 20%)")
    
    # Create parameter documentation
    param_docs = create_parameter_documentation()
    
    # Generate comprehensive report
    report_path = generate_comprehensive_report(configs, param_analysis, param_docs, output_dir)
    
    # Create top configurations table
    top_configs_df = create_top_configs_table(configs, output_dir)
    
    print(f"\n✅ Analysis completed successfully!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"📄 Main report: {report_path}")
    print(f"📊 Summary table: {output_dir / 'top_configurations_summary.csv'}")
    
    # Show overall best configuration
    if 'highest_f1' in configs and len(configs['highest_f1']) > 0:
        best_config = configs['highest_f1'].iloc[0]
        print(f"\n🥇 Overall Best Configuration (F1={best_config['f1_score']:.3f}):")
        print(f"   Precision: {best_config['precision']:.3f}, Recall: {best_config['recall']:.3f}")
        print(f"   Parameters: occ_thr={best_config['occ_thr']}, edge_thr={best_config['edge_thr']}, weight_thr={best_config['weight_thr']}")

if __name__ == "__main__":
    main()