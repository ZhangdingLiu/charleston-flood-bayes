#!/usr/bin/env python3
"""
Complete Top Configurations Dataset

Creates a comprehensive CSV file with ALL columns from complete_results.csv
for the selected top configurations, including confusion matrix values,
sample counts, timing data, and metadata.
"""

import pandas as pd
from pathlib import Path

def load_complete_results(csv_path):
    """Load the complete results dataset"""
    print("Loading complete results dataset...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} total configurations")
    print(f"Columns available: {len(df.columns)}")
    return df

def apply_filtering_criteria(df):
    """Apply the same filtering criteria used in analysis"""
    print("\nApplying filtering criteria...")
    print(f"Original data: {len(df)} configurations")
    
    # Apply filters
    filtered_df = df[
        (df['negative_candidates_count'] >= 9) &
        (df['precision'] < 1.0) &
        (df['recall'] < 1.0)
    ].copy()
    
    print(f"Filtered data: {len(filtered_df)} configurations")
    print(f"Removed {len(df) - len(filtered_df)} configurations due to filtering")
    
    return filtered_df

def find_complete_top_configurations(df):
    """Find top configurations using the same criteria as before"""
    
    print("\nIdentifying top configurations...")
    
    configs_with_metadata = []
    
    # 1. Highest F1 Score (overall best) - Top 5
    top_f1 = df.nlargest(5, 'f1_score')
    for rank, (idx, config) in enumerate(top_f1.iterrows(), 1):
        config_data = config.copy()
        config_data['category'] = 'Highest F1'
        config_data['rank'] = rank
        configs_with_metadata.append(config_data)
    print(f"   - Highest F1: {len(top_f1)} configurations")
    
    # 2. Highest Precision (with reasonable recall > 0.7) - Top 3
    high_prec_df = df[df['recall'] > 0.7]
    if len(high_prec_df) > 0:
        top_precision = high_prec_df.nlargest(3, 'precision')
        for rank, (idx, config) in enumerate(top_precision.iterrows(), 1):
            config_data = config.copy()
            config_data['category'] = 'Highest Precision'
            config_data['rank'] = rank
            configs_with_metadata.append(config_data)
        print(f"   - Highest Precision: {len(top_precision)} configurations")
    
    # 3. Highest Recall (with reasonable precision > 0.7) - Top 3
    high_recall_df = df[df['precision'] > 0.7]
    if len(high_recall_df) > 0:
        top_recall = high_recall_df.nlargest(3, 'recall')
        for rank, (idx, config) in enumerate(top_recall.iterrows(), 1):
            config_data = config.copy()
            config_data['category'] = 'Highest Recall'
            config_data['rank'] = rank
            configs_with_metadata.append(config_data)
        print(f"   - Highest Recall: {len(top_recall)} configurations")
    
    # 4. Best Balanced (precision > 0.8 AND recall > 0.8) - Top 3
    balanced_df = df[(df['precision'] > 0.8) & (df['recall'] > 0.8)]
    if len(balanced_df) > 0:
        top_balanced = balanced_df.nlargest(3, 'f1_score')
        for rank, (idx, config) in enumerate(top_balanced.iterrows(), 1):
            config_data = config.copy()
            config_data['category'] = 'Best Balanced'
            config_data['rank'] = rank
            configs_with_metadata.append(config_data)
        print(f"   - Best Balanced: {len(top_balanced)} configurations")
    
    # 5. Most Robust (high negative_candidates_count, good performance) - Top 3
    robust_df = df[(df['f1_score'] > 0.7) & (df['negative_candidates_count'] >= 15)]
    if len(robust_df) > 0:
        top_robust = robust_df.nlargest(3, 'negative_candidates_count')
        for rank, (idx, config) in enumerate(top_robust.iterrows(), 1):
            config_data = config.copy()
            config_data['category'] = 'Most Robust'
            config_data['rank'] = rank
            configs_with_metadata.append(config_data)
        print(f"   - Most Robust: {len(top_robust)} configurations")
    
    return configs_with_metadata

def create_complete_dataset(configs_with_metadata, output_path):
    """Create the complete dataset with all columns plus metadata"""
    
    print(f"\nCreating complete dataset with {len(configs_with_metadata)} configurations...")
    
    # Convert to DataFrame
    complete_df = pd.DataFrame(configs_with_metadata)
    
    # Reorder columns to put metadata first, then all original columns
    metadata_cols = ['category', 'rank']
    original_cols = [col for col in complete_df.columns if col not in metadata_cols]
    
    # Ensure we have all 23 original columns as expected
    expected_original_cols = [
        'occ_thr', 'edge_thr', 'weight_thr', 'evidence_count', 'pred_threshold', 
        'neg_pos_ratio', 'marginal_prob_threshold',  # 7 parameter columns
        'tp', 'fp', 'tn', 'fn',  # 4 confusion matrix columns
        'total_samples', 'positive_samples', 'negative_samples',  # 3 sample columns
        'precision', 'recall', 'f1_score', 'accuracy',  # 4 performance columns
        'valid_days', 'total_days',  # 2 data info columns
        'network_nodes', 'negative_candidates_count',  # 2 model info columns
        'runtime_seconds'  # 1 timing column
    ]
    
    # Verify all expected columns are present
    missing_cols = [col for col in expected_original_cols if col not in original_cols]
    if missing_cols:
        print(f"⚠️  Warning: Missing columns: {missing_cols}")
    
    extra_cols = [col for col in original_cols if col not in expected_original_cols]
    if extra_cols:
        print(f"ℹ️  Additional columns found: {extra_cols}")
    
    # Final column order: metadata + all original columns
    final_cols = metadata_cols + original_cols
    complete_df = complete_df[final_cols]
    
    # Save to CSV
    complete_df.to_csv(output_path, index=False)
    
    print(f"✅ Complete dataset saved to: {output_path}")
    print(f"📊 Dataset info:")
    print(f"   - Total configurations: {len(complete_df)}")
    print(f"   - Total columns: {len(complete_df.columns)}")
    print(f"   - Metadata columns: {len(metadata_cols)}")
    print(f"   - Original data columns: {len(original_cols)}")
    
    return complete_df

def generate_dataset_summary(complete_df, output_dir):
    """Generate a summary of the complete dataset"""
    
    summary_text = """# Complete Top Configurations Dataset Summary

## Dataset Overview
This dataset contains the complete information for all top-performing parameter configurations,
including ALL original columns from the parameter optimization results plus metadata.

## Dataset Structure
"""
    
    summary_text += f"- **Total Configurations**: {len(complete_df)}\n"
    summary_text += f"- **Total Columns**: {len(complete_df.columns)}\n"
    summary_text += f"- **Categories**: {complete_df['category'].nunique()}\n\n"
    
    # Configuration breakdown by category
    summary_text += "## Configuration Breakdown\n\n"
    category_counts = complete_df['category'].value_counts()
    for category, count in category_counts.items():
        summary_text += f"- **{category}**: {count} configurations\n"
    
    # Column information
    summary_text += "\n## Column Information\n\n"
    summary_text += "### Metadata Columns\n"
    summary_text += "- `category`: Type of optimization criteria (Highest F1, Highest Precision, etc.)\n"
    summary_text += "- `rank`: Rank within the category (1 = best in category)\n\n"
    
    summary_text += "### Parameter Columns (7)\n"
    param_cols = ['occ_thr', 'edge_thr', 'weight_thr', 'evidence_count', 
                  'pred_threshold', 'neg_pos_ratio', 'marginal_prob_threshold']
    for col in param_cols:
        if col in complete_df.columns:
            unique_vals = sorted(complete_df[col].unique())
            summary_text += f"- `{col}`: {unique_vals}\n"
    
    summary_text += "\n### Performance Metrics (4)\n"
    perf_cols = ['precision', 'recall', 'f1_score', 'accuracy']
    for col in perf_cols:
        if col in complete_df.columns:
            min_val = complete_df[col].min()
            max_val = complete_df[col].max()
            mean_val = complete_df[col].mean()
            summary_text += f"- `{col}`: Range [{min_val:.3f}, {max_val:.3f}], Mean: {mean_val:.3f}\n"
    
    summary_text += "\n### Confusion Matrix (4)\n"
    conf_cols = ['tp', 'fp', 'tn', 'fn']
    for col in conf_cols:
        if col in complete_df.columns:
            min_val = complete_df[col].min()
            max_val = complete_df[col].max()
            summary_text += f"- `{col}`: Range [{int(min_val)}, {int(max_val)}]\n"
    
    summary_text += "\n### Sample Information (3)\n"
    sample_cols = ['total_samples', 'positive_samples', 'negative_samples']
    for col in sample_cols:
        if col in complete_df.columns:
            min_val = complete_df[col].min()
            max_val = complete_df[col].max()
            summary_text += f"- `{col}`: Range [{int(min_val)}, {int(max_val)}]\n"
    
    summary_text += "\n### Model & Data Information\n"
    info_cols = ['valid_days', 'total_days', 'network_nodes', 'negative_candidates_count', 'runtime_seconds']
    for col in info_cols:
        if col in complete_df.columns:
            min_val = complete_df[col].min()
            max_val = complete_df[col].max()
            if col == 'runtime_seconds':
                summary_text += f"- `{col}`: Range [{min_val:.2f}s, {max_val:.2f}s]\n"
            else:
                summary_text += f"- `{col}`: Range [{int(min_val)}, {int(max_val)}]\n"
    
    summary_text += f"""
## Top Performers Summary

### Best Overall Configuration (Highest F1)
"""
    best_config = complete_df[complete_df['category'] == 'Highest F1'].iloc[0]
    summary_text += f"- **F1 Score**: {best_config['f1_score']:.3f}\n"
    summary_text += f"- **Precision**: {best_config['precision']:.3f}\n"
    summary_text += f"- **Recall**: {best_config['recall']:.3f}\n"
    summary_text += f"- **Accuracy**: {best_config['accuracy']:.3f}\n"
    summary_text += f"- **True Positives**: {int(best_config['tp'])}\n"
    summary_text += f"- **False Positives**: {int(best_config['fp'])}\n"
    summary_text += f"- **Total Samples**: {int(best_config['total_samples'])}\n"
    summary_text += f"- **Network Nodes**: {int(best_config['network_nodes'])}\n"
    summary_text += f"- **Runtime**: {best_config['runtime_seconds']:.2f} seconds\n"
    
    summary_text += "\n## Usage Notes\n"
    summary_text += "- All configurations have been filtered to exclude overfitting cases\n"
    summary_text += "- Only configurations with negative_candidates_count ≥ 9 are included\n"
    summary_text += "- Perfect precision or recall (1.0) configurations were excluded\n"
    summary_text += "- This dataset provides complete transparency for parameter selection decisions\n"
    
    # Save summary
    summary_path = output_dir / 'COMPLETE_DATASET_SUMMARY.md'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    
    print(f"📄 Dataset summary saved to: {summary_path}")

def main():
    """Main function to create complete top configurations dataset"""
    
    # Paths
    results_dir = Path("results/parameter_optimization_20250721_140722")
    analysis_dir = results_dir / "analysis"
    complete_results_path = results_dir / "complete_results.csv"
    output_path = analysis_dir / "complete_top_configurations.csv"
    
    print("=== Creating Complete Top Configurations Dataset ===\n")
    
    # Load complete results
    df = load_complete_results(complete_results_path)
    
    # Apply filtering criteria
    filtered_df = apply_filtering_criteria(df)
    
    if len(filtered_df) == 0:
        print("❌ No data remaining after filtering!")
        return
    
    # Find top configurations with all data
    configs_with_metadata = find_complete_top_configurations(filtered_df)
    
    # Create complete dataset
    complete_df = create_complete_dataset(configs_with_metadata, output_path)
    
    # Generate summary documentation
    generate_dataset_summary(complete_df, analysis_dir)
    
    print(f"\n✅ Complete dataset creation finished!")
    print(f"📁 Output location: {output_path}")
    print(f"📊 Total configurations: {len(complete_df)}")
    print(f"🔍 All {len(complete_df.columns)} columns preserved from original data")
    
    # Show sample of the data
    print(f"\n📋 Sample of complete dataset (first 3 rows):")
    print("Columns:", list(complete_df.columns))
    
    print(f"\n🏆 Best configuration details:")
    best = complete_df[complete_df['category'] == 'Highest F1'].iloc[0]
    print(f"   Category: {best['category']}, Rank: {best['rank']}")
    print(f"   Performance: F1={best['f1_score']:.3f}, P={best['precision']:.3f}, R={best['recall']:.3f}")
    print(f"   Confusion Matrix: TP={int(best['tp'])}, FP={int(best['fp'])}, TN={int(best['tn'])}, FN={int(best['fn'])}")
    print(f"   Samples: Total={int(best['total_samples'])}, Pos={int(best['positive_samples'])}, Neg={int(best['negative_samples'])}")

if __name__ == "__main__":
    main()