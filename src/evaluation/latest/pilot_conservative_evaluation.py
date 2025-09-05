#!/usr/bin/env python3
"""
Pilot Conservative Evaluation
Simplified version for quick testing of conservative negative sampling strategies
Based on ultra_conservative_ratio_analysis.py with reduced parameter grid
"""

import random
import numpy as np
import pandas as pd
from model import FloodBayesNetwork
import warnings
from datetime import datetime
import itertools
import os
import time
import json
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def load_and_preprocess_data():
    """Load and preprocess the flood data with detailed analysis"""
    print("=" * 80)
    print("PILOT CONSERVATIVE EVALUATION")
    print("Simplified conservative negative sampling strategies evaluation")
    print("=" * 80)
    
    # Load data
    df = pd.read_csv("Road_Closures_2024.csv")
    df = df[df["REASON"].str.upper() == "FLOOD"].copy()
    
    # Preprocess
    df["time_create"] = pd.to_datetime(df["START"], utc=True)
    df["link_id"] = df["STREET"].str.upper().str.replace(" ", "_")
    df["link_id"] = df["link_id"].astype(str)
    df["id"] = df["OBJECTID"].astype(str)
    df["year"] = df["time_create"].dt.year
    
    # Temporal split (70% train, 30% test)
    df_sorted = df.sort_values('time_create')
    split_idx = int(len(df_sorted) * 0.7)
    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()
    
    print(f"Data loaded: {len(df)} total flood records")
    print(f"Train set: {len(train_df)} records ({train_df['time_create'].min().strftime('%Y-%m-%d')} to {train_df['time_create'].max().strftime('%Y-%m-%d')})")
    print(f"Test set: {len(test_df)} records ({test_df['time_create'].min().strftime('%Y-%m-%d')} to {test_df['time_create'].max().strftime('%Y-%m-%d')})")
    
    return train_df, test_df

def build_selected_networks(train_df):
    """Build selected Bayesian networks with 3 promising parameter configurations"""
    print(f"\n{'='*60}")
    print("BUILDING SELECTED BAYESIAN NETWORKS")
    print(f"{'='*60}")
    
    # Selected network parameter configurations (3 most promising)
    network_params = [
        {'name': 'Conservative', 'occ_thr': 2, 'edge_thr': 1, 'weight_thr': 0.2},  # More inclusive
        {'name': 'Baseline', 'occ_thr': 3, 'edge_thr': 2, 'weight_thr': 0.3},     # Default/tested
        {'name': 'Restrictive', 'occ_thr': 4, 'edge_thr': 3, 'weight_thr': 0.4}   # Higher quality
    ]
    
    networks = {}
    road_stats_cache = None
    
    for i, params in enumerate(network_params):
        print(f"Building {params['name']} network: occ_thr={params['occ_thr']}, edge_thr={params['edge_thr']}, weight_thr={params['weight_thr']}")
        
        # Build network
        flood_net = FloodBayesNetwork(t_window="D")
        flood_net.fit_marginal(train_df)
        flood_net.build_network_by_co_occurrence(
            train_df, occ_thr=params['occ_thr'], edge_thr=params['edge_thr'], 
            weight_thr=params['weight_thr'], report=False
        )
        flood_net.fit_conditional(train_df, max_parents=2, alpha=1.0)
        flood_net.build_bayes_network()
        
        bn_nodes = set(flood_net.network_bayes.nodes())
        marginals_dict = dict(zip(flood_net.marginals['link_id'], flood_net.marginals['p']))
        
        # Build road statistics (only once, reuse for all networks)
        if road_stats_cache is None:
            road_stats = train_df.groupby('link_id').agg({
                'time_create': ['count', 'min', 'max'],
                'year': lambda x: len(x.unique())
            }).round(4)
            road_stats.columns = ['flood_count', 'first_flood', 'last_flood', 'years_active']
            road_stats = road_stats.reset_index()
            
            # Recent activity analysis (2020 onwards)
            recent_activity = train_df[train_df['year'] >= 2020].groupby('link_id')['time_create'].count()
            road_stats['recent_floods'] = road_stats['link_id'].map(recent_activity).fillna(0).astype(int)
            road_stats_cache = road_stats
        
        # Add marginal probabilities for this network
        current_road_stats = road_stats_cache.copy()
        current_road_stats['marginal_prob'] = current_road_stats['link_id'].map(marginals_dict)
        current_road_stats = current_road_stats[current_road_stats['link_id'].isin(bn_nodes)].copy()
        current_road_stats = current_road_stats.sort_values('marginal_prob')
        
        networks[params['name']] = {
            'params': params,
            'flood_net': flood_net,
            'bn_nodes': bn_nodes,
            'marginals_dict': marginals_dict,
            'road_stats': current_road_stats
        }
        
        print(f"  Network built: {len(bn_nodes)} nodes, marginal prob range: {current_road_stats['marginal_prob'].min():.4f} - {current_road_stats['marginal_prob'].max():.4f}")
    
    return networks

def get_conservative_negative_candidates(road_stats, strategy, **params):
    """
    Get negative sample candidates using conservative strategies (same as original)
    """
    
    if strategy == "MarginalProbability":
        threshold = params.get('prob_threshold', 0.10)
        candidates = road_stats[road_stats['marginal_prob'] <= threshold]['link_id'].tolist()
        
    elif strategy == "FloodFrequency":
        max_floods = params.get('max_flood_count', 3)
        candidates = road_stats[road_stats['flood_count'] <= max_floods]['link_id'].tolist()
        
    elif strategy == "RecentActivity":
        max_recent = params.get('max_recent_floods', 1)
        candidates = road_stats[road_stats['recent_floods'] <= max_recent]['link_id'].tolist()
        
    elif strategy == "CombinedConservative":
        prob_threshold = params.get('prob_threshold', 0.15)
        max_floods = params.get('max_flood_count', 3)
        max_recent = params.get('max_recent_floods', 2)
        candidates = road_stats[
            (road_stats['marginal_prob'] <= prob_threshold) & 
            (road_stats['flood_count'] <= max_floods) &
            (road_stats['recent_floods'] <= max_recent)
        ]['link_id'].tolist()
        
    else:
        candidates = []
    
    return candidates

def evaluate_conservative_configuration(network_data, test_df, negative_candidates, 
                                      pred_threshold, evidence_ratio, neg_pos_ratio, 
                                      config_info):
    """
    Evaluate a specific conservative negative sampling configuration (same as original)
    """
    
    flood_net = network_data['flood_net']
    bn_nodes = network_data['bn_nodes']
    
    if len(negative_candidates) == 0:
        return None
    
    results = []
    test_by_date = test_df.groupby(test_df["time_create"].dt.floor("D"))
    evaluated_days = 0
    skipped_days = 0
    
    for date, day_group in test_by_date:
        flooded_roads = list(day_group["link_id"].unique())
        flooded_in_bn = [road for road in flooded_roads if road in bn_nodes]
        
        # Need at least 2 roads in network for evidence + target
        if len(flooded_in_bn) < 2:
            skipped_days += 1
            continue
            
        evaluated_days += 1
        
        # Evidence selection - use portion of flooded roads as evidence
        evidence_count = max(1, int(len(flooded_in_bn) * evidence_ratio))
        evidence_count = min(evidence_count, len(flooded_in_bn) - 1)  # Leave at least 1 for target
        evidence_roads = flooded_in_bn[:evidence_count]
        target_roads = flooded_in_bn[evidence_count:]
        evidence = {road: 1 for road in evidence_roads}
        
        # Positive samples (confirmed floods)
        for target_road in target_roads:
            try:
                result = flood_net.infer_w_evidence(target_road, evidence)
                prob_flood = result['flooded']
                prediction = 1 if prob_flood >= pred_threshold else 0
                
                results.append({
                    'sample_type': 'Positive',
                    'target_road': target_road,
                    'prob_flood': prob_flood,
                    'prediction': prediction,
                    'true_label': 1,
                    'date': date.strftime('%Y-%m-%d'),
                    'is_correct': (prediction == 1)
                })
            except Exception:
                continue
        
        # Negative samples using conservative strategy
        available_negatives = [road for road in negative_candidates if road not in flooded_roads]
        n_negatives = min(len(available_negatives), max(1, int(len(target_roads) * neg_pos_ratio)))
        selected_negatives = available_negatives[:n_negatives]
        
        for neg_road in selected_negatives:
            try:
                result = flood_net.infer_w_evidence(neg_road, evidence)
                prob_flood = result['flooded']
                prediction = 1 if prob_flood >= pred_threshold else 0
                
                results.append({
                    'sample_type': 'Negative',
                    'target_road': neg_road,
                    'prob_flood': prob_flood,
                    'prediction': prediction,
                    'true_label': 0,
                    'date': date.strftime('%Y-%m-%d'),
                    'is_correct': (prediction == 0)
                })
            except Exception:
                continue
    
    if len(results) == 0:
        return None
    
    # Calculate confusion matrix
    tp = sum(1 for r in results if r['sample_type'] == 'Positive' and r['prediction'] == 1)
    fp = sum(1 for r in results if r['sample_type'] == 'Negative' and r['prediction'] == 1)
    fn = sum(1 for r in results if r['sample_type'] == 'Positive' and r['prediction'] == 0)
    tn = sum(1 for r in results if r['sample_type'] == 'Negative' and r['prediction'] == 0)
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
    
    # Sample counts
    positive_samples = sum(1 for r in results if r['sample_type'] == 'Positive')
    negative_samples = sum(1 for r in results if r['sample_type'] == 'Negative')
    total_samples = len(results)
    
    return {
        # Configuration info
        'network_config': config_info['network_config'],
        'strategy': config_info['strategy'],
        'strategy_params': config_info['strategy_params'],
        'neg_pos_ratio': neg_pos_ratio,
        'pred_threshold': pred_threshold,
        'evidence_ratio': evidence_ratio,
        
        # Network parameters
        'occ_thr': network_data['params']['occ_thr'],
        'edge_thr': network_data['params']['edge_thr'],
        'weight_thr': network_data['params']['weight_thr'],
        
        # Sample statistics
        'total_samples': total_samples,
        'positive_samples': positive_samples,
        'negative_samples': negative_samples,
        'negative_candidates_count': len(negative_candidates),
        'evaluated_days': evaluated_days,
        'skipped_days': skipped_days,
        
        # Confusion matrix
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        
        # Metrics
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy,
        
        # Raw results for detailed analysis
        'sample_details': results
    }

def run_pilot_evaluation():
    """Run the pilot conservative negative sampling evaluation with simplified parameters"""
    
    start_time = time.time()
    
    # Load data and build selected networks
    train_df, test_df = load_and_preprocess_data()
    networks = build_selected_networks(train_df)
    
    print(f"\n{'='*80}")
    print("DEFINING SIMPLIFIED EVALUATION GRID")
    print(f"{'='*80}")
    
    # Simplified conservative negative sampling strategies
    strategy_configs = {
        'MarginalProbability': [
            {'prob_threshold': 0.05},  # Very low probability roads
            {'prob_threshold': 0.10},  # Low probability roads
            {'prob_threshold': 0.15}   # Moderate-low probability roads
        ],
        'FloodFrequency': [
            {'max_flood_count': 1},    # Roads that flooded once or never
            {'max_flood_count': 2},    # Roads that flooded twice or less
            {'max_flood_count': 3}     # Roads that flooded 3 times or less
        ],
        'RecentActivity': [
            {'max_recent_floods': 0},  # No recent floods (2020+)
            {'max_recent_floods': 1},  # At most 1 recent flood
            {'max_recent_floods': 2}   # At most 2 recent floods
        ],
        'CombinedConservative': [
            {'prob_threshold': 0.10, 'max_flood_count': 2, 'max_recent_floods': 1},  # Conservative combo 1
            {'prob_threshold': 0.15, 'max_flood_count': 3, 'max_recent_floods': 2}   # Conservative combo 2
        ]
    }
    
    # Simplified parameter ranges
    neg_pos_ratios = [1.0, 2.0]  # Test 1:1 and 1:2 ratios
    pred_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]  # 5 prediction thresholds
    evidence_ratios = [0.2, 0.3, 0.5, 0.7]  # 4 evidence ratios
    
    # Calculate total combinations
    total_strategies = sum(len(params) for params in strategy_configs.values())
    total_combinations = (len(networks) * total_strategies * len(neg_pos_ratios) * 
                         len(pred_thresholds) * len(evidence_ratios))
    
    print(f"Pilot evaluation configuration:")
    print(f"  Networks: {len(networks)} (Conservative, Baseline, Restrictive)")
    print(f"  Strategy configurations: {total_strategies}")
    print(f"    • MarginalProbability: {len(strategy_configs['MarginalProbability'])} configs")
    print(f"    • FloodFrequency: {len(strategy_configs['FloodFrequency'])} configs")
    print(f"    • RecentActivity: {len(strategy_configs['RecentActivity'])} configs")
    print(f"    • CombinedConservative: {len(strategy_configs['CombinedConservative'])} configs")
    print(f"  Negative sample ratios: {len(neg_pos_ratios)} (1:1, 1:2)")
    print(f"  Prediction thresholds: {len(pred_thresholds)} (0.1-0.5)")
    print(f"  Evidence ratios: {len(evidence_ratios)} (0.2-0.7)")
    print(f"  TOTAL COMBINATIONS: {total_combinations:,}")
    print(f"\nEstimated runtime: 30-60 minutes")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"pilot_conservative_analysis_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("RUNNING PILOT EVALUATION")
    print(f"{'='*80}")
    
    all_results = []
    combination_count = 0
    
    # Run evaluation for each combination
    for network_name, network_data in networks.items():
        print(f"\nTesting {network_name} network: {network_data['params']}")
        
        for strategy_name, param_list in strategy_configs.items():
            for strategy_params in param_list:
                
                # Get negative candidates for this strategy
                negative_candidates = get_conservative_negative_candidates(
                    network_data['road_stats'], strategy_name, **strategy_params
                )
                
                if len(negative_candidates) == 0:
                    print(f"  {strategy_name} {strategy_params}: No candidates found, skipping")
                    continue
                
                print(f"  {strategy_name} {strategy_params}: {len(negative_candidates)} candidates")
                
                # Test all parameter combinations for this strategy
                for neg_ratio, pred_thresh, ev_ratio in itertools.product(
                    neg_pos_ratios, pred_thresholds, evidence_ratios
                ):
                    combination_count += 1
                    
                    if combination_count % 100 == 0:
                        elapsed = time.time() - start_time
                        progress = combination_count / total_combinations * 100
                        print(f"    Progress: {combination_count:,}/{total_combinations:,} "
                              f"({progress:.1f}%) - Elapsed: {elapsed/60:.1f}min")
                    
                    config_info = {
                        'network_config': network_name,
                        'strategy': strategy_name,
                        'strategy_params': str(strategy_params)
                    }
                    
                    # Run evaluation
                    result = evaluate_conservative_configuration(
                        network_data, test_df, negative_candidates,
                        pred_thresh, ev_ratio, neg_ratio, config_info
                    )
                    
                    if result is not None:
                        all_results.append(result)
    
    # Save results (same output format as original)
    print(f"\n{'='*80}")
    print("SAVING PILOT EVALUATION RESULTS")
    print(f"{'='*80}")
    
    elapsed_time = time.time() - start_time
    print(f"Total evaluation time: {elapsed_time/60:.1f} minutes")
    print(f"Completed evaluations: {len(all_results):,}")
    print(f"Output directory: {output_dir}")
    
    # 1. Complete results
    complete_results = []
    for result in all_results:
        result_copy = result.copy()
        del result_copy['sample_details']  # Remove for main summary
        complete_results.append(result_copy)
    
    complete_df = pd.DataFrame(complete_results)
    complete_file = os.path.join(output_dir, "complete_results.csv")
    complete_df.to_csv(complete_file, index=False)
    print(f"✓ Complete results saved: {complete_file}")
    
    # 2. Ratio sensitivity analysis
    ratio_sensitivity = []
    for ratio in neg_pos_ratios:
        ratio_results = complete_df[complete_df['neg_pos_ratio'] == ratio]
        if len(ratio_results) > 0:
            ratio_sensitivity.append({
                'neg_pos_ratio': ratio,
                'count': len(ratio_results),
                'avg_precision': ratio_results['precision'].mean(),
                'avg_recall': ratio_results['recall'].mean(),
                'avg_f1': ratio_results['f1_score'].mean(),
                'max_precision': ratio_results['precision'].max(),
                'max_recall': ratio_results['recall'].max(),
                'max_f1': ratio_results['f1_score'].max(),
                'std_precision': ratio_results['precision'].std(),
                'std_recall': ratio_results['recall'].std(),
                'std_f1': ratio_results['f1_score'].std()
            })
    
    ratio_df = pd.DataFrame(ratio_sensitivity)
    ratio_file = os.path.join(output_dir, "ratio_sensitivity_summary.csv")
    ratio_df.to_csv(ratio_file, index=False)
    print(f"✓ Ratio sensitivity analysis saved: {ratio_file}")
    
    # 3. Best configurations
    best_configs = []
    
    # Top by precision, recall, F1
    for metric, label in [('precision', 'top_precision'), ('recall', 'top_recall'), ('f1_score', 'top_f1')]:
        top_results = complete_df.nlargest(10, metric)
        for _, row in top_results.iterrows():
            best_configs.append({**row.to_dict(), 'selection_criteria': label})
    
    # High precision results
    high_precision = complete_df[complete_df['precision'] >= 0.7]
    for _, row in high_precision.iterrows():
        best_configs.append({**row.to_dict(), 'selection_criteria': 'high_precision'})
    
    best_df = pd.DataFrame(best_configs).drop_duplicates()
    best_file = os.path.join(output_dir, "best_configurations.csv")
    best_df.to_csv(best_file, index=False)
    print(f"✓ Best configurations saved: {best_file}")
    
    # 4. Strategy comparison
    strategy_comparison = []
    for strategy in strategy_configs.keys():
        strategy_results = complete_df[complete_df['strategy'] == strategy]
        if len(strategy_results) > 0:
            strategy_comparison.append({
                'strategy': strategy,
                'count': len(strategy_results),
                'avg_precision': strategy_results['precision'].mean(),
                'avg_recall': strategy_results['recall'].mean(),
                'avg_f1': strategy_results['f1_score'].mean(),
                'max_precision': strategy_results['precision'].max(),
                'max_recall': strategy_results['recall'].max(),
                'max_f1': strategy_results['f1_score'].max(),
                'best_config_precision': strategy_results.loc[strategy_results['precision'].idxmax(), 'strategy_params'],
                'best_config_f1': strategy_results.loc[strategy_results['f1_score'].idxmax(), 'strategy_params']
            })
    
    strategy_df = pd.DataFrame(strategy_comparison)
    strategy_file = os.path.join(output_dir, "strategy_comparison.csv")
    strategy_df.to_csv(strategy_file, index=False)
    print(f"✓ Strategy comparison saved: {strategy_file}")
    
    # 5. Deployment recommendations
    recommendations = generate_pilot_recommendations(complete_df, ratio_df, networks)
    rec_file = os.path.join(output_dir, "deployment_recommendations.txt")
    with open(rec_file, 'w') as f:
        f.write(recommendations)
    print(f"✓ Deployment recommendations saved: {rec_file}")
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("PILOT EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total configurations evaluated: {len(complete_df):,}")
    print(f"High precision results (≥0.7): {len(complete_df[complete_df['precision'] >= 0.7])}")
    print(f"High recall results (≥0.7): {len(complete_df[complete_df['recall'] >= 0.7])}")
    print(f"Balanced results (P≥0.5, R≥0.5): {len(complete_df[(complete_df['precision'] >= 0.5) & (complete_df['recall'] >= 0.5)])}")
    
    if len(complete_df) > 0:
        best_overall = complete_df.loc[complete_df['f1_score'].idxmax()]
        print(f"\nBest overall F1-Score: {best_overall['f1_score']:.3f}")
        print(f"  Strategy: {best_overall['strategy']} {best_overall['strategy_params']}")
        print(f"  Network: {best_overall['network_config']}")
        print(f"  Ratio: 1:{best_overall['neg_pos_ratio']}")
        print(f"  Precision: {best_overall['precision']:.3f}, Recall: {best_overall['recall']:.3f}")
    
    print(f"\n{'='*80}")
    print("PILOT EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved in: {output_dir}")
    print("\nKey files to analyze:")
    print(f"  📊 complete_results.csv")
    print(f"  📈 ratio_sensitivity_summary.csv")
    print(f"  🏆 best_configurations.csv")
    print(f"  🔍 strategy_comparison.csv")
    print(f"  📋 deployment_recommendations.txt")
    
    return output_dir

def generate_pilot_recommendations(complete_df, ratio_df, networks):
    """Generate deployment recommendations based on pilot evaluation results"""
    
    recommendations = f"""
PILOT CONSERVATIVE EVALUATION - DEPLOYMENT RECOMMENDATIONS
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================

EXECUTIVE SUMMARY
================================================================================
Total configurations evaluated: {len(complete_df):,}
Best overall F1-Score: {complete_df['f1_score'].max():.3f}
Best precision: {complete_df['precision'].max():.3f}
Best recall: {complete_df['recall'].max():.3f}

PILOT FINDINGS
================================================================================
"""
    
    # Network performance analysis
    network_performance = complete_df.groupby('network_config').agg({
        'precision': ['mean', 'max'],
        'recall': ['mean', 'max'], 
        'f1_score': ['mean', 'max']
    }).round(3)
    
    recommendations += "NETWORK CONFIGURATION PERFORMANCE:\n"
    for network in network_performance.index:
        avg_f1 = network_performance.loc[network, ('f1_score', 'mean')]
        max_f1 = network_performance.loc[network, ('f1_score', 'max')]
        recommendations += f"• {network}: Avg F1={avg_f1:.3f}, Max F1={max_f1:.3f}\n"
    
    # Ratio analysis
    if len(ratio_df) > 0:
        best_ratio_f1 = ratio_df.loc[ratio_df['avg_f1'].idxmax()]
        best_ratio_precision = ratio_df.loc[ratio_df['avg_precision'].idxmax()]
        
        recommendations += f"""

NEGATIVE SAMPLE RATIO ANALYSIS:
• Best ratio for F1-Score: 1:{best_ratio_f1['neg_pos_ratio']} (avg F1: {best_ratio_f1['avg_f1']:.3f})
• Best ratio for Precision: 1:{best_ratio_precision['neg_pos_ratio']} (avg precision: {best_ratio_precision['avg_precision']:.3f})

"""
    
    # Strategy effectiveness
    strategy_performance = complete_df.groupby('strategy').agg({
        'precision': ['mean', 'max'],
        'recall': ['mean', 'max'], 
        'f1_score': ['mean', 'max']
    }).round(3)
    
    recommendations += "STRATEGY EFFECTIVENESS:\n"
    for strategy in strategy_performance.index:
        avg_f1 = strategy_performance.loc[strategy, ('f1_score', 'mean')]
        max_f1 = strategy_performance.loc[strategy, ('f1_score', 'max')]
        recommendations += f"• {strategy}: Avg F1={avg_f1:.3f}, Max F1={max_f1:.3f}\n"
    
    # Top configuration recommendations
    top_configs = complete_df.nlargest(3, 'f1_score')
    recommendations += f"""

TOP 3 PILOT RECOMMENDATIONS
================================================================================
"""
    
    for i, (_, config) in enumerate(top_configs.iterrows(), 1):
        recommendations += f"""
PILOT RECOMMENDATION #{i}:
• Strategy: {config['strategy']} {config['strategy_params']}
• Network: {config['network_config']}
• Negative ratio: 1:{config['neg_pos_ratio']}
• Prediction threshold: {config['pred_threshold']}
• Evidence ratio: {config['evidence_ratio']}
• Performance: Precision={config['precision']:.3f}, Recall={config['recall']:.3f}, F1={config['f1_score']:.3f}
• Sample size: {config['total_samples']} ({config['positive_samples']} pos, {config['negative_samples']} neg)

"""
    
    recommendations += """
NEXT STEPS FOR FULL EVALUATION
================================================================================
Based on pilot results, consider refining parameter ranges for comprehensive evaluation:
1. Focus on most promising network configurations
2. Narrow down strategy parameter ranges based on pilot performance
3. Test additional negative sample ratios around best-performing values
4. Expand prediction threshold range around optimal values

DEPLOYMENT GUIDELINES
================================================================================
1. Conservative negative sampling shows promise for addressing observation bias
2. Monitor precision vs recall trade-offs based on operational requirements
3. Consider computational resources when selecting network configuration
4. Validate performance stability across different time periods

"""
    
    return recommendations

def main():
    """Main execution function"""
    try:
        print("Starting Pilot Conservative Evaluation...")
        print("This simplified evaluation will take 30-60 minutes to complete.")
        print("Progress will be shown during execution.")
        
        output_dir = run_pilot_evaluation()
        
        print(f"\n🎉 PILOT ANALYSIS COMPLETE!")
        print(f"📁 Results directory: {output_dir}")
        print(f"\n📧 Send these files for analysis:")
        print(f"   • complete_results.csv")
        print(f"   • ratio_sensitivity_summary.csv") 
        print(f"   • best_configurations.csv")
        print(f"   • strategy_comparison.csv")
        print(f"   • deployment_recommendations.txt")
        
        return output_dir
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    output_dir = main()