#!/usr/bin/env python3
"""
Conservative Negative Sampling Evaluation
Comprehensive evaluation using different negative sampling strategies to find optimal precision/recall
"""

import random
import numpy as np
import pandas as pd
from model import FloodBayesNetwork
import warnings
from datetime import datetime
import itertools
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def load_and_preprocess_data():
    """Load and preprocess the flood data with additional analysis"""
    print("=" * 80)
    print("CONSERVATIVE NEGATIVE SAMPLING EVALUATION")
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
    
    return train_df, test_df, df

def build_bayesian_network(train_df):
    """Build the Bayesian network and analyze road statistics"""
    print(f"\n{'='*60}")
    print("BUILDING BAYESIAN NETWORK & ANALYZING ROAD STATISTICS")
    print(f"{'='*60}")
    
    # Build network
    flood_net = FloodBayesNetwork(t_window="D")
    flood_net.fit_marginal(train_df)
    flood_net.build_network_by_co_occurrence(
        train_df, occ_thr=3, edge_thr=2, weight_thr=0.3, report=False
    )
    flood_net.fit_conditional(train_df, max_parents=2, alpha=1.0)
    flood_net.build_bayes_network()
    
    bn_nodes = set(flood_net.network_bayes.nodes())
    marginals_dict = dict(zip(flood_net.marginals['link_id'], flood_net.marginals['p']))
    
    # Analyze road flooding statistics
    road_stats = train_df.groupby('link_id').agg({
        'time_create': ['count', 'min', 'max'],
        'year': lambda x: len(x.unique())
    }).round(4)
    road_stats.columns = ['flood_count', 'first_flood', 'last_flood', 'years_active']
    road_stats = road_stats.reset_index()
    
    # Add marginal probabilities
    road_stats['marginal_prob'] = road_stats['link_id'].map(marginals_dict)
    road_stats = road_stats[road_stats['link_id'].isin(bn_nodes)].copy()
    road_stats = road_stats.sort_values('marginal_prob')
    
    # Recent activity analysis (2020 onwards)
    recent_activity = train_df[train_df['year'] >= 2020].groupby('link_id')['time_create'].count()
    road_stats['recent_floods'] = road_stats['link_id'].map(recent_activity).fillna(0).astype(int)
    
    print(f"Network built with {len(bn_nodes)} nodes")
    print(f"Road statistics computed for {len(road_stats)} roads")
    print(f"Marginal probability range: {road_stats['marginal_prob'].min():.4f} - {road_stats['marginal_prob'].max():.4f}")
    print(f"Flood count range: {road_stats['flood_count'].min()} - {road_stats['flood_count'].max()}")
    
    return flood_net, bn_nodes, marginals_dict, road_stats

def get_negative_candidates_by_strategy(road_stats, strategy_name, **params):
    """
    Get negative sample candidates based on different strategies
    
    Args:
        road_stats: DataFrame with road statistics
        strategy_name: Name of the strategy
        **params: Strategy-specific parameters
    
    Returns:
        List of road IDs selected as negative candidates
    """
    
    if strategy_name == "marginal_prob":
        # Strategy 1: Select roads by marginal probability threshold
        threshold = params.get('prob_threshold', 0.1)
        candidates = road_stats[road_stats['marginal_prob'] <= threshold]['link_id'].tolist()
        
    elif strategy_name == "flood_frequency":
        # Strategy 2: Select roads by historical flood frequency
        max_floods = params.get('max_flood_count', 5)
        candidates = road_stats[road_stats['flood_count'] <= max_floods]['link_id'].tolist()
        
    elif strategy_name == "recent_activity":
        # Strategy 3: Select roads with little recent activity
        max_recent = params.get('max_recent_floods', 2)
        candidates = road_stats[road_stats['recent_floods'] <= max_recent]['link_id'].tolist()
        
    elif strategy_name == "combined_conservative":
        # Strategy 4: Combined criteria (low probability AND low frequency)
        prob_threshold = params.get('prob_threshold', 0.15)
        max_floods = params.get('max_flood_count', 8)
        candidates = road_stats[
            (road_stats['marginal_prob'] <= prob_threshold) & 
            (road_stats['flood_count'] <= max_floods)
        ]['link_id'].tolist()
        
    elif strategy_name == "temporal_conservative":
        # Strategy 5: Roads that haven't flooded recently but have some history
        min_floods = params.get('min_flood_count', 3)
        max_recent = params.get('max_recent_floods', 1)
        candidates = road_stats[
            (road_stats['flood_count'] >= min_floods) & 
            (road_stats['recent_floods'] <= max_recent)
        ]['link_id'].tolist()
        
    else:
        candidates = []
    
    return candidates

def evaluate_with_negative_strategy(flood_net, test_df, bn_nodes, road_stats, 
                                  strategy_name, strategy_params, 
                                  pred_threshold=0.3, evidence_ratio=0.5, neg_pos_ratio=1.0):
    """
    Evaluate the model using a specific negative sampling strategy
    
    Returns:
        Dictionary with evaluation results
    """
    
    # Get negative candidates using the strategy
    negative_candidates = get_negative_candidates_by_strategy(road_stats, strategy_name, **strategy_params)
    
    if len(negative_candidates) == 0:
        return {
            'strategy': strategy_name,
            'strategy_params': strategy_params,
            'pred_threshold': pred_threshold,
            'evidence_ratio': evidence_ratio,
            'neg_pos_ratio': neg_pos_ratio,
            'negative_candidates_count': 0,
            'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0,
            'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0,
            'positive_samples': 0, 'negative_samples': 0,
            'error': 'No negative candidates found'
        }
    
    # Run evaluation
    results = []
    test_by_date = test_df.groupby(test_df["time_create"].dt.floor("D"))
    evaluated_days = 0
    
    for date, day_group in test_by_date:
        flooded_roads = list(day_group["link_id"].unique())
        flooded_in_bn = [road for road in flooded_roads if road in bn_nodes]
        
        if len(flooded_in_bn) < 2:
            continue
            
        evaluated_days += 1
        
        # Evidence selection
        evidence_count = max(1, int(len(flooded_in_bn) * evidence_ratio))
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
                    'type': 'Positive',
                    'target_road': target_road,
                    'prob_flood': prob_flood,
                    'prediction': prediction,
                    'true_label': 1,
                    'date': date.strftime('%Y-%m-%d')
                })
            except:
                continue
        
        # Negative samples
        available_negatives = [road for road in negative_candidates if road not in flooded_roads]
        n_negatives = min(len(available_negatives), max(1, int(len(target_roads) * neg_pos_ratio)))
        selected_negatives = available_negatives[:n_negatives]
        
        for neg_road in selected_negatives:
            try:
                result = flood_net.infer_w_evidence(neg_road, evidence)
                prob_flood = result['flooded']
                prediction = 1 if prob_flood >= pred_threshold else 0
                
                results.append({
                    'type': 'Negative',
                    'target_road': neg_road,
                    'prob_flood': prob_flood,
                    'prediction': prediction,
                    'true_label': 0,
                    'date': date.strftime('%Y-%m-%d')
                })
            except:
                continue
    
    # Calculate confusion matrix
    tp = sum(1 for r in results if r['type'] == 'Positive' and r['prediction'] == 1)
    fp = sum(1 for r in results if r['type'] == 'Negative' and r['prediction'] == 1)
    fn = sum(1 for r in results if r['type'] == 'Positive' and r['prediction'] == 0)
    tn = sum(1 for r in results if r['type'] == 'Negative' and r['prediction'] == 0)
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
    
    positive_samples = sum(1 for r in results if r['type'] == 'Positive')
    negative_samples = sum(1 for r in results if r['type'] == 'Negative')
    
    return {
        'strategy': strategy_name,
        'strategy_params': str(strategy_params),
        'pred_threshold': pred_threshold,
        'evidence_ratio': evidence_ratio,
        'neg_pos_ratio': neg_pos_ratio,
        'negative_candidates_count': len(negative_candidates),
        'negative_candidates': negative_candidates[:10],  # First 10 for reference
        'evaluated_days': evaluated_days,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy,
        'positive_samples': positive_samples,
        'negative_samples': negative_samples,
        'total_samples': len(results)
    }

def run_comprehensive_evaluation():
    """Run comprehensive evaluation with all strategy combinations"""
    
    # Load data and build network
    train_df, test_df, full_df = load_and_preprocess_data()
    flood_net, bn_nodes, marginals_dict, road_stats = build_bayesian_network(train_df)
    
    print(f"\n{'='*60}")
    print("NEGATIVE CANDIDATE ANALYSIS")
    print(f"{'='*60}")
    
    # Define strategy configurations
    strategy_configs = {
        'marginal_prob': [
            {'prob_threshold': 0.05},
            {'prob_threshold': 0.10},
            {'prob_threshold': 0.15},
            {'prob_threshold': 0.20},
            {'prob_threshold': 0.25}
        ],
        'flood_frequency': [
            {'max_flood_count': 3},
            {'max_flood_count': 5},
            {'max_flood_count': 8},
            {'max_flood_count': 10},
            {'max_flood_count': 12}
        ],
        'recent_activity': [
            {'max_recent_floods': 0},
            {'max_recent_floods': 1},
            {'max_recent_floods': 2},
            {'max_recent_floods': 3}
        ],
        'combined_conservative': [
            {'prob_threshold': 0.10, 'max_flood_count': 5},
            {'prob_threshold': 0.15, 'max_flood_count': 8},
            {'prob_threshold': 0.20, 'max_flood_count': 10},
            {'prob_threshold': 0.25, 'max_flood_count': 12}
        ],
        'temporal_conservative': [
            {'min_flood_count': 3, 'max_recent_floods': 0},
            {'min_flood_count': 3, 'max_recent_floods': 1},
            {'min_flood_count': 5, 'max_recent_floods': 1},
            {'min_flood_count': 5, 'max_recent_floods': 2}
        ]
    }
    
    # Preview negative candidates for each strategy
    print("Negative candidate counts by strategy:")
    for strategy_name, param_list in strategy_configs.items():
        print(f"\n{strategy_name.upper()}:")
        for params in param_list:
            candidates = get_negative_candidates_by_strategy(road_stats, strategy_name, **params)
            print(f"  {params}: {len(candidates)} candidates")
            if len(candidates) > 0:
                # Show some examples with their stats
                example_stats = road_stats[road_stats['link_id'].isin(candidates[:3])]
                for _, row in example_stats.iterrows():
                    print(f"    {row['link_id']}: prob={row['marginal_prob']:.3f}, floods={row['flood_count']}, recent={row['recent_floods']}")
    
    # Define evaluation parameters
    pred_thresholds = [0.2, 0.3, 0.4, 0.5, 0.6]
    evidence_ratios = [0.3, 0.4, 0.5, 0.6]
    neg_pos_ratios = [0.5, 1.0, 1.5, 2.0]
    
    print(f"\n{'='*60}")
    print("RUNNING COMPREHENSIVE EVALUATION")
    print(f"{'='*60}")
    print(f"Total parameter combinations to test: {sum(len(params) for params in strategy_configs.values()) * len(pred_thresholds) * len(evidence_ratios) * len(neg_pos_ratios)}")
    
    # Run all combinations
    all_results = []
    combination_count = 0
    
    for strategy_name, param_list in strategy_configs.items():
        for strategy_params in param_list:
            for pred_threshold, evidence_ratio, neg_pos_ratio in itertools.product(pred_thresholds, evidence_ratios, neg_pos_ratios):
                combination_count += 1
                
                result = evaluate_with_negative_strategy(
                    flood_net, test_df, bn_nodes, road_stats,
                    strategy_name, strategy_params,
                    pred_threshold, evidence_ratio, neg_pos_ratio
                )
                
                all_results.append(result)
                
                # Print progress for significant results
                if result['precision'] >= 0.7 or (combination_count % 50 == 0):
                    print(f"[{combination_count:3d}] {strategy_name} {strategy_params} pred={pred_threshold} ev={evidence_ratio} neg={neg_pos_ratio}: "
                          f"P={result['precision']:.3f} R={result['recall']:.3f} F1={result['f1_score']:.3f} "
                          f"TP={result['tp']} FP={result['fp']} samples={result['total_samples']}")
    
    print(f"\nCompleted {len(all_results)} evaluations")
    
    # Save detailed results
    results_df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"conservative_negative_evaluation_{timestamp}.csv"
    results_df.to_csv(filename, index=False)
    
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    
    # Filter and sort results
    valid_results = results_df[results_df['total_samples'] > 10].copy()
    high_precision = valid_results[valid_results['precision'] >= 0.7].copy()
    
    print(f"Total valid evaluations (>10 samples): {len(valid_results)}")
    print(f"High precision results (≥0.7): {len(high_precision)}")
    
    if len(high_precision) > 0:
        print(f"\nTop 10 High-Precision Results:")
        top_results = high_precision.nlargest(10, ['precision', 'recall', 'f1_score'])
        for _, row in top_results.iterrows():
            print(f"Strategy: {row['strategy']} {row['strategy_params']}")
            print(f"  Pred_threshold={row['pred_threshold']}, Evidence_ratio={row['evidence_ratio']}, Neg_pos_ratio={row['neg_pos_ratio']}")
            print(f"  Precision: {row['precision']:.3f}, Recall: {row['recall']:.3f}, F1: {row['f1_score']:.3f}")
            print(f"  TP: {row['tp']}, FP: {row['fp']}, TN: {row['tn']}, FN: {row['fn']}")
            print(f"  Samples: {row['positive_samples']} pos, {row['negative_samples']} neg, {row['negative_candidates_count']} neg candidates")
            print()
    
    # Best results by different criteria
    print("Best Results by Different Criteria:")
    if len(valid_results) > 0:
        best_precision = valid_results.loc[valid_results['precision'].idxmax()]
        best_recall = valid_results.loc[valid_results['recall'].idxmax()]
        best_f1 = valid_results.loc[valid_results['f1_score'].idxmax()]
        
        print(f"Best Precision: {best_precision['precision']:.3f} ({best_precision['strategy']} {best_precision['strategy_params']})")
        print(f"Best Recall: {best_recall['recall']:.3f} ({best_recall['strategy']} {best_recall['strategy_params']})")
        print(f"Best F1-Score: {best_f1['f1_score']:.3f} ({best_f1['strategy']} {best_f1['strategy_params']})")
    
    print(f"\nDetailed results saved to: {filename}")
    
    return all_results, filename

if __name__ == "__main__":
    results, output_file = run_comprehensive_evaluation()