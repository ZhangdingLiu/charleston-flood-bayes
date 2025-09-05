#!/usr/bin/env python3
"""
Focused Network Evaluation
Build smaller Bayesian networks using high-probability roads and evaluate on ALL network roads
This addresses the deployment reality gap between conservative negative sampling and actual usage
"""

import random
import numpy as np
import pandas as pd
from model import FloodBayesNetwork
import warnings
from datetime import datetime
import time
import json
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def load_and_preprocess_data():
    """Load and preprocess the flood data"""
    print("=" * 80)
    print("FOCUSED NETWORK EVALUATION")
    print("Build smaller networks with high-probability roads, evaluate on ALL roads")
    print("=" * 80)
    
    # Load data
    df = pd.read_csv("Road_Closures_2024.csv")
    df = df[df["REASON"].str.upper() == "FLOOD"].copy()
    
    # Preprocess
    df["time_create"] = pd.to_datetime(df["START"], utc=True)
    df["link_id"] = df["STREET"].str.upper().str.replace(" ", "_")
    df["link_id"] = df["link_id"].astype(str)
    df["id"] = df["OBJECTID"].astype(str)
    
    # Temporal split (70% train, 30% test)
    df_sorted = df.sort_values('time_create')
    split_idx = int(len(df_sorted) * 0.7)
    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()
    
    print(f"Data loaded: {len(df)} total flood records")
    print(f"Train set: {len(train_df)} records")
    print(f"Test set: {len(test_df)} records")
    
    return train_df, test_df

def analyze_road_statistics(train_df):
    """Analyze road flooding statistics for selection strategies"""
    print(f"\n{'='*60}")
    print("ROAD STATISTICS ANALYSIS")
    print(f"{'='*60}")
    
    # Calculate road statistics
    road_stats = train_df.groupby('link_id').agg({
        'time_create': 'count',
        'id': 'nunique'
    }).reset_index()
    road_stats.columns = ['link_id', 'flood_count', 'unique_incidents']
    
    # Calculate marginal probabilities (rough estimate based on training data)
    total_days = (train_df['time_create'].max() - train_df['time_create'].min()).days
    road_stats['marginal_prob_estimate'] = road_stats['flood_count'] / total_days
    
    # Sort by different criteria
    road_stats = road_stats.sort_values('flood_count', ascending=False)
    
    print(f"Total unique roads in training data: {len(road_stats)}")
    print(f"Road flood count range: {road_stats['flood_count'].min()} - {road_stats['flood_count'].max()}")
    print(f"\nTop 10 roads by flood frequency:")
    for i, row in road_stats.head(10).iterrows():
        print(f"  {row['link_id']:<25} | Floods: {row['flood_count']:3d} | Est. prob: {row['marginal_prob_estimate']:.4f}")
    
    return road_stats

def select_high_probability_roads(road_stats, strategy, n_roads):
    """
    Select high-probability roads using different strategies
    
    Args:
        road_stats: DataFrame with road statistics
        strategy: Selection strategy ('frequency', 'probability', 'combined')
        n_roads: Number of roads to select
    
    Returns:
        List of selected road IDs
    """
    
    if strategy == 'frequency':
        # Select by flood frequency
        selected = road_stats.nlargest(n_roads, 'flood_count')['link_id'].tolist()
        
    elif strategy == 'probability':
        # Select by estimated marginal probability
        selected = road_stats.nlargest(n_roads, 'marginal_prob_estimate')['link_id'].tolist()
        
    elif strategy == 'combined':
        # Combined score: normalize both metrics and take weighted average
        freq_norm = road_stats['flood_count'] / road_stats['flood_count'].max()
        prob_norm = road_stats['marginal_prob_estimate'] / road_stats['marginal_prob_estimate'].max()
        road_stats_copy = road_stats.copy()
        road_stats_copy['combined_score'] = 0.6 * freq_norm + 0.4 * prob_norm
        selected = road_stats_copy.nlargest(n_roads, 'combined_score')['link_id'].tolist()
        
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return selected

def build_focused_network(train_df, selected_roads, occ_thr=3, edge_thr=2, weight_thr=0.3, max_parents=2):
    """
    Build Bayesian network using only selected high-probability roads
    
    Args:
        train_df: Training data
        selected_roads: List of road IDs to include in network
        Network construction parameters
    
    Returns:
        Tuple of (flood_net, bn_nodes, marginals_dict, network_stats)
    """
    
    # Filter training data to only include selected roads
    filtered_train_df = train_df[train_df['link_id'].isin(selected_roads)].copy()
    
    if len(filtered_train_df) == 0:
        return None, set(), {}, {}
    
    # Build network
    flood_net = FloodBayesNetwork(t_window="D")
    flood_net.fit_marginal(filtered_train_df)
    flood_net.build_network_by_co_occurrence(
        filtered_train_df, occ_thr=occ_thr, edge_thr=edge_thr, weight_thr=weight_thr, report=False
    )
    flood_net.fit_conditional(filtered_train_df, max_parents=max_parents, alpha=1.0)
    flood_net.build_bayes_network()
    
    bn_nodes = set(flood_net.network_bayes.nodes())
    marginals_dict = dict(zip(flood_net.marginals['link_id'], flood_net.marginals['p']))
    
    # Calculate network statistics
    edges = list(flood_net.network_bayes.edges())
    parent_counts = [len(list(flood_net.network_bayes.predecessors(node))) for node in bn_nodes]
    
    network_stats = {
        'selected_roads_count': len(selected_roads),
        'final_nodes_count': len(bn_nodes),
        'edges_count': len(edges),
        'avg_parents': np.mean(parent_counts) if parent_counts else 0,
        'max_parents': np.max(parent_counts) if parent_counts else 0,
        'nodes_with_no_parents': sum(1 for count in parent_counts if count == 0),
        'coverage_ratio': len(bn_nodes) / len(selected_roads) if len(selected_roads) > 0 else 0
    }
    
    return flood_net, bn_nodes, marginals_dict, network_stats

def evaluate_focused_network(flood_net, test_df, bn_nodes, pred_threshold, evidence_ratio):
    """
    Evaluate focused network on ALL network roads (no negative filtering)
    
    Args:
        flood_net: Trained Bayesian network
        test_df: Test dataset
        bn_nodes: Set of nodes in the network
        pred_threshold: Probability threshold for flood prediction
        evidence_ratio: Ratio of flooded roads to use as evidence
    
    Returns:
        Dictionary with evaluation results
    """
    
    if len(bn_nodes) == 0:
        return None
    
    # Get test roads that are in the network
    test_roads = set(test_df["link_id"].unique())
    test_roads_in_bn = test_roads.intersection(bn_nodes)
    
    if len(test_roads_in_bn) == 0:
        return None
    
    results = []
    test_by_date = test_df.groupby(test_df["time_create"].dt.floor("D"))
    evaluated_days = 0
    skipped_days = 0
    
    for date, day_group in test_by_date:
        flooded_roads = list(day_group["link_id"].unique())
        flooded_in_bn = [road for road in flooded_roads if road in bn_nodes]
        
        # Need at least 1 evidence road and 1 target road
        if len(flooded_in_bn) < 1:
            skipped_days += 1
            continue
            
        # If only 1 flooded road in network, use external evidence or skip
        if len(flooded_in_bn) == 1:
            # Use the single flooded road as evidence
            evidence_roads = flooded_in_bn
            target_roads = [road for road in test_roads_in_bn if road not in evidence_roads]
        else:
            # Normal case: use some flooded roads as evidence
            evidence_count = max(1, int(len(flooded_in_bn) * evidence_ratio))
            evidence_count = min(evidence_count, len(flooded_in_bn) - 1)
            evidence_roads = flooded_in_bn[:evidence_count]
            target_roads = [road for road in test_roads_in_bn if road not in evidence_roads]
        
        if len(target_roads) == 0:
            skipped_days += 1
            continue
            
        evaluated_days += 1
        evidence = {road: 1 for road in evidence_roads}
        
        # Predict for ALL target roads in the network
        for target_road in target_roads:
            try:
                result = flood_net.infer_w_evidence(target_road, evidence)
                prob_flood = result['flooded']
                prediction = 1 if prob_flood >= pred_threshold else 0
                
                # Ground truth: did this road actually flood on this day?
                true_label = 1 if target_road in flooded_roads else 0
                
                results.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'target_road': target_road,
                    'prob_flood': prob_flood,
                    'prediction': prediction,
                    'true_label': true_label,
                    'is_correct': (prediction == true_label)
                })
                
            except Exception as e:
                # Skip roads that can't be predicted
                continue
    
    if len(results) == 0:
        return None
    
    # Calculate metrics
    tp = sum(1 for r in results if r['true_label'] == 1 and r['prediction'] == 1)
    fp = sum(1 for r in results if r['true_label'] == 0 and r['prediction'] == 1)
    fn = sum(1 for r in results if r['true_label'] == 1 and r['prediction'] == 0)
    tn = sum(1 for r in results if r['true_label'] == 0 and r['prediction'] == 0)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
    
    return {
        'pred_threshold': pred_threshold,
        'evidence_ratio': evidence_ratio,
        'evaluated_days': evaluated_days,
        'skipped_days': skipped_days,
        'test_roads_in_network': len(test_roads_in_bn),
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'precision': precision, 'recall': recall, 'f1_score': f1_score, 'accuracy': accuracy,
        'total_samples': len(results),
        'positive_samples': sum(1 for r in results if r['true_label'] == 1),
        'negative_samples': sum(1 for r in results if r['true_label'] == 0),
        'positive_predictions': sum(1 for r in results if r['prediction'] == 1),
        'negative_predictions': sum(1 for r in results if r['prediction'] == 0)
    }

def run_focused_network_evaluation():
    """Run comprehensive focused network evaluation"""
    
    start_time = time.time()
    
    # Load data and analyze road statistics
    train_df, test_df = load_and_preprocess_data()
    road_stats = analyze_road_statistics(train_df)
    
    print(f"\n{'='*60}")
    print("FOCUSED NETWORK CONFIGURATIONS")
    print(f"{'='*60}")
    
    # Define evaluation configurations
    network_configs = [
        # (strategy, n_roads, occ_thr, edge_thr, weight_thr)
        ('frequency', 15, 3, 2, 0.3),
        ('frequency', 20, 3, 2, 0.3),
        ('frequency', 25, 3, 2, 0.3),
        ('frequency', 30, 3, 2, 0.3),
        
        ('frequency', 20, 2, 1, 0.2),  # More permissive
        ('frequency', 20, 3, 2, 0.3),  # Default
        ('frequency', 20, 4, 3, 0.4),  # More restrictive
        
        ('combined', 15, 3, 2, 0.3),
        ('combined', 20, 3, 2, 0.3),
        ('combined', 25, 3, 2, 0.3),
    ]
    
    eval_params = [
        # (pred_threshold, evidence_ratio)
        (0.15, 0.4), (0.15, 0.5),
        (0.2, 0.4), (0.2, 0.5),
        (0.25, 0.4), (0.25, 0.5),
        (0.3, 0.4), (0.3, 0.5),
        (0.35, 0.4), (0.4, 0.4),
    ]
    
    total_combinations = len(network_configs) * len(eval_params)
    print(f"Testing {total_combinations} total combinations")
    print(f"Network configurations: {len(network_configs)}")
    print(f"Evaluation parameters: {len(eval_params)}")
    
    # Store all results
    all_results = []
    network_info = []
    combination_count = 0
    
    # Test each network configuration
    for strategy, n_roads, occ_thr, edge_thr, weight_thr in network_configs:
        print(f"\n{'='*50}")
        print(f"BUILDING NETWORK: {strategy}, {n_roads} roads, occ={occ_thr}, edge={edge_thr}, weight={weight_thr}")
        print(f"{'='*50}")
        
        # Select roads and build network
        selected_roads = select_high_probability_roads(road_stats, strategy, n_roads)
        flood_net, bn_nodes, marginals_dict, network_stats = build_focused_network(
            train_df, selected_roads, occ_thr, edge_thr, weight_thr
        )
        
        if flood_net is None or len(bn_nodes) == 0:
            print(f"Failed to build network with {n_roads} roads using {strategy} strategy")
            continue
        
        print(f"Network built: {network_stats['final_nodes_count']} nodes, {network_stats['edges_count']} edges")
        print(f"Selected {n_roads} roads → {network_stats['final_nodes_count']} final nodes ({network_stats['coverage_ratio']:.2f} coverage)")
        
        # Store network info
        network_info.append({
            'strategy': strategy,
            'n_roads_requested': n_roads,
            'occ_thr': occ_thr,
            'edge_thr': edge_thr,
            'weight_thr': weight_thr,
            'selected_roads': selected_roads,
            'final_nodes': list(bn_nodes),
            **network_stats
        })
        
        # Test each evaluation parameter combination
        for pred_threshold, evidence_ratio in eval_params:
            combination_count += 1
            
            print(f"  [{combination_count:3d}/{total_combinations}] pred_thresh={pred_threshold}, ev_ratio={evidence_ratio}", end=" ")
            
            # Run evaluation
            eval_result = evaluate_focused_network(
                flood_net, test_df, bn_nodes, pred_threshold, evidence_ratio
            )
            
            if eval_result is None:
                print("→ SKIPPED (no valid evaluation)")
                continue
            
            # Add network configuration info to result
            result = {
                'strategy': strategy,
                'n_roads_requested': n_roads,
                'occ_thr': occ_thr,
                'edge_thr': edge_thr,
                'weight_thr': weight_thr,
                'final_nodes_count': network_stats['final_nodes_count'],
                'edges_count': network_stats['edges_count'],
                'coverage_ratio': network_stats['coverage_ratio'],
                **eval_result
            }
            
            all_results.append(result)
            
            # Print results
            print(f"→ P={result['precision']:.3f}, R={result['recall']:.3f}, F1={result['f1_score']:.3f}")
            print(f"      TP={result['tp']}, FP={result['fp']}, TN={result['tn']}, FN={result['fn']} | "
                  f"Samples={result['total_samples']} ({result['positive_samples']} pos)")
    
    # Save results
    elapsed_time = time.time() - start_time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print(f"{'='*60}")
    print(f"Total evaluation time: {elapsed_time:.2f} seconds")
    print(f"Total combinations evaluated: {len(all_results)}")
    
    # Save summary results
    if all_results:
        results_df = pd.DataFrame(all_results)
        summary_file = f"focused_network_evaluation_{timestamp}.csv"
        results_df.to_csv(summary_file, index=False)
        print(f"Summary results saved to: {summary_file}")
    
    # Save network information
    if network_info:
        network_file = f"focused_network_info_{timestamp}.json"
        with open(network_file, 'w') as f:
            json.dump(network_info, f, indent=2, default=str)
        print(f"Network information saved to: {network_file}")
    
    # Analysis
    if all_results:
        analyze_results(results_df)
    
    return all_results, network_info

def analyze_results(results_df):
    """Analyze and summarize the evaluation results"""
    print(f"\n{'='*60}")
    print("RESULTS ANALYSIS")
    print(f"{'='*60}")
    
    # Overall statistics
    print(f"Total configurations evaluated: {len(results_df)}")
    print(f"Network sizes tested: {sorted(results_df['final_nodes_count'].unique())}")
    print(f"Strategies tested: {results_df['strategy'].unique().tolist()}")
    
    # Best results overall
    if len(results_df) > 0:
        best_precision_idx = results_df['precision'].idxmax()
        best_recall_idx = results_df['recall'].idxmax()
        best_f1_idx = results_df['f1_score'].idxmax()
        
        print(f"\nBest Overall Results:")
        best_p = results_df.loc[best_precision_idx]
        print(f"Best Precision: {best_p['precision']:.3f} | Strategy: {best_p['strategy']}, {best_p['final_nodes_count']} nodes, thresh={best_p['pred_threshold']}")
        
        best_r = results_df.loc[best_recall_idx]
        print(f"Best Recall: {best_r['recall']:.3f} | Strategy: {best_r['strategy']}, {best_r['final_nodes_count']} nodes, thresh={best_r['pred_threshold']}")
        
        best_f1 = results_df.loc[best_f1_idx]
        print(f"Best F1-Score: {best_f1['f1_score']:.3f} | Strategy: {best_f1['strategy']}, {best_f1['final_nodes_count']} nodes, thresh={best_f1['pred_threshold']}")
    
    # Network size analysis
    print(f"\nNetwork Size vs Performance:")
    for size in sorted(results_df['final_nodes_count'].unique()):
        subset = results_df[results_df['final_nodes_count'] == size]
        if len(subset) > 0:
            avg_precision = subset['precision'].mean()
            avg_recall = subset['recall'].mean()
            avg_f1 = subset['f1_score'].mean()
            best_precision = subset['precision'].max()
            print(f"  {size:2d} nodes: Avg P={avg_precision:.3f}, R={avg_recall:.3f}, F1={avg_f1:.3f} | Best P={best_precision:.3f}")
    
    # High precision results
    high_precision = results_df[results_df['precision'] >= 0.5]
    print(f"\nHigh Precision Results (≥0.5): {len(high_precision)}")
    if len(high_precision) > 0:
        print("Top high-precision configurations:")
        top_hp = high_precision.nlargest(5, ['precision', 'f1_score'])
        for _, row in top_hp.iterrows():
            print(f"  {row['strategy']} {row['final_nodes_count']} nodes, thresh={row['pred_threshold']}: "
                  f"P={row['precision']:.3f}, R={row['recall']:.3f}, F1={row['f1_score']:.3f}")
    
    # Balanced results
    balanced = results_df[(results_df['precision'] >= 0.3) & (results_df['recall'] >= 0.3)]
    print(f"\nBalanced Results (P≥0.3, R≥0.3): {len(balanced)}")
    if len(balanced) > 0:
        print("Top balanced configurations:")
        top_balanced = balanced.nlargest(5, 'f1_score')
        for _, row in top_balanced.iterrows():
            print(f"  {row['strategy']} {row['final_nodes_count']} nodes, thresh={row['pred_threshold']}: "
                  f"P={row['precision']:.3f}, R={row['recall']:.3f}, F1={row['f1_score']:.3f}")

def main():
    """Main evaluation function"""
    try:
        print("Starting focused network evaluation...")
        print("This may take several minutes to complete all combinations...")
        
        results, network_info = run_focused_network_evaluation()
        
        print(f"\n{'='*80}")
        print("FOCUSED NETWORK EVALUATION COMPLETE")
        print(f"{'='*80}")
        print("Results saved. Check the CSV and JSON files for detailed analysis.")
        
        return results, network_info
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    results, network_info = main()