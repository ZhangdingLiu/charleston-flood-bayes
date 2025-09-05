#!/usr/bin/env python3
"""
Focused No Negative Filtering Evaluation
Quick evaluation with focused parameter range based on initial observations
"""

import random
import numpy as np
import pandas as pd
from model import FloodBayesNetwork
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def load_and_preprocess_data():
    """Load and preprocess the flood data"""
    print("=" * 80)
    print("FOCUSED NO NEGATIVE FILTERING EVALUATION")
    print("Based on initial results: focus on promising parameter ranges")
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

def build_bayesian_network(train_df, test_df):
    """Build the Bayesian network"""
    print(f"\n{'='*60}")
    print("BUILDING BAYESIAN NETWORK")
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
    
    print(f"Network built with {len(bn_nodes)} nodes")
    
    # Get test set coverage
    test_roads = set(test_df["link_id"].unique())
    test_roads_in_bn = test_roads.intersection(bn_nodes)
    
    print(f"Test roads in network: {len(test_roads_in_bn)}/{len(test_roads)} ({len(test_roads_in_bn)/len(test_roads)*100:.1f}%)")
    
    return flood_net, bn_nodes, test_roads_in_bn

def evaluate_configuration(flood_net, test_df, bn_nodes, test_roads_in_bn, 
                         pred_threshold, evidence_ratio):
    """Quick evaluation of a specific configuration"""
    
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
        evidence_count = min(evidence_count, len(flooded_in_bn) - 1)
        evidence_roads = flooded_in_bn[:evidence_count]
        evidence = {road: 1 for road in evidence_roads}
        
        # Target roads: ALL roads in network that appear in test set, excluding evidence
        target_roads = [road for road in test_roads_in_bn if road not in evidence_roads]
        
        # Predict for ALL target roads
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
                
            except:
                continue
    
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
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'precision': precision, 'recall': recall, 'f1_score': f1_score, 'accuracy': accuracy,
        'total_samples': len(results),
        'positive_samples': sum(1 for r in results if r['true_label'] == 1),
        'negative_samples': sum(1 for r in results if r['true_label'] == 0),
        'positive_predictions': sum(1 for r in results if r['prediction'] == 1),
        'sample_details': results
    }

def run_focused_evaluation():
    """Run focused evaluation with promising parameter ranges"""
    
    # Load data and build network
    train_df, test_df = load_and_preprocess_data()
    flood_net, bn_nodes, test_roads_in_bn = build_bayesian_network(train_df, test_df)
    
    print(f"\n{'='*60}")
    print("FOCUSED PARAMETER EVALUATION")
    print(f"{'='*60}")
    
    # Focused parameter ranges based on initial observations
    pred_thresholds = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]  # Focus on balanced range
    evidence_ratios = [0.3, 0.4, 0.5, 0.6]  # Focus on most practical ratios
    
    print(f"Testing {len(pred_thresholds) * len(evidence_ratios)} focused combinations")
    print(f"Prediction thresholds: {pred_thresholds}")
    print(f"Evidence ratios: {evidence_ratios}")
    
    # Run focused combinations
    all_results = []
    combination_count = 0
    
    for pred_thresh in pred_thresholds:
        for ev_ratio in evidence_ratios:
            combination_count += 1
            
            print(f"\n[{combination_count:2d}] Testing pred_thresh={pred_thresh}, ev_ratio={ev_ratio}")
            
            result = evaluate_configuration(
                flood_net, test_df, bn_nodes, test_roads_in_bn,
                pred_thresh, ev_ratio
            )
            
            all_results.append(result)
            
            # Print results
            print(f"    Results: P={result['precision']:.3f}, R={result['recall']:.3f}, F1={result['f1_score']:.3f}, "
                  f"Acc={result['accuracy']:.3f}")
            print(f"    Confusion: TP={result['tp']}, FP={result['fp']}, TN={result['tn']}, FN={result['fn']}")
            print(f"    Samples: {result['total_samples']} total ({result['positive_samples']} pos, {result['negative_samples']} neg)")
            print(f"    Predictions: {result['positive_predictions']} flood predictions")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Summary results
    summary_results = []
    for result in all_results:
        summary_results.append({
            'pred_threshold': result['pred_threshold'],
            'evidence_ratio': result['evidence_ratio'],
            'evaluated_days': result['evaluated_days'],
            'tp': result['tp'], 'fp': result['fp'], 'tn': result['tn'], 'fn': result['fn'],
            'precision': result['precision'], 'recall': result['recall'], 
            'f1_score': result['f1_score'], 'accuracy': result['accuracy'],
            'total_samples': result['total_samples'],
            'positive_samples': result['positive_samples'],
            'negative_samples': result['negative_samples'],
            'positive_predictions': result['positive_predictions']
        })
    
    summary_df = pd.DataFrame(summary_results)
    summary_file = f"focused_no_filtering_summary_{timestamp}.csv"
    summary_df.to_csv(summary_file, index=False)
    
    # Analysis
    print(f"\n{'='*60}")
    print("RESULTS ANALYSIS")
    print(f"{'='*60}")
    
    # Best results by different criteria
    best_precision_idx = summary_df['precision'].idxmax()
    best_recall_idx = summary_df['recall'].idxmax()
    best_f1_idx = summary_df['f1_score'].idxmax()
    best_accuracy_idx = summary_df['accuracy'].idxmax()
    
    print(f"\nBest Results:")
    print(f"Best Precision: {summary_df.loc[best_precision_idx, 'precision']:.3f} "
          f"(thresh={summary_df.loc[best_precision_idx, 'pred_threshold']}, "
          f"ev_ratio={summary_df.loc[best_precision_idx, 'evidence_ratio']})")
    
    print(f"Best Recall: {summary_df.loc[best_recall_idx, 'recall']:.3f} "
          f"(thresh={summary_df.loc[best_recall_idx, 'pred_threshold']}, "
          f"ev_ratio={summary_df.loc[best_recall_idx, 'evidence_ratio']})")
    
    print(f"Best F1-Score: {summary_df.loc[best_f1_idx, 'f1_score']:.3f} "
          f"(thresh={summary_df.loc[best_f1_idx, 'pred_threshold']}, "
          f"ev_ratio={summary_df.loc[best_f1_idx, 'evidence_ratio']})")
    
    print(f"Best Accuracy: {summary_df.loc[best_accuracy_idx, 'accuracy']:.3f} "
          f"(thresh={summary_df.loc[best_accuracy_idx, 'pred_threshold']}, "
          f"ev_ratio={summary_df.loc[best_accuracy_idx, 'evidence_ratio']})")
    
    # High precision results
    high_precision = summary_df[summary_df['precision'] >= 0.5].copy()
    print(f"\nHigh Precision Results (≥0.5): {len(high_precision)}")
    
    if len(high_precision) > 0:
        print(f"High-Precision Configurations:")
        for _, row in high_precision.iterrows():
            print(f"  thresh={row['pred_threshold']:.2f}, ev_ratio={row['evidence_ratio']:.1f} | "
                  f"P={row['precision']:.3f}, R={row['recall']:.3f}, F1={row['f1_score']:.3f}")
    
    # Balanced results (good precision and recall)
    balanced = summary_df[(summary_df['precision'] >= 0.3) & (summary_df['recall'] >= 0.3)].copy()
    print(f"\nBalanced Results (P≥0.3, R≥0.3): {len(balanced)}")
    
    if len(balanced) > 0:
        balanced_sorted = balanced.nlargest(5, 'f1_score')
        print(f"Top Balanced Configurations:")
        for _, row in balanced_sorted.iterrows():
            print(f"  thresh={row['pred_threshold']:.2f}, ev_ratio={row['evidence_ratio']:.1f} | "
                  f"P={row['precision']:.3f}, R={row['recall']:.3f}, F1={row['f1_score']:.3f}")
    
    # Save detailed results for top 3 configurations
    top_configs = [
        all_results[best_precision_idx],
        all_results[best_f1_idx],
        all_results[best_recall_idx]
    ]
    
    # Remove duplicates
    unique_configs = []
    seen = set()
    for config in top_configs:
        key = (config['pred_threshold'], config['evidence_ratio'])
        if key not in seen:
            unique_configs.append(config)
            seen.add(key)
    
    # Save sample-by-sample details for top configs
    detailed_results = []
    for i, config in enumerate(unique_configs):
        config_name = f"thresh_{config['pred_threshold']}_ev_{config['evidence_ratio']}"
        rank = ['best_precision', 'best_f1', 'best_recall'][i] if i < 3 else f'config_{i+1}'
        
        for sample in config['sample_details']:
            detailed_results.append({
                'config_name': config_name,
                'config_rank': rank,
                'pred_threshold': config['pred_threshold'],
                'evidence_ratio': config['evidence_ratio'],
                'date': sample['date'],
                'target_road': sample['target_road'],
                'prob_flood': sample['prob_flood'],
                'prediction': sample['prediction'],
                'true_label': sample['true_label'],
                'is_correct': sample['is_correct']
            })
    
    if detailed_results:
        detailed_df = pd.DataFrame(detailed_results)
        detailed_file = f"focused_no_filtering_detailed_{timestamp}.csv"
        detailed_df.to_csv(detailed_file, index=False)
        print(f"\nDetailed results saved to: {detailed_file}")
    
    print(f"\nSummary results saved to: {summary_file}")
    print(f"Total configurations tested: {len(all_results)}")
    
    return all_results, summary_file

if __name__ == "__main__":
    try:
        results, summary_file = run_focused_evaluation()
        print(f"\n{'='*80}")
        print("FOCUSED EVALUATION COMPLETE")
        print(f"{'='*80}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()