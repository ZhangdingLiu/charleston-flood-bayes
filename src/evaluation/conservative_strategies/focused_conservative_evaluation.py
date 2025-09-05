#!/usr/bin/env python3
"""
Focused Conservative Negative Evaluation
Based on promising results from comprehensive evaluation, focus on high-performing combinations
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
    """Load and preprocess the flood data with additional analysis"""
    print("=" * 80)
    print("FOCUSED CONSERVATIVE NEGATIVE EVALUATION")
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
    print(f"Train set: {len(train_df)} records")
    print(f"Test set: {len(test_df)} records")
    
    return train_df, test_df

def build_bayesian_network(train_df):
    """Build the Bayesian network and analyze road statistics"""
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
    print(f"Marginal probability range: {road_stats['marginal_prob'].min():.4f} - {road_stats['marginal_prob'].max():.4f}")
    
    return flood_net, bn_nodes, marginals_dict, road_stats

def get_negative_candidates(road_stats, strategy, **params):
    """Get negative candidates based on strategy"""
    
    if strategy == "marginal_prob":
        threshold = params.get('threshold', 0.1)
        candidates = road_stats[road_stats['marginal_prob'] <= threshold]['link_id'].tolist()
        
    elif strategy == "recent_activity":
        max_recent = params.get('max_recent', 1)
        candidates = road_stats[road_stats['recent_floods'] <= max_recent]['link_id'].tolist()
        
    elif strategy == "combined":
        prob_thresh = params.get('prob_threshold', 0.15)
        max_recent = params.get('max_recent', 2)
        candidates = road_stats[
            (road_stats['marginal_prob'] <= prob_thresh) & 
            (road_stats['recent_floods'] <= max_recent)
        ]['link_id'].tolist()
        
    else:
        candidates = []
    
    return candidates

def evaluate_configuration(flood_net, test_df, bn_nodes, negative_candidates, 
                         pred_threshold, evidence_ratio, neg_pos_ratio, config_name):
    """Evaluate a specific configuration"""
    
    if len(negative_candidates) == 0:
        return None
    
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
    
    # Calculate metrics
    tp = sum(1 for r in results if r['type'] == 'Positive' and r['prediction'] == 1)
    fp = sum(1 for r in results if r['type'] == 'Negative' and r['prediction'] == 1)
    fn = sum(1 for r in results if r['type'] == 'Positive' and r['prediction'] == 0)
    tn = sum(1 for r in results if r['type'] == 'Negative' and r['prediction'] == 0)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
    
    pos_samples = sum(1 for r in results if r['type'] == 'Positive')
    neg_samples = sum(1 for r in results if r['type'] == 'Negative')
    
    return {
        'config_name': config_name,
        'pred_threshold': pred_threshold,
        'evidence_ratio': evidence_ratio,
        'neg_pos_ratio': neg_pos_ratio,
        'negative_candidates_count': len(negative_candidates),
        'negative_examples': negative_candidates[:5],
        'evaluated_days': evaluated_days,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy,
        'positive_samples': pos_samples,
        'negative_samples': neg_samples,
        'total_samples': len(results),
        'sample_details': results
    }

def run_focused_evaluation():
    """Run focused evaluation on most promising configurations"""
    
    # Load data and build network
    train_df, test_df = load_and_preprocess_data()
    flood_net, bn_nodes, marginals_dict, road_stats = build_bayesian_network(train_df)
    
    print(f"\n{'='*60}")
    print("ANALYZING NEGATIVE CANDIDATE STRATEGIES")
    print(f"{'='*60}")
    
    # Define focused configurations based on promising patterns
    configurations = [
        # Strategy 1: Marginal Probability based (most promising from partial results)
        {
            'name': 'MargProb_0.05',
            'strategy': 'marginal_prob',
            'params': {'threshold': 0.05},
            'eval_configs': [
                (0.2, 0.5, 0.5), (0.3, 0.5, 0.5), (0.4, 0.5, 0.5),
                (0.2, 0.4, 1.0), (0.3, 0.4, 1.0), (0.4, 0.4, 1.0)
            ]
        },
        {
            'name': 'MargProb_0.10',
            'strategy': 'marginal_prob',
            'params': {'threshold': 0.10},
            'eval_configs': [
                (0.2, 0.5, 0.5), (0.3, 0.5, 0.5), (0.4, 0.5, 0.5),
                (0.2, 0.4, 1.0), (0.3, 0.4, 1.0), (0.4, 0.4, 1.0)
            ]
        },
        {
            'name': 'MargProb_0.20',
            'strategy': 'marginal_prob',
            'params': {'threshold': 0.20},
            'eval_configs': [
                (0.2, 0.5, 0.5), (0.3, 0.5, 0.5), (0.4, 0.5, 0.5),
                (0.2, 0.4, 1.0), (0.3, 0.4, 1.0), (0.4, 0.4, 1.0)
            ]
        },
        
        # Strategy 2: Recent Activity based
        {
            'name': 'RecentActivity_0',
            'strategy': 'recent_activity',
            'params': {'max_recent': 0},
            'eval_configs': [
                (0.2, 0.5, 0.5), (0.3, 0.5, 0.5), (0.4, 0.5, 0.5),
                (0.2, 0.4, 1.0), (0.3, 0.4, 1.0)
            ]
        },
        {
            'name': 'RecentActivity_1',
            'strategy': 'recent_activity',
            'params': {'max_recent': 1},
            'eval_configs': [
                (0.2, 0.5, 0.5), (0.3, 0.5, 0.5), (0.4, 0.5, 0.5),
                (0.2, 0.4, 1.0), (0.3, 0.4, 1.0)
            ]
        },
        
        # Strategy 3: Combined Conservative
        {
            'name': 'Combined_Conservative',
            'strategy': 'combined',
            'params': {'prob_threshold': 0.15, 'max_recent': 2},
            'eval_configs': [
                (0.2, 0.5, 0.5), (0.3, 0.5, 0.5), (0.4, 0.5, 0.5),
                (0.2, 0.4, 1.0), (0.3, 0.4, 1.0)
            ]
        }
    ]
    
    # Show negative candidates for each strategy
    for config in configurations:
        candidates = get_negative_candidates(road_stats, config['strategy'], **config['params'])
        print(f"\n{config['name']}: {len(candidates)} candidates")
        if len(candidates) > 0:
            examples = road_stats[road_stats['link_id'].isin(candidates[:3])]
            for _, row in examples.iterrows():
                print(f"  {row['link_id']}: prob={row['marginal_prob']:.3f}, floods={row['flood_count']}, recent={row['recent_floods']}")
    
    print(f"\n{'='*60}")
    print("RUNNING FOCUSED EVALUATIONS")
    print(f"{'='*60}")
    
    all_results = []
    
    for config in configurations:
        negative_candidates = get_negative_candidates(road_stats, config['strategy'], **config['params'])
        
        if len(negative_candidates) == 0:
            print(f"Skipping {config['name']} - no candidates")
            continue
            
        print(f"\nTesting {config['name']} with {len(negative_candidates)} negative candidates:")
        
        for pred_thresh, ev_ratio, neg_ratio in config['eval_configs']:
            result = evaluate_configuration(
                flood_net, test_df, bn_nodes, negative_candidates,
                pred_thresh, ev_ratio, neg_ratio, config['name']
            )
            
            if result:
                all_results.append(result)
                print(f"  pred={pred_thresh} ev={ev_ratio} neg={neg_ratio}: "
                      f"P={result['precision']:.3f} R={result['recall']:.3f} F1={result['f1_score']:.3f} "
                      f"TP={result['tp']} FP={result['fp']} samples={result['total_samples']}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Summary results
    summary_results = []
    for result in all_results:
        summary_results.append({
            'config_name': result['config_name'],
            'pred_threshold': result['pred_threshold'],
            'evidence_ratio': result['evidence_ratio'],
            'neg_pos_ratio': result['neg_pos_ratio'],
            'negative_candidates_count': result['negative_candidates_count'],
            'tp': result['tp'],
            'fp': result['fp'],
            'tn': result['tn'],
            'fn': result['fn'],
            'precision': result['precision'],
            'recall': result['recall'],
            'f1_score': result['f1_score'],
            'accuracy': result['accuracy'],
            'positive_samples': result['positive_samples'],
            'negative_samples': result['negative_samples'],
            'total_samples': result['total_samples']
        })
    
    summary_df = pd.DataFrame(summary_results)
    summary_file = f"focused_conservative_summary_{timestamp}.csv"
    summary_df.to_csv(summary_file, index=False)
    
    # Detailed results with sample-by-sample data
    detailed_results = []
    for result in all_results:
        for sample in result['sample_details']:
            detailed_results.append({
                'config_name': result['config_name'],
                'pred_threshold': result['pred_threshold'],
                'evidence_ratio': result['evidence_ratio'],
                'neg_pos_ratio': result['neg_pos_ratio'],
                'sample_type': sample['type'],
                'target_road': sample['target_road'],
                'prob_flood': sample['prob_flood'],
                'prediction': sample['prediction'],
                'true_label': sample['true_label'],
                'date': sample['date'],
                'is_correct': (sample['prediction'] == sample['true_label'])
            })
    
    detailed_df = pd.DataFrame(detailed_results)
    detailed_file = f"focused_conservative_detailed_{timestamp}.csv"
    detailed_df.to_csv(detailed_file, index=False)
    
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    
    if len(summary_df) > 0:
        # High precision results
        high_precision = summary_df[summary_df['precision'] >= 0.7].copy()
        print(f"Total configurations tested: {len(summary_df)}")
        print(f"High precision results (≥0.7): {len(high_precision)}")
        
        if len(high_precision) > 0:
            print(f"\nTop 10 High-Precision Results:")
            top_results = high_precision.nlargest(10, ['precision', 'recall', 'f1_score'])
            
            for _, row in top_results.iterrows():
                print(f"\nConfig: {row['config_name']}")
                print(f"  Parameters: pred_threshold={row['pred_threshold']}, evidence_ratio={row['evidence_ratio']}, neg_pos_ratio={row['neg_pos_ratio']}")
                print(f"  Metrics: Precision={row['precision']:.3f}, Recall={row['recall']:.3f}, F1={row['f1_score']:.3f}")
                print(f"  Confusion Matrix: TP={row['tp']}, FP={row['fp']}, TN={row['tn']}, FN={row['fn']}")
                print(f"  Samples: {row['positive_samples']} positive, {row['negative_samples']} negative ({row['negative_candidates_count']} neg candidates)")
        
        # Best results by criteria
        print(f"\nBest Results by Different Criteria:")
        best_precision = summary_df.loc[summary_df['precision'].idxmax()]
        best_recall = summary_df.loc[summary_df['recall'].idxmax()]
        best_f1 = summary_df.loc[summary_df['f1_score'].idxmax()]
        
        print(f"Best Precision: {best_precision['precision']:.3f} ({best_precision['config_name']})")
        print(f"Best Recall: {best_recall['recall']:.3f} ({best_recall['config_name']})")
        print(f"Best F1-Score: {best_f1['f1_score']:.3f} ({best_f1['config_name']})")
    
    print(f"\nResults saved to:")
    print(f"  Summary: {summary_file}")
    print(f"  Detailed: {detailed_file}")
    
    return all_results, summary_file, detailed_file

if __name__ == "__main__":
    results, summary_file, detailed_file = run_focused_evaluation()