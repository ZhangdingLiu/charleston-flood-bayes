#!/usr/bin/env python3
"""
No Negative Filtering Evaluation
Evaluate Bayesian network by predicting ALL roads without pre-filtering negative candidates
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
    """Load and preprocess the flood data"""
    print("=" * 80)
    print("NO NEGATIVE FILTERING EVALUATION")
    print("Predict ALL roads in network without pre-filtering negative candidates")
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

def build_bayesian_network(train_df):
    """Build the Bayesian network"""
    print(f"\n{'='*60}")
    print("BUILDING BAYESIAN NETWORK")
    print(f"{'='*60}")
    
    # Build network with same parameters as before
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
    print(f"Network parameters: occ_thr=3, edge_thr=2, weight_thr=0.3, max_parents=2")
    print(f"Marginal probability range: {min(marginals_dict.values()):.4f} - {max(marginals_dict.values()):.4f}")
    
    return flood_net, bn_nodes, marginals_dict

def analyze_test_set_coverage(test_df, bn_nodes):
    """Analyze which roads from test set are in the Bayesian network"""
    print(f"\n{'='*60}")
    print("TEST SET COVERAGE ANALYSIS")
    print(f"{'='*60}")
    
    # Get all roads that appear in test set
    test_roads = set(test_df["link_id"].unique())
    test_roads_in_bn = test_roads.intersection(bn_nodes)
    test_roads_not_in_bn = test_roads - bn_nodes
    
    print(f"Total unique roads in test set: {len(test_roads)}")
    print(f"Test roads in Bayesian network: {len(test_roads_in_bn)}")
    print(f"Test roads NOT in network: {len(test_roads_not_in_bn)}")
    print(f"Coverage: {len(test_roads_in_bn)/len(test_roads)*100:.1f}%")
    
    if len(test_roads_not_in_bn) > 0:
        print(f"\nRoads in test set but not in network:")
        for road in sorted(test_roads_not_in_bn):
            count = len(test_df[test_df["link_id"] == road])
            print(f"  {road}: {count} occurrences")
    
    # Analyze test set by day
    test_by_date = test_df.groupby(test_df["time_create"].dt.floor("D"))
    evaluable_days = 0
    total_test_days = len(test_by_date)
    
    for date, day_group in test_by_date:
        flooded_roads = list(day_group["link_id"].unique())
        flooded_in_bn = [road for road in flooded_roads if road in bn_nodes]
        
        if len(flooded_in_bn) >= 2:  # Need at least 2 roads for evidence + target
            evaluable_days += 1
    
    print(f"\nTest set daily analysis:")
    print(f"  Total test days: {total_test_days}")
    print(f"  Evaluable days (≥2 BN roads): {evaluable_days}")
    print(f"  Evaluation coverage: {evaluable_days/total_test_days*100:.1f}%")
    
    return test_roads_in_bn, test_roads_not_in_bn

def evaluate_configuration(flood_net, test_df, bn_nodes, test_roads_in_bn, 
                         pred_threshold, evidence_ratio, min_evidence=1):
    """
    Evaluate a specific configuration without negative filtering
    
    Args:
        flood_net: Trained Bayesian network
        test_df: Test dataset
        bn_nodes: Set of nodes in Bayesian network  
        test_roads_in_bn: Roads from test set that are in the network
        pred_threshold: Probability threshold for flood prediction
        evidence_ratio: Ratio of flooded roads to use as evidence
        min_evidence: Minimum number of evidence roads required
    
    Returns:
        Dictionary with evaluation results
    """
    
    results = []
    detailed_predictions = []
    test_by_date = test_df.groupby(test_df["time_create"].dt.floor("D"))
    evaluated_days = 0
    skipped_days = 0
    
    for date, day_group in test_by_date:
        flooded_roads = list(day_group["link_id"].unique())
        flooded_in_bn = [road for road in flooded_roads if road in bn_nodes]
        
        # Need at least min_evidence+1 roads (evidence + at least 1 target)
        if len(flooded_in_bn) < min_evidence + 1:
            skipped_days += 1
            continue
            
        evaluated_days += 1
        
        # Evidence selection
        evidence_count = max(min_evidence, int(len(flooded_in_bn) * evidence_ratio))
        evidence_count = min(evidence_count, len(flooded_in_bn) - 1)  # Leave at least 1 for target
        evidence_roads = flooded_in_bn[:evidence_count]
        evidence = {road: 1 for road in evidence_roads}
        
        # Target roads: ALL roads in network that appear in test set, excluding evidence
        target_roads = [road for road in test_roads_in_bn if road not in evidence_roads]
        
        # Store day details
        day_detail = {
            'date': date.strftime('%Y-%m-%d'),
            'all_flooded_roads': flooded_roads,
            'flooded_in_bn': flooded_in_bn,
            'evidence_roads': evidence_roads,
            'target_roads': target_roads,
            'predictions': []
        }
        
        # Predict for ALL target roads
        for target_road in target_roads:
            try:
                result = flood_net.infer_w_evidence(target_road, evidence)
                prob_flood = result['flooded']
                prediction = 1 if prob_flood >= pred_threshold else 0
                
                # Ground truth: did this road actually flood on this day?
                true_label = 1 if target_road in flooded_roads else 0
                is_correct = (prediction == true_label)
                
                pred_detail = {
                    'target_road': target_road,
                    'prob_flood': prob_flood,
                    'prediction': prediction,
                    'true_label': true_label,
                    'is_correct': is_correct,
                    'sample_type': 'Positive' if true_label == 1 else 'Negative'
                }
                
                results.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'target_road': target_road,
                    'prob_flood': prob_flood,
                    'prediction': prediction,
                    'true_label': true_label,
                    'is_correct': is_correct,
                    'sample_type': 'Positive' if true_label == 1 else 'Negative'
                })
                
                day_detail['predictions'].append(pred_detail)
                
            except Exception as e:
                # Skip roads that can't be predicted (e.g., not connected in network)
                continue
        
        detailed_predictions.append(day_detail)
    
    # Calculate confusion matrix
    tp = sum(1 for r in results if r['true_label'] == 1 and r['prediction'] == 1)
    fp = sum(1 for r in results if r['true_label'] == 0 and r['prediction'] == 1)
    fn = sum(1 for r in results if r['true_label'] == 1 and r['prediction'] == 0)
    tn = sum(1 for r in results if r['true_label'] == 0 and r['prediction'] == 0)
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
    
    positive_samples = sum(1 for r in results if r['true_label'] == 1)
    negative_samples = sum(1 for r in results if r['true_label'] == 0)
    total_samples = len(results)
    
    # Prediction distribution
    positive_predictions = sum(1 for r in results if r['prediction'] == 1)
    negative_predictions = sum(1 for r in results if r['prediction'] == 0)
    
    return {
        'pred_threshold': pred_threshold,
        'evidence_ratio': evidence_ratio,
        'min_evidence': min_evidence,
        'evaluated_days': evaluated_days,
        'skipped_days': skipped_days,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy,
        'positive_samples': positive_samples,
        'negative_samples': negative_samples,
        'total_samples': total_samples,
        'positive_predictions': positive_predictions,
        'negative_predictions': negative_predictions,
        'detailed_predictions': detailed_predictions,
        'all_results': results
    }

def run_comprehensive_evaluation():
    """Run comprehensive evaluation with parameter grid"""
    
    # Load data and build network
    train_df, test_df = load_and_preprocess_data()
    flood_net, bn_nodes, marginals_dict = build_bayesian_network(train_df)
    test_roads_in_bn, test_roads_not_in_bn = analyze_test_set_coverage(test_df, bn_nodes)
    
    print(f"\n{'='*60}")
    print("PARAMETER GRID EVALUATION")
    print(f"{'='*60}")
    
    # Define parameter grid
    pred_thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6]
    evidence_ratios = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    min_evidences = [1, 2]  # Minimum number of evidence roads
    
    total_combinations = len(pred_thresholds) * len(evidence_ratios) * len(min_evidences)
    print(f"Testing {total_combinations} parameter combinations")
    print(f"Prediction thresholds: {pred_thresholds}")
    print(f"Evidence ratios: {evidence_ratios}")
    print(f"Min evidence counts: {min_evidences}")
    
    # Run all combinations
    all_results = []
    combination_count = 0
    
    for pred_thresh, ev_ratio, min_ev in itertools.product(pred_thresholds, evidence_ratios, min_evidences):
        combination_count += 1
        
        print(f"\n[{combination_count:3d}/{total_combinations}] Testing pred_thresh={pred_thresh}, ev_ratio={ev_ratio}, min_ev={min_ev}")
        
        result = evaluate_configuration(
            flood_net, test_df, bn_nodes, test_roads_in_bn,
            pred_thresh, ev_ratio, min_ev
        )
        
        all_results.append(result)
        
        # Print key metrics
        print(f"    Results: P={result['precision']:.3f}, R={result['recall']:.3f}, F1={result['f1_score']:.3f}, "
              f"Acc={result['accuracy']:.3f} | TP={result['tp']}, FP={result['fp']}, TN={result['tn']}, FN={result['fn']} | "
              f"Samples: {result['total_samples']} ({result['positive_samples']} pos, {result['negative_samples']} neg)")
    
    print(f"\nCompleted {len(all_results)} evaluations")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Summary results
    summary_results = []
    for result in all_results:
        summary_results.append({
            'pred_threshold': result['pred_threshold'],
            'evidence_ratio': result['evidence_ratio'],
            'min_evidence': result['min_evidence'],
            'evaluated_days': result['evaluated_days'],
            'skipped_days': result['skipped_days'],
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
            'total_samples': result['total_samples'],
            'positive_predictions': result['positive_predictions'],
            'negative_predictions': result['negative_predictions']
        })
    
    summary_df = pd.DataFrame(summary_results)
    summary_file = f"no_filtering_evaluation_summary_{timestamp}.csv"
    summary_df.to_csv(summary_file, index=False)
    
    # Detailed sample-by-sample results (for best configurations only to save space)
    print(f"\n{'='*60}")
    print("RESULTS ANALYSIS")
    print(f"{'='*60}")
    
    # Filter valid results (with reasonable sample sizes)
    valid_results = summary_df[summary_df['total_samples'] >= 50].copy()
    print(f"Valid configurations (≥50 samples): {len(valid_results)}")
    
    if len(valid_results) > 0:
        # Best results by different criteria
        best_precision_idx = valid_results['precision'].idxmax()
        best_recall_idx = valid_results['recall'].idxmax() 
        best_f1_idx = valid_results['f1_score'].idxmax()
        best_accuracy_idx = valid_results['accuracy'].idxmax()
        
        print(f"\nBest Results:")
        print(f"Best Precision: {valid_results.loc[best_precision_idx, 'precision']:.3f} "
              f"(thresh={valid_results.loc[best_precision_idx, 'pred_threshold']}, "
              f"ev_ratio={valid_results.loc[best_precision_idx, 'evidence_ratio']}, "
              f"min_ev={valid_results.loc[best_precision_idx, 'min_evidence']})")
        
        print(f"Best Recall: {valid_results.loc[best_recall_idx, 'recall']:.3f} "
              f"(thresh={valid_results.loc[best_recall_idx, 'pred_threshold']}, "
              f"ev_ratio={valid_results.loc[best_recall_idx, 'evidence_ratio']}, "
              f"min_ev={valid_results.loc[best_recall_idx, 'min_evidence']})")
        
        print(f"Best F1-Score: {valid_results.loc[best_f1_idx, 'f1_score']:.3f} "
              f"(thresh={valid_results.loc[best_f1_idx, 'pred_threshold']}, "
              f"ev_ratio={valid_results.loc[best_f1_idx, 'evidence_ratio']}, "
              f"min_ev={valid_results.loc[best_f1_idx, 'min_evidence']})")
        
        print(f"Best Accuracy: {valid_results.loc[best_accuracy_idx, 'accuracy']:.3f} "
              f"(thresh={valid_results.loc[best_accuracy_idx, 'pred_threshold']}, "
              f"ev_ratio={valid_results.loc[best_accuracy_idx, 'evidence_ratio']}, "
              f"min_ev={valid_results.loc[best_accuracy_idx, 'min_evidence']})")
        
        # High precision results (≥0.7)
        high_precision = valid_results[valid_results['precision'] >= 0.7].copy()
        print(f"\nHigh Precision Results (≥0.7): {len(high_precision)}")
        
        if len(high_precision) > 0:
            high_precision_sorted = high_precision.nlargest(10, ['precision', 'recall', 'f1_score'])
            print(f"\nTop High-Precision Configurations:")
            for _, row in high_precision_sorted.iterrows():
                print(f"  thresh={row['pred_threshold']:.2f}, ev_ratio={row['evidence_ratio']:.1f}, min_ev={row['min_evidence']} | "
                      f"P={row['precision']:.3f}, R={row['recall']:.3f}, F1={row['f1_score']:.3f} | "
                      f"TP={row['tp']}, FP={row['fp']}, samples={row['total_samples']}")
        
        # Save detailed results for top configurations
        top_configs = []
        top_configs.append(all_results[best_precision_idx])
        top_configs.append(all_results[best_recall_idx])
        top_configs.append(all_results[best_f1_idx])
        
        # Remove duplicates
        top_configs_unique = []
        seen_configs = set()
        for config in top_configs:
            config_key = (config['pred_threshold'], config['evidence_ratio'], config['min_evidence'])
            if config_key not in seen_configs:
                top_configs_unique.append(config)
                seen_configs.add(config_key)
        
        # Save detailed predictions for top configurations
        detailed_results = []
        for config in top_configs_unique:
            config_name = f"thresh_{config['pred_threshold']}_ev_{config['evidence_ratio']}_min_{config['min_evidence']}"
            for sample in config['all_results']:
                detailed_results.append({
                    'config_name': config_name,
                    'pred_threshold': config['pred_threshold'],
                    'evidence_ratio': config['evidence_ratio'],
                    'min_evidence': config['min_evidence'],
                    'date': sample['date'],
                    'target_road': sample['target_road'],
                    'prob_flood': sample['prob_flood'],
                    'prediction': sample['prediction'],
                    'true_label': sample['true_label'],
                    'is_correct': sample['is_correct'],
                    'sample_type': sample['sample_type']
                })
        
        if detailed_results:
            detailed_df = pd.DataFrame(detailed_results)
            detailed_file = f"no_filtering_evaluation_detailed_{timestamp}.csv"
            detailed_df.to_csv(detailed_file, index=False)
            print(f"\nDetailed results for top configurations saved to: {detailed_file}")
    
    print(f"\nSummary results saved to: {summary_file}")
    print(f"Total configurations tested: {len(all_results)}")
    
    return all_results, summary_file

def main():
    """Main evaluation function"""
    try:
        results, summary_file = run_comprehensive_evaluation()
        print(f"\n{'='*80}")
        print("EVALUATION COMPLETE")
        print(f"{'='*80}")
        print(f"Results saved to: {summary_file}")
        return results
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()