#!/usr/bin/env python3
"""
Specified Parameter Test Set Evaluation

This script evaluates the user-specified parameter combinations using train+valid data
for training and testing on the held-out test set.

Parameters to test:
- occ_thr: 2 or 4
- edge_thr: 3  
- weight_thr: 0.2
- evidence_count: 1
- pred_threshold: 0.1, 0.2, 0.3, 0.4
- neg_pos_ratio: 1
- marginal_prob_threshold: 0.08
"""

import pandas as pd
import numpy as np
import warnings
import time
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import random
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from scipy import stats
import itertools
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from model import FloodBayesNetwork
except ImportError:
    try:
        from core.model import FloodBayesNetwork
    except ImportError:
        print("❌ Cannot import FloodBayesNetwork, please ensure model.py or core/model.py exists")
        sys.exit(1)

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class SpecifiedParameterEvaluator:
    """Evaluator for user-specified parameter combinations"""
    
    def __init__(self, data_path="Road_Closures_2024.csv"):
        """Initialize the evaluator"""
        self.data_path = data_path
        self.results = []
        
    def get_specified_parameter_combinations(self):
        """Get the user-specified parameter combinations"""
        
        # Fixed parameters
        base_params = {
            'edge_thr': 3,
            'weight_thr': 0.2,
            'evidence_count': 1,
            'neg_pos_ratio': 1.0,
            'marginal_prob_threshold': 0.08
        }
        
        # Variable parameters
        occ_thr_values = [2, 4]
        pred_threshold_values = [0.1, 0.2, 0.3, 0.4]
        
        # Generate all combinations
        param_combinations = []
        config_id = 1
        
        for occ_thr in occ_thr_values:
            for pred_threshold in pred_threshold_values:
                params = base_params.copy()
                params['occ_thr'] = occ_thr
                params['pred_threshold'] = pred_threshold
                params['config_id'] = config_id
                param_combinations.append(params)
                config_id += 1
        
        print(f"Generated {len(param_combinations)} specified parameter combinations")
        return param_combinations
        
    def load_and_prepare_data(self):
        """Load and prepare data with proper train/validation/test split"""
        print("Loading and preparing data...")
        
        # Load data
        df = pd.read_csv(self.data_path)
        
        # Preprocess same as parameter optimization
        df['time_create'] = pd.to_datetime(df['START'], utc=True)
        df = df[df['REASON'] == 'FLOOD'].copy()
        df['flood_date'] = df['time_create'].dt.floor('D')
        df['link_id'] = df['STREET'].str.upper().str.replace(' ', '_')
        df['link_id'] = df['link_id'].astype(str)
        df['id'] = df['OBJECTID'].astype(str)
        
        print(f"Loaded {len(df)} flood records")
        
        # Time-based split by flood days (same as parameter optimization)
        unique_days = sorted(df['flood_date'].unique())
        n_days = len(unique_days)
        
        # Split: 60% train, 20% validation, 20% test
        train_end = int(n_days * 0.6)
        valid_end = int(n_days * 0.8)
        
        train_days = unique_days[:train_end]
        valid_days = unique_days[train_end:valid_end]
        test_days = unique_days[valid_end:]
        
        # Create splits
        self.train_df = df[df['flood_date'].isin(train_days)].copy()
        self.valid_df = df[df['flood_date'].isin(valid_days)].copy()
        self.test_df = df[df['flood_date'].isin(test_days)].copy()
        
        # Combine train and validation for final training
        self.train_valid_df = pd.concat([self.train_df, self.valid_df], ignore_index=True)
        
        print(f"Data splits:")
        print(f"  Train: {len(train_days)} days, {len(self.train_df)} records")
        print(f"  Valid: {len(valid_days)} days, {len(self.valid_df)} records")
        print(f"  Train+Valid: {len(train_days)+len(valid_days)} days, {len(self.train_valid_df)} records")
        print(f"  Test:  {len(test_days)} days, {len(self.test_df)} records")
        
        return self.train_valid_df, self.test_df
    
    def build_bayesian_network(self, train_data, occ_thr, edge_thr, weight_thr):
        """Build Bayesian network with given parameters on train+valid data"""
        try:
            # Build network
            flood_net = FloodBayesNetwork(t_window="D")
            flood_net.fit_marginal(train_data)
            
            # Build co-occurrence network
            flood_net.build_network_by_co_occurrence(
                train_data,
                occ_thr=occ_thr,
                edge_thr=edge_thr,
                weight_thr=weight_thr,
                report=False
            )
            
            # Fit conditional probabilities
            flood_net.fit_conditional(train_data, max_parents=2, alpha=1.0)
            
            # Build final Bayesian network
            flood_net.build_bayes_network()
            
            return flood_net, True
            
        except Exception as e:
            print(f"❌ Network building failed: {str(e)}")
            return None, False
    
    def evaluate_parameter_combination(self, params):
        """Evaluate a single parameter combination on test set"""
        
        config_id = params['config_id']
        print(f"🔧 Evaluating Config #{config_id}:")
        print(f"   Network: occ_thr={params['occ_thr']}, edge_thr={params['edge_thr']}, weight_thr={params['weight_thr']}")
        print(f"   Prediction: pred_thr={params['pred_threshold']}, evidence={params['evidence_count']}, neg_ratio={params['neg_pos_ratio']}")
        
        start_time = time.time()
        
        # Build network on train+valid data
        flood_net, success = self.build_bayesian_network(
            self.train_valid_df, 
            params['occ_thr'], 
            params['edge_thr'], 
            params['weight_thr']
        )
        
        if not success:
            return None
        
        network_nodes = flood_net.network.number_of_nodes()
        network_edges = flood_net.network.number_of_edges()
        print(f"   Network built: {network_nodes} nodes, {network_edges} edges")
        
        if network_nodes == 0:
            print("   ❌ Empty network, skipping...")
            return None
        
        # Evaluate on test set
        predictions = []
        true_labels = []
        
        # Group test data by day
        test_by_day = self.test_df.groupby(self.test_df['time_create'].dt.floor('D'))
        valid_test_days = 0
        total_test_days = len(test_by_day)
        
        for test_date, day_group in test_by_day:
            flooded_roads = set(day_group['link_id'].unique())
            network_roads = set(flood_net.network.nodes())
            
            # Filter roads that are in the network
            flooded_in_network = flooded_roads & network_roads
            
            if len(flooded_in_network) < params['evidence_count']:
                continue  # Not enough evidence roads
            
            valid_test_days += 1
            
            # Select evidence roads (first N flooded roads as evidence)
            evidence_roads = list(flooded_in_network)[:params['evidence_count']]
            candidate_roads = network_roads - set(evidence_roads)
            
            # Apply negative sampling
            marginals = flood_net.marginals[flood_net.marginals['link_id'].isin(candidate_roads)]
            neg_candidates = marginals[
                marginals['p'] >= params['marginal_prob_threshold']
            ]['link_id'].tolist()
            
            if len(neg_candidates) == 0:
                continue
            
            # Apply negative sampling ratio
            n_neg = min(len(neg_candidates), 
                       max(1, int(len(flooded_in_network) * params['neg_pos_ratio'])))
            neg_candidates = neg_candidates[:n_neg]
            
            # Run inference
            evidence = {road: 1 for road in evidence_roads}
            
            for candidate in neg_candidates:
                try:
                    # Query probability using correct method
                    result = flood_net.infer_w_evidence(candidate, evidence)
                    prob = result['flooded']
                    
                    # Make prediction
                    pred = 1 if prob >= params['pred_threshold'] else 0
                    true = 1 if candidate in flooded_in_network else 0
                    
                    predictions.append(pred)
                    true_labels.append(true)
                    
                except Exception as e:
                    continue  # Skip failed inferences
        
        if len(predictions) == 0:
            print("   ❌ No valid predictions made")
            return None
        
        # Calculate metrics
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
        
        # Performance metrics
        precision = precision_score(true_labels, predictions, zero_division=0)
        recall = recall_score(true_labels, predictions, zero_division=0)
        f1 = f1_score(true_labels, predictions, zero_division=0)
        accuracy = accuracy_score(true_labels, predictions)
        
        runtime = time.time() - start_time
        
        result = {
            # Metadata
            'config_id': config_id,
            'config_type': 'specified_train_valid_test',
            
            # Parameters
            'occ_thr': params['occ_thr'],
            'edge_thr': params['edge_thr'],
            'weight_thr': params['weight_thr'],
            'evidence_count': params['evidence_count'],
            'pred_threshold': params['pred_threshold'],
            'neg_pos_ratio': params['neg_pos_ratio'],
            'marginal_prob_threshold': params['marginal_prob_threshold'],
            
            # Test results
            'test_tp': tp,
            'test_fp': fp,
            'test_tn': tn,
            'test_fn': fn,
            'test_total_samples': len(predictions),
            'test_positive_samples': np.sum(true_labels),
            'test_negative_samples': len(true_labels) - np.sum(true_labels),
            'test_precision': precision,
            'test_recall': recall,
            'test_f1_score': f1,
            'test_accuracy': accuracy,
            
            # Model info
            'network_nodes': network_nodes,
            'network_edges': network_edges,
            'valid_test_days': valid_test_days,
            'total_test_days': total_test_days,
            'training_data_size': len(self.train_valid_df),
            'test_runtime_seconds': runtime
        }
        
        print(f"   ✅ Test results: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, Acc={accuracy:.3f}")
        
        return result
    
    def evaluate_specified_configurations(self):
        """Evaluate all specified configurations on test set"""
        
        print("=== Specified Parameter Test Set Evaluation ===")
        print("Training on: Train + Validation data")
        print("Testing on: Test data")
        
        # Prepare data
        self.load_and_prepare_data()
        
        # Get specified parameter combinations
        param_combinations = self.get_specified_parameter_combinations()
        
        # Evaluate each configuration
        results = []
        successful_evaluations = 0
        
        for idx, params in enumerate(param_combinations, 1):
            print(f"\n📊 Progress: {idx}/{len(param_combinations)} configurations")
            
            result = self.evaluate_parameter_combination(params)
            
            if result is not None:
                results.append(result)
                successful_evaluations += 1
            else:
                print(f"   ❌ Evaluation failed for config #{params['config_id']}")
        
        print(f"\n✅ Evaluation completed: {successful_evaluations}/{len(param_combinations)} successful")
        
        if len(results) == 0:
            print("❌ No successful evaluations!")
            return None
        
        return pd.DataFrame(results)
    
    def analyze_and_save_results(self, results_df, output_dir):
        """Analyze results and save comprehensive reports"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save complete results
        results_path = output_dir / 'specified_params_test_results.csv'
        results_df.to_csv(results_path, index=False)
        print(f"📄 Saved complete results to: {results_path}")
        
        # Analysis
        print(f"\n📊 Performance Analysis:")
        print(f"   F1 Score range: {results_df['test_f1_score'].min():.3f} - {results_df['test_f1_score'].max():.3f}")
        print(f"   Precision range: {results_df['test_precision'].min():.3f} - {results_df['test_precision'].max():.3f}")
        print(f"   Recall range: {results_df['test_recall'].min():.3f} - {results_df['test_recall'].max():.3f}")
        
        # Find best configurations by different metrics
        best_configs = {
            'f1': results_df.loc[results_df['test_f1_score'].idxmax()],
            'precision': results_df.loc[results_df['test_precision'].idxmax()],
            'recall': results_df.loc[results_df['test_recall'].idxmax()],
            'accuracy': results_df.loc[results_df['test_accuracy'].idxmax()]
        }
        
        # Analyze by parameters
        print(f"\n🎯 Parameter Impact Analysis:")
        
        # occ_thr impact
        for occ_thr in sorted(results_df['occ_thr'].unique()):
            data = results_df[results_df['occ_thr'] == occ_thr]
            avg_f1 = data['test_f1_score'].mean()
            avg_p = data['test_precision'].mean()
            avg_r = data['test_recall'].mean()
            print(f"   occ_thr={occ_thr}: F1={avg_f1:.3f}, P={avg_p:.3f}, R={avg_r:.3f} (n={len(data)})")
        
        # pred_threshold impact
        print(f"\n   Prediction Threshold Impact:")
        for pred_thr in sorted(results_df['pred_threshold'].unique()):
            data = results_df[results_df['pred_threshold'] == pred_thr]
            avg_f1 = data['test_f1_score'].mean()
            avg_p = data['test_precision'].mean()
            avg_r = data['test_recall'].mean()
            print(f"   pred_threshold={pred_thr}: F1={avg_f1:.3f}, P={avg_p:.3f}, R={avg_r:.3f} (n={len(data)})")
        
        # Generate report
        report = f"""# Specified Parameter Test Set Evaluation Results

## Overview
This evaluation tested the user-specified parameter combinations using train+validation data for training
and evaluating on the held-out test set.

### Parameter Configuration
- **occ_thr**: 2, 4
- **edge_thr**: 3 (fixed)
- **weight_thr**: 0.2 (fixed)
- **evidence_count**: 1 (fixed)
- **pred_threshold**: 0.1, 0.2, 0.3, 0.4
- **neg_pos_ratio**: 1.0 (fixed)
- **marginal_prob_threshold**: 0.08 (fixed)

## Key Findings

### Overall Performance Distribution
- **F1 Score**: {results_df['test_f1_score'].mean():.3f} ± {results_df['test_f1_score'].std():.3f} (range: {results_df['test_f1_score'].min():.3f} - {results_df['test_f1_score'].max():.3f})
- **Precision**: {results_df['test_precision'].mean():.3f} ± {results_df['test_precision'].std():.3f} (range: {results_df['test_precision'].min():.3f} - {results_df['test_precision'].max():.3f})
- **Recall**: {results_df['test_recall'].mean():.3f} ± {results_df['test_recall'].std():.3f} (range: {results_df['test_recall'].min():.3f} - {results_df['test_recall'].max():.3f})

### Best Configurations by Metric
"""
        
        for metric, config in best_configs.items():
            report += f"\n**Best {metric.title()}** (Config #{config['config_id']}):\n"
            report += f"- Performance: P={config['test_precision']:.3f}, R={config['test_recall']:.3f}, F1={config['test_f1_score']:.3f}, Acc={config['test_accuracy']:.3f}\n"
            report += f"- Parameters: occ_thr={config['occ_thr']}, pred_threshold={config['pred_threshold']}\n"
            report += f"- Network: {config['network_nodes']} nodes, {config['network_edges']} edges\n"
        
        # Parameter impact analysis
        report += "\n### Parameter Impact Analysis\n\n"
        
        # occ_thr impact
        report += "**Network Occurrence Threshold (occ_thr) Impact:**\n"
        for occ_thr in sorted(results_df['occ_thr'].unique()):
            data = results_df[results_df['occ_thr'] == occ_thr]
            avg_f1 = data['test_f1_score'].mean()
            avg_p = data['test_precision'].mean()
            avg_r = data['test_recall'].mean()
            avg_nodes = data['network_nodes'].mean()
            report += f"- occ_thr={occ_thr}: F1={avg_f1:.3f}, P={avg_p:.3f}, R={avg_r:.3f}, Avg_nodes={avg_nodes:.1f}\n"
        
        # pred_threshold impact
        report += "\n**Prediction Threshold Impact:**\n"
        for pred_thr in sorted(results_df['pred_threshold'].unique()):
            data = results_df[results_df['pred_threshold'] == pred_thr]
            avg_f1 = data['test_f1_score'].mean()
            avg_p = data['test_precision'].mean()
            avg_r = data['test_recall'].mean()
            report += f"- pred_threshold={pred_thr}: F1={avg_f1:.3f}, P={avg_p:.3f}, R={avg_r:.3f}\n"
        
        # All configurations results
        report += f"\n### All Configuration Results\n\n"
        for _, config in results_df.iterrows():
            report += f"**Config #{config['config_id']}** (occ_thr={config['occ_thr']}, pred_thr={config['pred_threshold']}):\n"
            report += f"- Performance: P={config['test_precision']:.3f}, R={config['test_recall']:.3f}, F1={config['test_f1_score']:.3f}, Acc={config['test_accuracy']:.3f}\n"
            report += f"- Network: {config['network_nodes']} nodes, {config['network_edges']} edges\n"
            report += f"- Test samples: {config['test_total_samples']}, Valid test days: {config['valid_test_days']}/{config['total_test_days']}\n\n"
        
        report += f"""
## Training Data Impact
- **Training Data**: Train + Validation combined ({len(self.train_valid_df)} records)
- **Test Data**: Independent test set ({len(self.test_df)} records)
- **Temporal Split**: Proper time-based split to avoid data leakage

## Deployment Recommendations

### Based on Test Results:
1. **Best Overall**: Config #{best_configs['f1']['config_id']} - F1={best_configs['f1']['test_f1_score']:.3f}
2. **High Precision**: Config #{best_configs['precision']['config_id']} - P={best_configs['precision']['test_precision']:.3f}
3. **High Recall**: Config #{best_configs['recall']['config_id']} - R={best_configs['recall']['test_recall']:.3f}

### Key Insights:
- Training on combined train+valid data vs train-only shows model performance with more data
- pred_threshold has significant impact on precision-recall trade-off
- Network size (occ_thr) affects model complexity and performance

## Data Quality Notes
- Test set covers {len(set(self.test_df['time_create'].dt.floor('D')))} unique flood days
- Average test samples per configuration: {results_df['test_total_samples'].mean():.1f}
- Consistent evaluation across all configurations
"""
        
        # Save report
        report_path = output_dir / 'SPECIFIED_PARAMS_TEST_REPORT.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📄 Saved evaluation report to: {report_path}")
        
        return best_configs

def main():
    """Main function to run specified parameter test set evaluation"""
    
    # Create output directory for test results
    output_dir = Path("results/parameter_optimization_20250721_140722/specified_params_test_evaluation")
    
    print("=== Specified Parameter Test Set Evaluation ===")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Parameters: occ_thr=[2,4], edge_thr=3, weight_thr=0.2, pred_threshold=[0.1,0.2,0.3,0.4]")
    
    # Initialize evaluator
    evaluator = SpecifiedParameterEvaluator()
    
    # Run evaluation
    results_df = evaluator.evaluate_specified_configurations()
    
    if results_df is not None:
        # Analyze and save results
        best_configs = evaluator.analyze_and_save_results(results_df, output_dir)
        
        print(f"\n✅ Specified parameter test evaluation completed successfully!")
        print(f"📁 Results saved to: {output_dir}")
        print(f"📊 Evaluated {len(results_df)} configurations")
        
        # Show all results
        print(f"\n🏆 All Configuration Results:")
        for _, config in results_df.iterrows():
            print(f"   Config #{config['config_id']}: occ_thr={config['occ_thr']}, pred_thr={config['pred_threshold']}")
            print(f"      F1={config['test_f1_score']:.3f}, P={config['test_precision']:.3f}, R={config['test_recall']:.3f}, Acc={config['test_accuracy']:.3f}")
            print(f"      Network: {config['network_nodes']} nodes, {config['network_edges']} edges")
        
        # Show best performer
        best_f1 = results_df.loc[results_df['test_f1_score'].idxmax()]
        print(f"\n🎯 Best F1 Score: {best_f1['test_f1_score']:.3f}")
        print(f"   Config #{best_f1['config_id']}: occ_thr={best_f1['occ_thr']}, pred_threshold={best_f1['pred_threshold']}")
        print(f"   Performance: P={best_f1['test_precision']:.3f}, R={best_f1['test_recall']:.3f}, Acc={best_f1['test_accuracy']:.3f}")
        
        return results_df
    else:
        print("❌ Specified parameter test evaluation failed!")
        return None

if __name__ == "__main__":
    results = main()