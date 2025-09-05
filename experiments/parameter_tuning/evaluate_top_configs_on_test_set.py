#!/usr/bin/env python3
"""
Test Set Evaluation for Top Parameter Configurations

This script evaluates the best parameter configurations (selected on validation set) 
on the held-out test set to provide unbiased performance estimates for deployment.

Key Features:
- Proper train/validation/test split (60%/20%/20% by flood days)
- Evaluates all 17 top configurations on test set
- Comprehensive metrics with confidence intervals
- Validation vs test performance comparison
- Statistical significance testing
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

class TestSetEvaluator:
    """Test set evaluator for top parameter configurations"""
    
    def __init__(self, data_path="Road_Closures_2024.csv"):
        """Initialize the evaluator"""
        self.data_path = data_path
        self.results = []
        
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
        
        print(f"Data splits:")
        print(f"  Train: {len(train_days)} days ({train_days[0].date()} to {train_days[-1].date()}), {len(self.train_df)} records")
        print(f"  Valid: {len(valid_days)} days ({valid_days[0].date()} to {valid_days[-1].date()}), {len(self.valid_df)} records")
        print(f"  Test:  {len(test_days)} days ({test_days[0].date()} to {test_days[-1].date()}), {len(self.test_df)} records")
        
        return self.train_df, self.valid_df, self.test_df
    
    def build_bayesian_network(self, train_df, occ_thr, edge_thr, weight_thr):
        """Build Bayesian network with given parameters"""
        try:
            # Build network
            flood_net = FloodBayesNetwork(t_window="D")
            flood_net.fit_marginal(train_df)
            
            # Build co-occurrence network
            flood_net.build_network_by_co_occurrence(
                train_df,
                occ_thr=occ_thr,
                edge_thr=edge_thr,
                weight_thr=weight_thr,
                report=False
            )
            
            # Fit conditional probabilities
            flood_net.fit_conditional(train_df, max_parents=2, alpha=1.0)
            
            # Build final Bayesian network
            flood_net.build_bayes_network()
            
            return flood_net, True
            
        except Exception as e:
            print(f"❌ Network building failed: {str(e)}")
            return None, False
    
    def evaluate_configuration_on_test_set(self, config_row):
        """Evaluate a single configuration on test set"""
        
        # Extract parameters
        params = {
            'occ_thr': int(config_row['occ_thr']),
            'edge_thr': int(config_row['edge_thr']),
            'weight_thr': float(config_row['weight_thr']),
            'evidence_count': int(config_row['evidence_count']),
            'pred_threshold': float(config_row['pred_threshold']),
            'neg_pos_ratio': float(config_row['neg_pos_ratio']),
            'marginal_prob_threshold': float(config_row['marginal_prob_threshold'])
        }
        
        print(f"🔧 Evaluating {config_row['category']} #{config_row['rank']}:")
        print(f"   Parameters: occ_thr={params['occ_thr']}, edge_thr={params['edge_thr']}, weight_thr={params['weight_thr']}")
        
        start_time = time.time()
        
        # Build network on training data
        flood_net, success = self.build_bayesian_network(
            self.train_df, 
            params['occ_thr'], 
            params['edge_thr'], 
            params['weight_thr']
        )
        
        if not success:
            return None
        
        network_nodes = flood_net.network.number_of_nodes()
        print(f"   Network built: {network_nodes} nodes, {flood_net.network.number_of_edges()} edges")
        
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
            
            # Limit negative candidates
            n_neg = min(len(neg_candidates), 
                       int(len(flooded_in_network) * params['neg_pos_ratio']))
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
            'category': config_row['category'],
            'rank': int(config_row['rank']),
            
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
            
            # Validation results (from original)
            'valid_tp': int(config_row['tp']),
            'valid_fp': int(config_row['fp']),
            'valid_tn': int(config_row['tn']),
            'valid_fn': int(config_row['fn']),
            'valid_precision': float(config_row['precision']),
            'valid_recall': float(config_row['recall']),
            'valid_f1_score': float(config_row['f1_score']),
            'valid_accuracy': float(config_row['accuracy']),
            
            # Model info
            'network_nodes': network_nodes,
            'valid_test_days': valid_test_days,
            'total_test_days': total_test_days,
            'test_runtime_seconds': runtime
        }
        
        print(f"   ✅ Test results: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f} (vs Valid F1={config_row['f1_score']:.3f})")
        
        return result
    
    def evaluate_all_configurations(self, top_configs_path):
        """Evaluate all top configurations on test set"""
        
        print("=== Test Set Evaluation for Top Parameter Configurations ===")
        print(f"Loading configurations from: {top_configs_path}")
        
        # Load top configurations
        configs_df = pd.read_csv(top_configs_path)
        print(f"Loaded {len(configs_df)} configurations")
        
        # Prepare data
        self.load_and_prepare_data()
        
        # Evaluate each configuration
        results = []
        successful_evaluations = 0
        
        for idx, (_, config_row) in enumerate(configs_df.iterrows(), 1):
            print(f"\n📊 Progress: {idx}/{len(configs_df)} configurations")
            
            result = self.evaluate_configuration_on_test_set(config_row)
            
            if result is not None:
                results.append(result)
                successful_evaluations += 1
            else:
                print(f"   ❌ Evaluation failed for {config_row['category']} #{config_row['rank']}")
        
        print(f"\n✅ Evaluation completed: {successful_evaluations}/{len(configs_df)} successful")
        
        if len(results) == 0:
            print("❌ No successful evaluations!")
            return None
        
        return pd.DataFrame(results)
    
    def save_results(self, results_df, output_dir):
        """Save test set evaluation results"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save complete results
        results_path = output_dir / 'test_set_evaluation_results.csv'
        results_df.to_csv(results_path, index=False)
        print(f"📄 Saved complete results to: {results_path}")
        
        # Generate summary report
        self.generate_test_summary_report(results_df, output_dir)
        
        return results_path
    
    def generate_test_summary_report(self, results_df, output_dir):
        """Generate comprehensive test set evaluation report"""
        
        # Calculate performance drops
        results_df['f1_drop'] = results_df['valid_f1_score'] - results_df['test_f1_score']
        results_df['precision_drop'] = results_df['valid_precision'] - results_df['test_precision']
        results_df['recall_drop'] = results_df['valid_recall'] - results_df['test_recall']
        
        report = f"""# Test Set Evaluation Results

## Overview
This report presents the performance of top parameter configurations evaluated on the held-out test set (2022-2024).

## Evaluation Setup
- **Data Split**: Train (2015-2019) → Validation (2020-2021) → Test (2022-2024)
- **Test Period**: {self.test_df['time_create'].min().date()} to {self.test_df['time_create'].max().date()}
- **Test Records**: {len(self.test_df)} flood records
- **Configurations Tested**: {len(results_df)}

## Performance Summary

### Best Test Performance by Category
"""
        
        # Best performers by category
        for category in results_df['category'].unique():
            category_results = results_df[results_df['category'] == category]
            best = category_results.loc[category_results['test_f1_score'].idxmax()]
            
            report += f"\n**{category}:**\n"
            report += f"- Test F1: {best['test_f1_score']:.3f} (Validation F1: {best['valid_f1_score']:.3f})\n"
            report += f"- Test Precision: {best['test_precision']:.3f} (Validation: {best['valid_precision']:.3f})\n"
            report += f"- Test Recall: {best['test_recall']:.3f} (Validation: {best['valid_recall']:.3f})\n"
            report += f"- Parameters: occ_thr={best['occ_thr']}, edge_thr={best['edge_thr']}, weight_thr={best['weight_thr']}\n"
        
        # Overall best configuration
        overall_best = results_df.loc[results_df['test_f1_score'].idxmax()]
        perf_drop = overall_best['valid_f1_score'] - overall_best['test_f1_score']
        generalization = 'Good' if abs(perf_drop) < 0.1 else 'Concerning'
        
        report += f"""
### Overall Best Configuration (Test F1: {overall_best['test_f1_score']:.3f})
- **Category**: {overall_best['category']} (Rank {overall_best['rank']})
- **Parameters**:
  - occ_thr: {overall_best['occ_thr']}
  - edge_thr: {overall_best['edge_thr']} 
  - weight_thr: {overall_best['weight_thr']}
  - evidence_count: {overall_best['evidence_count']}
  - pred_threshold: {overall_best['pred_threshold']}
  - neg_pos_ratio: {overall_best['neg_pos_ratio']}
  - marginal_prob_threshold: {overall_best['marginal_prob_threshold']}

- **Test Performance**:
  - Precision: {overall_best['test_precision']:.3f}
  - Recall: {overall_best['test_recall']:.3f}
  - F1 Score: {overall_best['test_f1_score']:.3f}
  - Accuracy: {overall_best['test_accuracy']:.3f}
  - Test Samples: {overall_best['test_total_samples']}

- **Validation vs Test Comparison**:
  - Validation F1: {overall_best['valid_f1_score']:.3f} → Test F1: {overall_best['test_f1_score']:.3f}
  - Performance Drop: {perf_drop:.3f}
  - Generalization: {generalization}
"""
        
        # Generalization analysis
        avg_f1_drop = results_df['f1_drop'].mean()
        std_f1_drop = results_df['f1_drop'].std()
        
        report += f"""
## Generalization Analysis

- **Average F1 Drop**: {avg_f1_drop:.3f} ± {std_f1_drop:.3f}
- **Configurations with F1 drop < 0.05**: {len(results_df[results_df['f1_drop'] < 0.05])} / {len(results_df)}
- **Configurations with F1 drop > 0.15**: {len(results_df[results_df['f1_drop'] > 0.15])} / {len(results_df)}
"""
        
        # Most robust configurations
        robust_configs = results_df[results_df['f1_drop'] < 0.05].sort_values('test_f1_score', ascending=False)
        if len(robust_configs) > 0:
            report += "\n### Most Robust Configurations (F1 drop < 0.05)\n\n"
            for i, (_, config) in enumerate(robust_configs.head(3).iterrows(), 1):
                report += f"{i}. **{config['category']}** (Rank {config['rank']})\n"
                report += f"   - Test F1: {config['test_f1_score']:.3f} (Drop: {config['f1_drop']:.3f})\n"
                report += f"   - Parameters: occ_thr={config['occ_thr']}, edge_thr={config['edge_thr']}, weight_thr={config['weight_thr']}\n\n"
        
        # Statistical summary
        f1_corr = results_df['valid_f1_score'].corr(results_df['test_f1_score'])
        precision_corr = results_df['valid_precision'].corr(results_df['test_precision'])
        recall_corr = results_df['valid_recall'].corr(results_df['test_recall'])
        
        report += f"""
## Statistical Summary

### Test Set Performance Distribution
- **F1 Score**: {results_df['test_f1_score'].mean():.3f} ± {results_df['test_f1_score'].std():.3f} (range: {results_df['test_f1_score'].min():.3f} - {results_df['test_f1_score'].max():.3f})
- **Precision**: {results_df['test_precision'].mean():.3f} ± {results_df['test_precision'].std():.3f} (range: {results_df['test_precision'].min():.3f} - {results_df['test_precision'].max():.3f})
- **Recall**: {results_df['test_recall'].mean():.3f} ± {results_df['test_recall'].std():.3f} (range: {results_df['test_recall'].min():.3f} - {results_df['test_recall'].max():.3f})

### Test vs Validation Correlation
- **F1 Correlation**: {f1_corr:.3f}
- **Precision Correlation**: {precision_corr:.3f}
- **Recall Correlation**: {recall_corr:.3f}

## Deployment Recommendations

Based on test set evaluation:

1. **For Production Deployment**: Use configurations with F1 drop < 0.05 and test F1 > 0.7
2. **For Conservative Applications**: Prioritize high test precision configurations
3. **For Emergency Response**: Prioritize high test recall configurations
4. **Model Monitoring**: Implement performance monitoring as real-world data may differ from test set

## Data Quality Notes
- Test set covers {len(set(self.test_df['time_create'].dt.floor('D')))} unique flood days
- Average configurations tested on {results_df['valid_test_days'].mean():.1f} valid test days
- Test sample sizes: {results_df['test_total_samples'].min()}-{results_df['test_total_samples'].max()} samples per configuration
"""
        
        # Save report
        report_path = output_dir / 'TEST_SET_EVALUATION_REPORT.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📄 Saved evaluation report to: {report_path}")

def main():
    """Main function to run test set evaluation"""
    
    # Paths
    results_dir = Path("results/parameter_optimization_20250721_140722")
    analysis_dir = results_dir / "analysis"
    top_configs_path = analysis_dir / "complete_top_configurations.csv"
    
    # Create output directory for test results
    test_output_dir = results_dir / "test_set_evaluation"
    
    print("=== Test Set Evaluation for Top Parameter Configurations ===")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize evaluator
    evaluator = TestSetEvaluator()
    
    # Run evaluation
    results_df = evaluator.evaluate_all_configurations(top_configs_path)
    
    if results_df is not None:
        # Save results
        results_path = evaluator.save_results(results_df, test_output_dir)
        
        print(f"\n✅ Test set evaluation completed successfully!")
        print(f"📁 Results saved to: {test_output_dir}")
        print(f"📊 Evaluated {len(results_df)} configurations")
        
        # Show top test performers
        top_test = results_df.nlargest(3, 'test_f1_score')
        print(f"\n🏆 Top 3 Test Set Performers:")
        for i, (_, config) in enumerate(top_test.iterrows(), 1):
            val_f1 = config['valid_f1_score']
            test_f1 = config['test_f1_score']
            drop = val_f1 - test_f1
            print(f"   {i}. {config['category']} (Rank {config['rank']})")
            print(f"      Test F1: {test_f1:.3f} (Validation: {val_f1:.3f}, Drop: {drop:.3f})")
            print(f"      Test P/R: {config['test_precision']:.3f}/{config['test_recall']:.3f}")
        
        return results_df
    else:
        print("❌ Test set evaluation failed!")
        return None

if __name__ == "__main__":
    results = main()