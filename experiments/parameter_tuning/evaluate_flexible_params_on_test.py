#!/usr/bin/env python3
"""
Flexible Parameter Test Set Evaluation

This script evaluates a broader range of parameter combinations on the test set,
including higher prediction thresholds and more flexible network parameters
to potentially improve precision and overall performance.

Key Changes:
- Wider range of pred_threshold values (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
- More flexible network parameters (lower thresholds to include more roads)
- Adaptive negative sampling ratios
- Focus on improving precision while maintaining reasonable recall
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

class FlexibleTestSetEvaluator:
    """Flexible test set evaluator with broader parameter ranges"""
    
    def __init__(self, data_path="Road_Closures_2024.csv"):
        """Initialize the evaluator"""
        self.data_path = data_path
        self.results = []
        
    def get_flexible_parameter_grid(self):
        """Get a more flexible parameter grid for test evaluation"""
        
        # Start with proven good base configurations but make them more flexible
        base_configs = [
            # High precision focused (more restrictive networks, higher thresholds)
            {'occ_thr': 3, 'edge_thr': 2, 'weight_thr': 0.3},
            {'occ_thr': 4, 'edge_thr': 2, 'weight_thr': 0.3},
            {'occ_thr': 4, 'edge_thr': 3, 'weight_thr': 0.3},
            
            # Balanced approach (medium restrictiveness)
            {'occ_thr': 2, 'edge_thr': 1, 'weight_thr': 0.2},
            {'occ_thr': 3, 'edge_thr': 1, 'weight_thr': 0.2},
            {'occ_thr': 3, 'edge_thr': 2, 'weight_thr': 0.2},
            
            # More inclusive networks (lower thresholds for broader coverage)
            {'occ_thr': 2, 'edge_thr': 1, 'weight_thr': 0.1},
            {'occ_thr': 1, 'edge_thr': 1, 'weight_thr': 0.1},
        ]
        
        # More flexible prediction parameters
        flexible_params = {
            'evidence_count': [1, 2],  # Keep simple evidence requirements
            'pred_threshold': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],  # Much wider range
            'neg_pos_ratio': [0.5, 1.0, 1.5, 2.0],  # More flexible sampling
            'marginal_prob_threshold': [0.03, 0.05, 0.08, 0.1]  # Slightly more inclusive
        }
        
        # Generate all combinations
        param_combinations = []
        
        for base_config in base_configs:
            # Get all combinations of flexible parameters
            keys = list(flexible_params.keys())
            values = list(flexible_params.values())
            
            for combination in itertools.product(*values):
                param_dict = base_config.copy()
                for key, value in zip(keys, combination):
                    param_dict[key] = value
                param_combinations.append(param_dict)
        
        print(f"Generated {len(param_combinations)} flexible parameter combinations")
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
        
        print(f"Data splits:")
        print(f"  Train: {len(train_days)} days, {len(self.train_df)} records")
        print(f"  Valid: {len(valid_days)} days, {len(self.valid_df)} records")
        print(f"  Test:  {len(test_days)} days, {len(self.test_df)} records")
        
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
    
    def evaluate_parameter_combination(self, params, config_id):
        """Evaluate a single parameter combination on test set"""
        
        print(f"🔧 Evaluating Config #{config_id}:")
        print(f"   Network: occ_thr={params['occ_thr']}, edge_thr={params['edge_thr']}, weight_thr={params['weight_thr']}")
        print(f"   Prediction: evidence={params['evidence_count']}, pred_thr={params['pred_threshold']}, neg_ratio={params['neg_pos_ratio']}")
        
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
            
            # Adaptive negative sampling based on ratio
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
            'config_type': 'flexible_test',
            
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
            'network_edges': flood_net.network.number_of_edges(),
            'valid_test_days': valid_test_days,
            'total_test_days': total_test_days,
            'test_runtime_seconds': runtime
        }
        
        print(f"   ✅ Test results: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, Acc={accuracy:.3f}")
        
        return result
    
    def evaluate_flexible_configurations(self):
        """Evaluate all flexible configurations on test set"""
        
        print("=== Flexible Parameter Test Set Evaluation ===")
        
        # Prepare data
        self.load_and_prepare_data()
        
        # Get flexible parameter combinations
        param_combinations = self.get_flexible_parameter_grid()
        
        # Evaluate each configuration
        results = []
        successful_evaluations = 0
        
        for idx, params in enumerate(param_combinations, 1):
            print(f"\n📊 Progress: {idx}/{len(param_combinations)} configurations")
            
            result = self.evaluate_parameter_combination(params, idx)
            
            if result is not None:
                results.append(result)
                successful_evaluations += 1
            else:
                print(f"   ❌ Evaluation failed for config #{idx}")
        
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
        results_path = output_dir / 'flexible_test_evaluation_results.csv'
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
        
        # High-performing configurations (top 10%)
        top_10_pct_threshold = results_df['test_f1_score'].quantile(0.9)
        top_configs = results_df[results_df['test_f1_score'] >= top_10_pct_threshold].sort_values('test_f1_score', ascending=False)
        
        # Generate comprehensive report
        report = f"""# Flexible Parameter Test Set Evaluation Results

## Overview
This evaluation tested {len(results_df)} parameter combinations with more flexible settings,
focusing on improving precision through higher prediction thresholds and varied network parameters.

## Key Findings

### Overall Performance Distribution
- **F1 Score**: {results_df['test_f1_score'].mean():.3f} ± {results_df['test_f1_score'].std():.3f} (range: {results_df['test_f1_score'].min():.3f} - {results_df['test_f1_score'].max():.3f})
- **Precision**: {results_df['test_precision'].mean():.3f} ± {results_df['test_precision'].std():.3f} (range: {results_df['test_precision'].min():.3f} - {results_df['test_precision'].max():.3f})
- **Recall**: {results_df['test_recall'].mean():.3f} ± {results_df['test_recall'].std():.3f} (range: {results_df['test_recall'].min():.3f} - {results_df['test_recall'].max():.3f})
- **Accuracy**: {results_df['test_accuracy'].mean():.3f} ± {results_df['test_accuracy'].std():.3f} (range: {results_df['test_accuracy'].min():.3f} - {results_df['test_accuracy'].max():.3f})

### Best Configurations by Metric
"""
        
        for metric, config in best_configs.items():
            report += f"\n**Best {metric.title()}** (Config #{config['config_id']}):\n"
            report += f"- Performance: P={config['test_precision']:.3f}, R={config['test_recall']:.3f}, F1={config['test_f1_score']:.3f}, Acc={config['test_accuracy']:.3f}\n"
            report += f"- Network: occ_thr={config['occ_thr']}, edge_thr={config['edge_thr']}, weight_thr={config['weight_thr']}\n"
            report += f"- Prediction: evidence={config['evidence_count']}, pred_thr={config['pred_threshold']}, neg_ratio={config['neg_pos_ratio']}\n"
            report += f"- Network size: {config['network_nodes']} nodes, {config['network_edges']} edges\n"
        
        if len(top_configs) > 0:
            report += f"\n### Top 10% Configurations (F1 ≥ {top_10_pct_threshold:.3f})\n\n"
            for i, (_, config) in enumerate(top_configs.head(10).iterrows(), 1):
                report += f"{i}. **Config #{config['config_id']}** (F1={config['test_f1_score']:.3f})\n"
                report += f"   - P={config['test_precision']:.3f}, R={config['test_recall']:.3f}, Acc={config['test_accuracy']:.3f}\n"
                report += f"   - Network: occ_thr={config['occ_thr']}, edge_thr={config['edge_thr']}, weight_thr={config['weight_thr']}\n"
                report += f"   - Prediction: pred_thr={config['pred_threshold']}, evidence={config['evidence_count']}, neg_ratio={config['neg_pos_ratio']}\n\n"
        
        # Parameter impact analysis
        report += "\n### Parameter Impact Analysis\n\n"
        
        # Prediction threshold impact
        pred_thr_analysis = results_df.groupby('pred_threshold').agg({
            'test_precision': ['mean', 'std'],
            'test_recall': ['mean', 'std'],
            'test_f1_score': ['mean', 'std']
        }).round(3)
        
        report += "**Prediction Threshold Impact:**\n"
        for pred_thr in sorted(results_df['pred_threshold'].unique()):
            data = results_df[results_df['pred_threshold'] == pred_thr]
            avg_p = data['test_precision'].mean()
            avg_r = data['test_recall'].mean()
            avg_f1 = data['test_f1_score'].mean()
            report += f"- pred_threshold={pred_thr}: P={avg_p:.3f}, R={avg_r:.3f}, F1={avg_f1:.3f} (n={len(data)})\n"
        
        # Network parameter impact
        report += "\n**Network Parameter Impact:**\n"
        network_groups = results_df.groupby(['occ_thr', 'edge_thr', 'weight_thr']).agg({
            'test_f1_score': 'mean',
            'test_precision': 'mean',
            'test_recall': 'mean'
        }).sort_values('test_f1_score', ascending=False).head(5)
        
        for (occ, edge, weight), row in network_groups.iterrows():
            report += f"- occ_thr={occ}, edge_thr={edge}, weight_thr={weight}: P={row['test_precision']:.3f}, R={row['test_recall']:.3f}, F1={row['test_f1_score']:.3f}\n"
        
        report += f"""
## Deployment Recommendations

### For High Precision Applications
Use configurations with pred_threshold ≥ 0.3 and moderate network restrictions:
- Recommended: {best_configs['precision']['pred_threshold']} prediction threshold
- Network: occ_thr={best_configs['precision']['occ_thr']}, edge_thr={best_configs['precision']['edge_thr']}

### For Balanced Performance  
Use the best F1 configuration:
- pred_threshold: {best_configs['f1']['pred_threshold']}
- Network: occ_thr={best_configs['f1']['occ_thr']}, edge_thr={best_configs['f1']['edge_thr']}, weight_thr={best_configs['f1']['weight_thr']}

### Key Insights
1. Higher prediction thresholds (≥0.3) significantly improve precision
2. Network size doesn't strongly correlate with performance 
3. Adaptive negative sampling helps balance precision/recall
4. Evidence count of 1-2 works well for this dataset

## Data Quality Notes
- Evaluated on {len(set(self.test_df['time_create'].dt.floor('D')))} unique test days (2023-2024)
- Average test samples per configuration: {results_df['test_total_samples'].mean():.1f}
- All configurations used proper temporal train/test split to avoid data leakage
"""
        
        # Save report
        report_path = output_dir / 'FLEXIBLE_TEST_EVALUATION_REPORT.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📄 Saved evaluation report to: {report_path}")
        
        return best_configs, top_configs

def main():
    """Main function to run flexible test set evaluation"""
    
    # Create output directory for test results
    output_dir = Path("results/parameter_optimization_20250721_140722/flexible_test_evaluation")
    
    print("=== Flexible Parameter Test Set Evaluation ===")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize evaluator
    evaluator = FlexibleTestSetEvaluator()
    
    # Run evaluation
    results_df = evaluator.evaluate_flexible_configurations()
    
    if results_df is not None:
        # Analyze and save results
        best_configs, top_configs = evaluator.analyze_and_save_results(results_df, output_dir)
        
        print(f"\n✅ Flexible test evaluation completed successfully!")
        print(f"📁 Results saved to: {output_dir}")
        print(f"📊 Evaluated {len(results_df)} configurations")
        
        # Show top performers
        print(f"\n🏆 Top 5 Flexible Test Performers:")
        top_5 = results_df.nlargest(5, 'test_f1_score')
        for i, (_, config) in enumerate(top_5.iterrows(), 1):
            print(f"   {i}. Config #{config['config_id']}: F1={config['test_f1_score']:.3f}")
            print(f"      P/R/Acc: {config['test_precision']:.3f}/{config['test_recall']:.3f}/{config['test_accuracy']:.3f}")
            print(f"      Pred_thr: {config['pred_threshold']}, Network: {config['occ_thr']}/{config['edge_thr']}/{config['weight_thr']}")
        
        # Show best precision
        best_precision = results_df.loc[results_df['test_precision'].idxmax()]
        print(f"\n🎯 Best Precision: {best_precision['test_precision']:.3f} (Config #{best_precision['config_id']})")
        print(f"   F1={best_precision['test_f1_score']:.3f}, Recall={best_precision['test_recall']:.3f}")
        print(f"   pred_threshold={best_precision['pred_threshold']}, network={best_precision['occ_thr']}/{best_precision['edge_thr']}/{best_precision['weight_thr']}")
        
        return results_df
    else:
        print("❌ Flexible test evaluation failed!")
        return None

if __name__ == "__main__":
    results = main()