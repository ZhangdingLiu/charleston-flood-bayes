#!/usr/bin/env python3
"""
Generate visualizations for existing validation results
Usage: python generate_visualizations.py <results_directory>
"""

import sys
import os
import json
import pandas as pd
from validation_focused_evaluation import *

def load_existing_results(results_dir):
    """Load existing validation results"""
    
    # Load experiment summary
    summary_file = os.path.join(results_dir, "experiment_summary.json")
    if not os.path.exists(summary_file):
        raise FileNotFoundError(f"Experiment summary not found: {summary_file}")
    
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    # Load validation results
    validation_file = os.path.join(results_dir, "validation_results.csv")
    if not os.path.exists(validation_file):
        raise FileNotFoundError(f"Validation results not found: {validation_file}")
    
    validation_df = pd.read_csv(validation_file)
    validation_results = validation_df.to_dict('records')
    
    # Extract final result from summary
    final_result = summary.get('final_test_performance', {})
    
    # Extract best parameters
    best_params = summary.get('best_parameters', {})
    
    return validation_results, final_result, best_params, summary

def recreate_network_for_visualization(summary):
    """Recreate network and data for visualization"""
    
    # Load and split data
    df = load_and_preprocess_data()
    train_df, valid_df, test_df = split_data_by_flood_days(df)
    
    # Build network
    flood_net, road_stats = build_bayesian_network(train_df)
    
    return flood_net, road_stats, train_df, valid_df, test_df

def main():
    """Main function to generate visualizations for existing results"""
    
    if len(sys.argv) != 2:
        print("Usage: python generate_visualizations.py <results_directory>")
        print("Example: python generate_visualizations.py validation_focused_results_20250714_174125")
        return
    
    results_dir = sys.argv[1]
    
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return
    
    try:
        print(f"Loading results from: {results_dir}")
        
        # Load existing results
        validation_results, final_result, best_params, summary = load_existing_results(results_dir)
        
        print(f"✓ Loaded {len(validation_results)} validation results")
        print(f"✓ Best parameters: {best_params}")
        print(f"✓ Final test F1-Score: {final_result.get('f1_score', 'N/A')}")
        
        # Recreate network and data for visualization
        print("Recreating network and data for visualization...")
        flood_net, road_stats, train_df, valid_df, test_df = recreate_network_for_visualization(summary)
        
        # Generate all visualizations
        print("Generating visualizations...")
        visualization_files = generate_all_visualizations(
            flood_net, road_stats, validation_results, best_params, final_result,
            train_df, valid_df, test_df, results_dir
        )
        
        print(f"\n✅ Successfully generated {len(visualization_files)} visualization files!")
        print(f"📁 Files saved in: {results_dir}")
        
    except Exception as e:
        print(f"❌ Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()