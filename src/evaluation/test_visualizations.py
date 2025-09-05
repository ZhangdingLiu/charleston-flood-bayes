#!/usr/bin/env python3
"""
Test visualization functions
"""

from validation_focused_evaluation import *
import pandas as pd
import os

def test_visualizations():
    """Test all visualization functions with sample data"""
    print("Testing visualization functions...")
    
    # Create test output directory
    test_output_dir = "test_visualizations_output"
    os.makedirs(test_output_dir, exist_ok=True)
    
    try:
        # 1. Load and prepare test data
        print("Loading test data...")
        df = load_and_preprocess_data()
        train_df, valid_df, test_df = split_data_by_flood_days(df)
        
        # 2. Build network
        print("Building test network...")
        flood_net, road_stats = build_bayesian_network(train_df)
        
        # 3. Create sample validation results
        print("Creating sample validation results...")
        validation_results = [
            {'evidence_count': 1, 'pred_threshold': 0.1, 'f1_score': 0.85, 'precision': 0.90, 'recall': 0.80, 'accuracy': 0.82, 'total_samples': 149},
            {'evidence_count': 1, 'pred_threshold': 0.3, 'f1_score': 0.42, 'precision': 1.00, 'recall': 0.27, 'accuracy': 0.63, 'total_samples': 149},
            {'evidence_count': 1, 'pred_threshold': 0.5, 'f1_score': 0.03, 'precision': 1.00, 'recall': 0.01, 'accuracy': 0.50, 'total_samples': 149},
            {'evidence_count': 2, 'pred_threshold': 0.1, 'f1_score': 0.83, 'precision': 0.81, 'recall': 0.85, 'accuracy': 0.83, 'total_samples': 110},
            {'evidence_count': 2, 'pred_threshold': 0.3, 'f1_score': 0.47, 'precision': 1.00, 'recall': 0.31, 'accuracy': 0.65, 'total_samples': 110},
            {'evidence_count': 2, 'pred_threshold': 0.5, 'f1_score': 0.07, 'precision': 1.00, 'recall': 0.04, 'accuracy': 0.52, 'total_samples': 110},
            {'evidence_count': 3, 'pred_threshold': 0.1, 'f1_score': 0.87, 'precision': 0.83, 'recall': 0.91, 'accuracy': 0.86, 'total_samples': 86},
            {'evidence_count': 3, 'pred_threshold': 0.3, 'f1_score': 0.46, 'precision': 1.00, 'recall': 0.30, 'accuracy': 0.65, 'total_samples': 86},
            {'evidence_count': 3, 'pred_threshold': 0.5, 'f1_score': 0.17, 'precision': 1.00, 'recall': 0.09, 'accuracy': 0.55, 'total_samples': 86}
        ]
        
        # 4. Create sample final result
        final_result = {
            'f1_score': 0.837, 'precision': 0.818, 'recall': 0.857, 'accuracy': 0.819,
            'total_samples': 83, 'positive_samples': 41, 'negative_samples': 42,
            'tp': 35, 'fp': 6, 'tn': 36, 'fn': 6, 'evaluated_days': 11, 'skipped_days': 18
        }
        
        # 5. Test each visualization function
        print("\nTesting visualization functions:")
        
        # Test 1: Bayesian Network
        print("1. Testing Bayesian network visualization...")
        network_file = visualize_bayesian_network(flood_net, road_stats, test_output_dir)
        
        # Test 2: Parameter Sensitivity
        print("2. Testing parameter sensitivity analysis...")
        sensitivity_file = visualize_parameter_sensitivity(validation_results, test_output_dir)
        
        # Test 3: Performance Comparison
        print("3. Testing performance comparison...")
        comparison_file = visualize_performance_comparison(validation_results, final_result, test_output_dir)
        
        # Test 4: Data Distribution
        print("4. Testing data distribution analysis...")
        distribution_file = visualize_data_distribution(train_df, valid_df, test_df, test_output_dir)
        
        print(f"\n✅ All visualization tests completed successfully!")
        print(f"📁 Test output directory: {test_output_dir}")
        print(f"Generated files:")
        for file in os.listdir(test_output_dir):
            if file.endswith(('.png', '.pdf')):
                print(f"   • {file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_visualizations()
    if success:
        print("\n🎉 Visualization functions are ready to use!")
    else:
        print("\n❌ Please fix the issues before using visualizations.")