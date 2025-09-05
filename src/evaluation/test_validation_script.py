#!/usr/bin/env python3
"""
Test script for validation_focused_evaluation.py
Quick test of key functions before full run
"""

import pandas as pd
from validation_focused_evaluation import (
    load_and_preprocess_data, 
    split_data_by_flood_days,
    build_bayesian_network
)

def test_data_loading():
    """Test data loading and preprocessing"""
    print("Testing data loading...")
    try:
        df = load_and_preprocess_data()
        print(f"✓ Data loaded: {len(df)} records")
        
        # Check required columns
        required_cols = ['time_create', 'link_id', 'id', 'flood_date']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ Missing columns: {missing_cols}")
            return False
        else:
            print("✓ All required columns present")
        
        return True
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def test_data_splitting():
    """Test temporal data splitting"""
    print("\nTesting data splitting...")
    try:
        df = load_and_preprocess_data()
        train_df, valid_df, test_df = split_data_by_flood_days(df)
        
        # Check splits are non-empty
        if len(train_df) == 0 or len(valid_df) == 0 or len(test_df) == 0:
            print("❌ One or more splits are empty")
            return False
        
        # Check no date overlap
        train_dates = set(train_df['flood_date'])
        valid_dates = set(valid_df['flood_date'])
        test_dates = set(test_df['flood_date'])
        
        if train_dates & valid_dates or train_dates & test_dates or valid_dates & test_dates:
            print("❌ Date overlap detected between splits")
            return False
        
        print("✓ Data splitting successful")
        print(f"  Train: {len(train_df)} records")
        print(f"  Valid: {len(valid_df)} records") 
        print(f"  Test: {len(test_df)} records")
        
        return True
    except Exception as e:
        print(f"❌ Data splitting failed: {e}")
        return False

def test_network_building():
    """Test Bayesian network building"""
    print("\nTesting network building...")
    try:
        df = load_and_preprocess_data()
        train_df, _, _ = split_data_by_flood_days(df)
        
        flood_net, road_stats = build_bayesian_network(train_df)
        
        # Check network is built
        if flood_net.network_bayes is None:
            print("❌ Bayesian network not built")
            return False
        
        # Check road stats
        if road_stats is None or len(road_stats) == 0:
            print("❌ Road statistics not generated")
            return False
        
        print("✓ Network building successful")
        print(f"  Network nodes: {len(flood_net.network_bayes.nodes())}")
        print(f"  Road statistics: {len(road_stats)} roads")
        
        return True
    except Exception as e:
        print(f"❌ Network building failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("TESTING VALIDATION_FOCUSED_EVALUATION.PY")
    print("=" * 60)
    
    tests = [
        test_data_loading,
        test_data_splitting, 
        test_network_building
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n{'='*60}")
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("✓ All tests passed! Ready for full evaluation.")
        return True
    else:
        print("❌ Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()