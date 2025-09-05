#!/usr/bin/env python3
"""
Debug script to understand why test evaluation is failing
"""

import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from model import FloodBayesNetwork
except ImportError:
    try:
        from core.model import FloodBayesNetwork
    except ImportError:
        print("❌ Cannot import FloodBayesNetwork")
        sys.exit(1)

def debug_data_splits():
    """Debug data splits and test set characteristics"""
    
    # Load and preprocess data
    df = pd.read_csv("Road_Closures_2024.csv")
    df['time_create'] = pd.to_datetime(df['START'], utc=True)
    df = df[df['REASON'] == 'FLOOD'].copy()
    df['flood_date'] = df['time_create'].dt.floor('D')
    df['link_id'] = df['STREET'].str.upper().str.replace(' ', '_')
    df['link_id'] = df['link_id'].astype(str)
    df['id'] = df['OBJECTID'].astype(str)
    
    print(f"Total flood records: {len(df)}")
    
    # Time-based split by flood days
    unique_days = sorted(df['flood_date'].unique())
    n_days = len(unique_days)
    
    train_end = int(n_days * 0.6)
    valid_end = int(n_days * 0.8)
    
    train_days = unique_days[:train_end]
    valid_days = unique_days[train_end:valid_end]
    test_days = unique_days[valid_end:]
    
    train_df = df[df['flood_date'].isin(train_days)].copy()
    valid_df = df[df['flood_date'].isin(valid_days)].copy()
    test_df = df[df['flood_date'].isin(test_days)].copy()
    
    print(f"\nData splits:")
    print(f"  Train: {len(train_days)} days, {len(train_df)} records")
    print(f"  Valid: {len(valid_days)} days, {len(valid_df)} records")  
    print(f"  Test:  {len(test_days)} days, {len(test_df)} records")
    
    print(f"\nTest days:")
    for i, day in enumerate(test_days):
        day_records = test_df[test_df['flood_date'] == day]
        unique_roads = day_records['link_id'].unique()
        print(f"  {i+1:2d}. {day.date()}: {len(day_records)} records, {len(unique_roads)} roads")
        if len(unique_roads) <= 10:
            print(f"      Roads: {list(unique_roads)}")
        else:
            print(f"      Roads: {list(unique_roads[:10])}...")
    
    return train_df, valid_df, test_df

def debug_network_building(train_df):
    """Debug network building process"""
    
    print(f"\n=== Network Building Debug ===")
    
    # Try different parameter combinations
    param_combinations = [
        (4, 3, 0.2),  # Best F1 config
        (2, 1, 0.2),  # High precision config
    ]
    
    for occ_thr, edge_thr, weight_thr in param_combinations:
        print(f"\nTesting parameters: occ_thr={occ_thr}, edge_thr={edge_thr}, weight_thr={weight_thr}")
        
        try:
            flood_net = FloodBayesNetwork(t_window="D")
            flood_net.fit_marginal(train_df)
            
            print(f"  Marginals calculated for {len(flood_net.marginals)} roads")
            
            flood_net.build_network_by_co_occurrence(
                train_df,
                occ_thr=occ_thr,
                edge_thr=edge_thr,
                weight_thr=weight_thr,
                report=False
            )
            
            print(f"  Network: {flood_net.network.number_of_nodes()} nodes, {flood_net.network.number_of_edges()} edges")
            
            if flood_net.network.number_of_nodes() > 0:
                print(f"  Network nodes: {list(flood_net.network.nodes())[:10]}...")
                
                # Check marginal probabilities
                marginals_in_network = flood_net.marginals[
                    flood_net.marginals['link_id'].isin(flood_net.network.nodes())
                ]
                print(f"  Marginal prob range: {marginals_in_network['p'].min():.3f} - {marginals_in_network['p'].max():.3f}")
                
                # Try to fit conditional probabilities
                flood_net.fit_conditional(train_df, max_parents=2, alpha=1.0)
                flood_net.build_bayes_network()
                print(f"  ✅ Network built successfully")
                
                return flood_net
            else:
                print(f"  ❌ Empty network")
                
        except Exception as e:
            print(f"  ❌ Failed: {str(e)}")
    
    return None

def debug_test_evaluation(flood_net, test_df):
    """Debug test evaluation process"""
    
    if flood_net is None:
        print("❌ No valid network to test")
        return
    
    print(f"\n=== Test Evaluation Debug ===")
    
    network_roads = set(flood_net.network.nodes())
    print(f"Network has {len(network_roads)} roads")
    
    # Check test data structure
    test_by_day = test_df.groupby(test_df['time_create'].dt.floor('D'))
    print(f"Test data has {len(test_by_day)} days")
    
    valid_days = 0
    total_candidates = 0
    
    # Use parameters from best config
    params = {
        'evidence_count': 1,
        'marginal_prob_threshold': 0.08,
        'neg_pos_ratio': 1.0,
        'pred_threshold': 0.1
    }
    
    for i, (test_date, day_group) in enumerate(test_by_day):
        flooded_roads = set(day_group['link_id'].unique())
        flooded_in_network = flooded_roads & network_roads
        
        print(f"\nDay {i+1} ({test_date.date()}):")
        print(f"  Total flooded roads: {len(flooded_roads)}")
        print(f"  Flooded roads in network: {len(flooded_in_network)}")
        
        if len(flooded_in_network) < params['evidence_count']:
            print(f"  ❌ Not enough evidence roads (need {params['evidence_count']})")
            continue
        
        valid_days += 1
        print(f"  ✅ Valid day for evaluation")
        
        # Select evidence roads
        evidence_roads = list(flooded_in_network)[:params['evidence_count']]
        candidate_roads = network_roads - set(evidence_roads)
        
        print(f"  Evidence roads: {evidence_roads}")
        print(f"  Candidate roads: {len(candidate_roads)}")
        
        # Apply negative sampling filter
        marginals = flood_net.marginals[flood_net.marginals['link_id'].isin(candidate_roads)]
        neg_candidates = marginals[
            marginals['p'] >= params['marginal_prob_threshold']
        ]['link_id'].tolist()
        
        print(f"  Candidates after marginal filter (≥{params['marginal_prob_threshold']}): {len(neg_candidates)}")
        
        if len(neg_candidates) == 0:
            print(f"  ❌ No candidates pass marginal probability filter")
            continue
        
        # Limit negative candidates
        n_neg = min(len(neg_candidates), 
                   int(len(flooded_in_network) * params['neg_pos_ratio']))
        neg_candidates = neg_candidates[:n_neg]
        
        print(f"  Final candidates (after ratio limit): {len(neg_candidates)}")
        total_candidates += len(neg_candidates)
        
        # Try a few inferences
        evidence = {road: 1 for road in evidence_roads}
        successful_inferences = 0
        
        for j, candidate in enumerate(neg_candidates[:5]):  # Test first 5
            try:
                prob = flood_net.query_road_flood_probability(candidate, evidence)
                pred = 1 if prob >= params['pred_threshold'] else 0
                true = 1 if candidate in flooded_in_network else 0
                
                print(f"    Candidate {j+1} ({candidate}): P={prob:.3f}, Pred={pred}, True={true}")
                successful_inferences += 1
                
            except Exception as e:
                print(f"    Candidate {j+1} ({candidate}): ❌ Inference failed: {str(e)}")
        
        print(f"  Successful inferences: {successful_inferences}/{min(5, len(neg_candidates))}")
        
        if i >= 5:  # Only show first 5 days in detail
            break
    
    print(f"\nSummary:")
    print(f"  Valid test days: {valid_days}/{len(test_by_day)}")
    print(f"  Total candidates for evaluation: {total_candidates}")

def main():
    """Main debug function"""
    
    print("=== Test Set Evaluation Debug ===")
    
    # Debug data splits
    train_df, valid_df, test_df = debug_data_splits()
    
    # Debug network building
    flood_net = debug_network_building(train_df)
    
    # Debug test evaluation
    debug_test_evaluation(flood_net, test_df)

if __name__ == "__main__":
    main()