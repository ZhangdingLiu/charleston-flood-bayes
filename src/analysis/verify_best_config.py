#!/usr/bin/env python3
"""
Verification and Analysis of Best Configuration
Verify RecentActivity_0 results and analyze Bayesian network structure
"""

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
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
    print("VERIFICATION OF BEST CONFIGURATION")
    print("Target: RecentActivity_0, pred_threshold=0.2, evidence_ratio=0.5")
    print("Expected: Precision=0.957, Recall=0.536, TP=45, FP=2, TN=40, FN=39")
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

def build_and_analyze_network(train_df):
    """Build Bayesian network and analyze its structure"""
    print(f"\n{'='*60}")
    print("BAYESIAN NETWORK CONSTRUCTION & ANALYSIS")
    print(f"{'='*60}")
    
    # Build network with exact same parameters
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
    
    # Analyze network structure
    edges = list(flood_net.network_bayes.edges())
    print(f"Network edges: {len(edges)}")
    
    # Node analysis
    print(f"\nNetwork Nodes ({len(bn_nodes)}):")
    sorted_nodes = sorted(bn_nodes)
    for i, node in enumerate(sorted_nodes):
        prob = marginals_dict.get(node, 0.0)
        parents = list(flood_net.network_bayes.predecessors(node))
        children = list(flood_net.network_bayes.successors(node))
        print(f"  {i+1:2d}. {node:<20} | Prob: {prob:.4f} | Parents: {len(parents)} | Children: {len(children)}")
        if len(parents) > 0:
            print(f"      Parents: {parents}")
    
    # Network statistics
    parent_counts = [len(list(flood_net.network_bayes.predecessors(node))) for node in bn_nodes]
    child_counts = [len(list(flood_net.network_bayes.successors(node))) for node in bn_nodes]
    
    print(f"\nNetwork Statistics:")
    print(f"  Total nodes: {len(bn_nodes)}")
    print(f"  Total edges: {len(edges)}")
    print(f"  Average parents per node: {np.mean(parent_counts):.2f}")
    print(f"  Max parents per node: {np.max(parent_counts)}")
    print(f"  Average children per node: {np.mean(child_counts):.2f}")
    print(f"  Max children per node: {np.max(child_counts)}")
    print(f"  Nodes with no parents: {sum(1 for count in parent_counts if count == 0)}")
    print(f"  Nodes with no children: {sum(1 for count in child_counts if count == 0)}")
    
    # Marginal probability distribution
    probs = list(marginals_dict.values())
    print(f"\nMarginal Probability Distribution:")
    print(f"  Min: {np.min(probs):.4f}")
    print(f"  Max: {np.max(probs):.4f}")
    print(f"  Mean: {np.mean(probs):.4f}")
    print(f"  Median: {np.median(probs):.4f}")
    print(f"  Std: {np.std(probs):.4f}")
    
    return flood_net, bn_nodes, marginals_dict

def analyze_negative_candidates(train_df, bn_nodes, marginals_dict):
    """Analyze the RecentActivity_0 negative candidates"""
    print(f"\n{'='*60}")
    print("NEGATIVE CANDIDATES ANALYSIS (RecentActivity_0)")
    print(f"{'='*60}")
    
    # Build road statistics
    road_stats = train_df.groupby('link_id').agg({
        'time_create': ['count', 'min', 'max'],
        'year': lambda x: len(x.unique())
    }).round(4)
    road_stats.columns = ['flood_count', 'first_flood', 'last_flood', 'years_active']
    road_stats = road_stats.reset_index()
    
    # Add marginal probabilities
    road_stats['marginal_prob'] = road_stats['link_id'].map(marginals_dict)
    road_stats = road_stats[road_stats['link_id'].isin(bn_nodes)].copy()
    
    # Recent activity analysis (2020 onwards)
    recent_activity = train_df[train_df['year'] >= 2020].groupby('link_id')['time_create'].count()
    road_stats['recent_floods'] = road_stats['link_id'].map(recent_activity).fillna(0).astype(int)
    
    # Get RecentActivity_0 candidates (roads with 0 recent floods)
    negative_candidates = road_stats[road_stats['recent_floods'] == 0]['link_id'].tolist()
    
    print(f"RecentActivity_0 Strategy: Select roads with 0 floods since 2020")
    print(f"Total negative candidates: {len(negative_candidates)}")
    print(f"Expected: 9 candidates")
    
    print(f"\nDetailed Negative Candidates:")
    neg_stats = road_stats[road_stats['link_id'].isin(negative_candidates)].copy()
    neg_stats = neg_stats.sort_values('marginal_prob')
    
    for i, (_, row) in enumerate(neg_stats.iterrows(), 1):
        print(f"  {i}. {row['link_id']:<20} | "
              f"Marginal Prob: {row['marginal_prob']:.4f} | "
              f"Total Floods: {row['flood_count']:2d} | "
              f"Recent Floods: {row['recent_floods']} | "
              f"First: {row['first_flood'].strftime('%Y-%m-%d')} | "
              f"Last: {row['last_flood'].strftime('%Y-%m-%d')}")
    
    # Verification
    if len(negative_candidates) == 9:
        print(f"\n✅ VERIFICATION PASSED: Found exactly 9 negative candidates")
    else:
        print(f"\n❌ VERIFICATION FAILED: Expected 9 candidates, found {len(negative_candidates)}")
    
    return negative_candidates, road_stats

def verify_configuration_results(flood_net, test_df, bn_nodes, negative_candidates):
    """Verify the exact configuration results"""
    print(f"\n{'='*60}")
    print("CONFIGURATION VERIFICATION")
    print("Config: RecentActivity_0, pred_threshold=0.2, evidence_ratio=0.5, neg_pos_ratio=0.5")
    print(f"{'='*60}")
    
    # Configuration parameters
    pred_threshold = 0.2
    evidence_ratio = 0.5
    neg_pos_ratio = 0.5
    
    # Expected results
    expected_results = {
        'tp': 45, 'fp': 2, 'tn': 40, 'fn': 39,
        'precision': 0.957446809,
        'recall': 0.535714286,
        'positive_samples': 84,
        'negative_samples': 42,
        'total_samples': 126
    }
    
    print(f"Expected Results:")
    print(f"  TP: {expected_results['tp']}, FP: {expected_results['fp']}, TN: {expected_results['tn']}, FN: {expected_results['fn']}")
    print(f"  Precision: {expected_results['precision']:.6f}")
    print(f"  Recall: {expected_results['recall']:.6f}")
    print(f"  Samples: {expected_results['positive_samples']} pos, {expected_results['negative_samples']} neg, {expected_results['total_samples']} total")
    
    # Run evaluation
    results = []
    detailed_predictions = []
    test_by_date = test_df.groupby(test_df["time_create"].dt.floor("D"))
    evaluated_days = 0
    
    print(f"\nRunning evaluation...")
    
    for date, day_group in test_by_date:
        flooded_roads = list(day_group["link_id"].unique())
        flooded_in_bn = [road for road in flooded_roads if road in bn_nodes]
        
        if len(flooded_in_bn) < 2:
            continue
            
        evaluated_days += 1
        
        # Evidence selection (50% of flooded roads)
        evidence_count = max(1, int(len(flooded_in_bn) * evidence_ratio))
        evidence_roads = flooded_in_bn[:evidence_count]
        target_roads = flooded_in_bn[evidence_count:]
        evidence = {road: 1 for road in evidence_roads}
        
        # Store day details
        day_detail = {
            'date': date.strftime('%Y-%m-%d'),
            'all_flooded_roads': flooded_roads,
            'flooded_in_bn': flooded_in_bn,
            'evidence_roads': evidence_roads,
            'positive_targets': target_roads,
            'negative_targets': [],
            'predictions': []
        }
        
        # Positive samples (confirmed floods)
        for target_road in target_roads:
            try:
                result = flood_net.infer_w_evidence(target_road, evidence)
                prob_flood = result['flooded']
                prediction = 1 if prob_flood >= pred_threshold else 0
                
                pred_detail = {
                    'target_road': target_road,
                    'type': 'Positive',
                    'prob_flood': prob_flood,
                    'prediction': prediction,
                    'true_label': 1,
                    'is_correct': (prediction == 1)
                }
                
                results.append({
                    'type': 'Positive',
                    'target_road': target_road,
                    'prob_flood': prob_flood,
                    'prediction': prediction,
                    'true_label': 1,
                    'date': date.strftime('%Y-%m-%d')
                })
                
                day_detail['predictions'].append(pred_detail)
                
            except Exception as e:
                print(f"Error predicting {target_road}: {e}")
                continue
        
        # Negative samples
        available_negatives = [road for road in negative_candidates if road not in flooded_roads]
        n_negatives = min(len(available_negatives), max(1, int(len(target_roads) * neg_pos_ratio)))
        selected_negatives = available_negatives[:n_negatives]
        day_detail['negative_targets'] = selected_negatives
        
        for neg_road in selected_negatives:
            try:
                result = flood_net.infer_w_evidence(neg_road, evidence)
                prob_flood = result['flooded']
                prediction = 1 if prob_flood >= pred_threshold else 0
                
                pred_detail = {
                    'target_road': neg_road,
                    'type': 'Negative',
                    'prob_flood': prob_flood,
                    'prediction': prediction,
                    'true_label': 0,
                    'is_correct': (prediction == 0)
                }
                
                results.append({
                    'type': 'Negative',
                    'target_road': neg_road,
                    'prob_flood': prob_flood,
                    'prediction': prediction,
                    'true_label': 0,
                    'date': date.strftime('%Y-%m-%d')
                })
                
                day_detail['predictions'].append(pred_detail)
                
            except Exception as e:
                print(f"Error predicting {neg_road}: {e}")
                continue
        
        detailed_predictions.append(day_detail)
    
    # Calculate actual results
    tp = sum(1 for r in results if r['type'] == 'Positive' and r['prediction'] == 1)
    fp = sum(1 for r in results if r['type'] == 'Negative' and r['prediction'] == 1)
    fn = sum(1 for r in results if r['type'] == 'Positive' and r['prediction'] == 0)
    tn = sum(1 for r in results if r['type'] == 'Negative' and r['prediction'] == 0)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    positive_samples = sum(1 for r in results if r['type'] == 'Positive')
    negative_samples = sum(1 for r in results if r['type'] == 'Negative')
    total_samples = len(results)
    
    # Results
    actual_results = {
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'positive_samples': positive_samples,
        'negative_samples': negative_samples,
        'total_samples': total_samples
    }
    
    print(f"\nActual Results:")
    print(f"  TP: {actual_results['tp']}, FP: {actual_results['fp']}, TN: {actual_results['tn']}, FN: {actual_results['fn']}")
    print(f"  Precision: {actual_results['precision']:.6f}")
    print(f"  Recall: {actual_results['recall']:.6f}")
    print(f"  F1-Score: {actual_results['f1_score']:.6f}")
    print(f"  Samples: {actual_results['positive_samples']} pos, {actual_results['negative_samples']} neg, {actual_results['total_samples']} total")
    
    # Verification
    print(f"\n{'='*60}")
    print("VERIFICATION RESULTS")
    print(f"{'='*60}")
    
    verification_passed = True
    tolerance = 1e-6
    
    for key in ['tp', 'fp', 'tn', 'fn', 'positive_samples', 'negative_samples', 'total_samples']:
        if actual_results[key] == expected_results[key]:
            print(f"✅ {key.upper()}: {actual_results[key]} (MATCH)")
        else:
            print(f"❌ {key.upper()}: Expected {expected_results[key]}, Got {actual_results[key]} (MISMATCH)")
            verification_passed = False
    
    for key in ['precision', 'recall']:
        diff = abs(actual_results[key] - expected_results[key])
        if diff < tolerance:
            print(f"✅ {key.upper()}: {actual_results[key]:.6f} (MATCH within tolerance)")
        else:
            print(f"❌ {key.upper()}: Expected {expected_results[key]:.6f}, Got {actual_results[key]:.6f} (DIFF: {diff:.6f})")
            verification_passed = False
    
    if verification_passed:
        print(f"\n🎉 OVERALL VERIFICATION: PASSED")
    else:
        print(f"\n⚠️ OVERALL VERIFICATION: FAILED")
    
    return actual_results, detailed_predictions

def print_detailed_test_data(detailed_predictions):
    """Print detailed test data for each evaluation day"""
    print(f"\n{'='*80}")
    print("DETAILED TEST DATA (Day-by-Day Analysis)")
    print(f"{'='*80}")
    
    for i, day in enumerate(detailed_predictions, 1):
        print(f"\nDay {i}: {day['date']}")
        print(f"  All flooded roads that day: {len(day['all_flooded_roads'])} roads")
        print(f"    {day['all_flooded_roads']}")
        print(f"  Flooded roads in Bayesian network: {len(day['flooded_in_bn'])} roads")
        print(f"    {day['flooded_in_bn']}")
        print(f"  Evidence roads (50%): {len(day['evidence_roads'])} roads")
        print(f"    {day['evidence_roads']}")
        print(f"  Target roads:")
        print(f"    Positive targets (actual floods): {len(day['positive_targets'])} roads")
        print(f"      {day['positive_targets']}")
        print(f"    Negative targets (selected candidates): {len(day['negative_targets'])} roads")
        print(f"      {day['negative_targets']}")
        
        print(f"\n  Predictions:")
        for j, pred in enumerate(day['predictions'], 1):
            status = "✅ CORRECT" if pred['is_correct'] else "❌ INCORRECT"
            pred_text = "FLOOD" if pred['prediction'] == 1 else "NO FLOOD"
            true_text = "FLOOD" if pred['true_label'] == 1 else "NO FLOOD"
            
            print(f"    {j:2d}. {pred['target_road']:<20} ({pred['type']:<8}) | "
                  f"Prob: {pred['prob_flood']:.4f} → {pred_text:<8} | "
                  f"Actual: {true_text:<8} | {status}")
        
        # Summary for this day
        day_correct = sum(1 for pred in day['predictions'] if pred['is_correct'])
        day_total = len(day['predictions'])
        day_accuracy = day_correct / day_total if day_total > 0 else 0.0
        print(f"  Day accuracy: {day_correct}/{day_total} = {day_accuracy:.3f}")

def visualize_bayesian_network(flood_net, marginals_dict, negative_candidates):
    """Create visualization of the Bayesian network"""
    print(f"\n{'='*60}")
    print("BAYESIAN NETWORK VISUALIZATION")
    print(f"{'='*60}")
    
    try:
        # Create network graph
        G = nx.DiGraph()
        
        # Add nodes
        for node in flood_net.network_bayes.nodes():
            G.add_node(node)
        
        # Add edges
        for edge in flood_net.network_bayes.edges():
            G.add_edge(edge[0], edge[1])
        
        # Create layout
        plt.figure(figsize=(20, 16))
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        
        # Node colors based on marginal probability
        node_colors = []
        node_sizes = []
        for node in G.nodes():
            prob = marginals_dict.get(node, 0.0)
            if node in negative_candidates:
                node_colors.append('lightcoral')  # Red for negative candidates
                node_sizes.append(800)
            elif prob >= 0.3:
                node_colors.append('darkred')  # High probability
                node_sizes.append(1000)
            elif prob >= 0.2:
                node_colors.append('orange')  # Medium-high probability
                node_sizes.append(800)
            elif prob >= 0.1:
                node_colors.append('yellow')  # Medium probability
                node_sizes.append(600)
            else:
                node_colors.append('lightblue')  # Low probability
                node_sizes.append(400)
        
        # Draw network
        nx.draw(G, pos, 
                node_color=node_colors,
                node_size=node_sizes,
                with_labels=True,
                font_size=8,
                font_weight='bold',
                arrows=True,
                arrowsize=20,
                edge_color='gray',
                alpha=0.8)
        
        plt.title("Charleston Flood Bayesian Network\n" +
                 "Red: Negative Candidates | Dark Red: High Prob (≥0.3) | Orange: Med-High (≥0.2)\n" +
                 "Yellow: Medium (≥0.1) | Light Blue: Low (<0.1)", 
                 fontsize=14, fontweight='bold')
        
        # Add legend
        legend_elements = [
            plt.scatter([], [], c='lightcoral', s=100, label='Negative Candidates'),
            plt.scatter([], [], c='darkred', s=100, label='High Prob (≥0.3)'),
            plt.scatter([], [], c='orange', s=100, label='Med-High Prob (≥0.2)'),
            plt.scatter([], [], c='yellow', s=100, label='Medium Prob (≥0.1)'),
            plt.scatter([], [], c='lightblue', s=100, label='Low Prob (<0.1)')
        ]
        plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        plt.tight_layout()
        plt.savefig('bayesian_network_visualization.png', dpi=300, bbox_inches='tight')
        print(f"Network visualization saved as 'bayesian_network_visualization.png'")
        
        # Print network structure summary
        print(f"\nNetwork Structure Summary:")
        print(f"  Nodes: {len(G.nodes())}")
        print(f"  Edges: {len(G.edges())}")
        print(f"  Density: {nx.density(G):.4f}")
        print(f"  Is connected: {nx.is_weakly_connected(G)}")
        if nx.is_weakly_connected(G):
            print(f"  Average shortest path length: {nx.average_shortest_path_length(G.to_undirected()):.2f}")
        
        # Most connected nodes
        in_degrees = dict(G.in_degree())
        out_degrees = dict(G.out_degree())
        total_degrees = {node: in_degrees[node] + out_degrees[node] for node in G.nodes()}
        
        most_connected = sorted(total_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\nMost connected nodes:")
        for node, degree in most_connected:
            prob = marginals_dict.get(node, 0.0)
            print(f"  {node}: {degree} connections (prob: {prob:.4f})")
        
    except Exception as e:
        print(f"Error creating visualization: {e}")
        print("Continuing without visualization...")

def main():
    """Main verification function"""
    # Load data
    train_df, test_df = load_and_preprocess_data()
    
    # Build and analyze network
    flood_net, bn_nodes, marginals_dict = build_and_analyze_network(train_df)
    
    # Analyze negative candidates
    negative_candidates, road_stats = analyze_negative_candidates(train_df, bn_nodes, marginals_dict)
    
    # Verify configuration results
    actual_results, detailed_predictions = verify_configuration_results(
        flood_net, test_df, bn_nodes, negative_candidates
    )
    
    # Print detailed test data
    print_detailed_test_data(detailed_predictions)
    
    # Visualize network
    visualize_bayesian_network(flood_net, marginals_dict, negative_candidates)
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save day-by-day analysis
    day_analysis = []
    for day in detailed_predictions:
        for pred in day['predictions']:
            day_analysis.append({
                'date': day['date'],
                'evidence_roads': ','.join(day['evidence_roads']),
                'target_road': pred['target_road'],
                'target_type': pred['type'],
                'prob_flood': pred['prob_flood'],
                'prediction': pred['prediction'],
                'true_label': pred['true_label'],
                'is_correct': pred['is_correct']
            })
    
    df_analysis = pd.DataFrame(day_analysis)
    analysis_file = f"verification_detailed_analysis_{timestamp}.csv"
    df_analysis.to_csv(analysis_file, index=False)
    
    print(f"\n{'='*80}")
    print("VERIFICATION COMPLETE")
    print(f"{'='*80}")
    print(f"Detailed analysis saved to: {analysis_file}")
    print(f"Network visualization saved to: bayesian_network_visualization.png")
    
    return actual_results, detailed_predictions

if __name__ == "__main__":
    results, predictions = main()