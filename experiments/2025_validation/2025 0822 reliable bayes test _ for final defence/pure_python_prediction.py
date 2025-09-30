#!/usr/bin/env python3
"""
Pure Python version - no pandas dependency
Uses simplified but reliable Bayesian logic to generate 9 JSON outputs
"""

import json
import os
import csv
from datetime import datetime, timedelta
from collections import defaultdict, Counter

# Set random seed
RANDOM_SEED = 42

class SimpleBayesianNetwork:
    def __init__(self, occ_thr=10, edge_thr=3, weight_thr=0.4):
        self.occ_thr = occ_thr
        self.edge_thr = edge_thr
        self.weight_thr = weight_thr
        self.nodes = set()
        self.road_freq = {}
        self.cooccurrence = defaultdict(lambda: defaultdict(int))
        self.conditional_probs = {}
        
    def fit(self, flood_records):
        # Count road frequencies
        road_counts = Counter(r['street'] for r in flood_records)
        self.road_freq = dict(road_counts)
        
        # Filter roads by occurrence threshold
        self.nodes = {road for road, count in road_counts.items() if count >= self.occ_thr}
        print(f"Network nodes after filtering: {len(self.nodes)}")
        
        # Build co-occurrence matrix
        date_groups = defaultdict(list)
        for record in flood_records:
            date_groups[record['date']].append(record['street'])
        
        for date, streets in date_groups.items():
            unique_streets = list(set(streets))
            for i, street1 in enumerate(unique_streets):
                for street2 in unique_streets[i+1:]:
                    if street1 in self.nodes and street2 in self.nodes:
                        self.cooccurrence[street1][street2] += 1
                        self.cooccurrence[street2][street1] += 1
        
        # Build conditional probabilities
        for road in self.nodes:
            self.conditional_probs[road] = {}
            for parent in self.nodes:
                if parent != road:
                    cooc_count = self.cooccurrence[parent][road]
                    parent_freq = self.road_freq[parent]
                    if cooc_count >= self.edge_thr and parent_freq > 0:
                        cond_prob = cooc_count / parent_freq
                        if cond_prob >= self.weight_thr:
                            self.conditional_probs[road][parent] = cond_prob
        
        print(f"Network built: {len(self.nodes)} nodes")
        return True
        
    def infer_w_evidence(self, target_road, evidence):
        if target_road not in self.nodes:
            return {'flooded': 0.01}
            
        if target_road in evidence and evidence[target_road] == 1:
            return {'flooded': 1.0}
        
        # Base probability from frequency
        total_days = sum(len(records) for records in self.cooccurrence.values()) // len(self.nodes) if self.nodes else 1
        base_prob = min(0.8, self.road_freq.get(target_road, 1) / max(total_days, 100))
        
        # Evidence influence
        evidence_boost = 0.0
        evidence_count = 0
        
        for ev_road, ev_value in evidence.items():
            if ev_value == 1 and ev_road in self.conditional_probs[target_road]:
                evidence_boost += self.conditional_probs[target_road][ev_road]
                evidence_count += 1
        
        if evidence_count > 0:
            final_prob = min(1.0, base_prob * 0.3 + (evidence_boost / evidence_count) * 0.7)
        else:
            final_prob = base_prob * 0.5
            
        return {'flooded': final_prob}

def load_training_data():
    """Load training data"""
    paths = ["src/models/Road_Closures_2024.csv", "../../src/models/Road_Closures_2024.csv"]
    
    training_file = None
    for path in paths:
        if os.path.exists(path):
            training_file = path
            break
    
    if not training_file:
        raise FileNotFoundError("Road_Closures_2024.csv not found")
    
    flood_records = []
    with open(training_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('REASON', '').upper() == 'FLOOD':
                try:
                    # Handle format: "2015/08/24 04:00:00+00" 
                    start_str = row['START'].split('+')[0]  # Remove timezone
                    start_date = datetime.strptime(start_str, '%Y/%m/%d %H:%M:%S')
                    street = row['STREET'].upper().replace(' ', '_')
                    flood_records.append({
                        'street': street,
                        'date': start_date.strftime('%Y-%m-%d'),
                        'datetime': start_date
                    })
                except Exception as e:
                    continue
    
    print(f"Loaded {len(flood_records)} flood records")
    return flood_records

def load_2025_test_data():
    """Load 2025 test data"""
    paths = ['archive/old_results/2025_flood_processed.csv', '../../archive/old_results/2025_flood_processed.csv']
    
    test_file = None
    for path in paths:
        if os.path.exists(path):
            test_file = path
            break
    
    if not test_file:
        raise FileNotFoundError("2025_flood_processed.csv not found")
    
    test_records = []
    with open(test_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                timestamp = datetime.strptime(row['start_time'], "%a, %b %d, %Y %I:%M %p")
                test_records.append({
                    'street': row['street'],
                    'timestamp': timestamp,
                    'split_type': row['split_type']
                })
            except:
                continue
    
    test_records.sort(key=lambda x: x['timestamp'])
    print(f"Loaded {len(test_records)} 2025 test records")
    return test_records

def create_time_windows(test_records, window_minutes=10):
    """Create 10-minute time windows"""
    if not test_records:
        return []
    
    start_time = min(r['timestamp'] for r in test_records)
    end_time = max(r['timestamp'] for r in test_records)
    
    windows = []
    current_time = start_time
    window_id = 1
    
    while current_time < end_time:
        window_end = current_time + timedelta(minutes=window_minutes)
        
        window_records = [r for r in test_records if current_time <= r['timestamp'] < window_end]
        
        if window_records:
            windows.append({
                'window_id': window_id,
                'window_start': current_time,
                'window_end': window_end,
                'records': window_records
            })
            window_id += 1
        
        current_time = window_end
    
    print(f"Created {len(windows)} time windows")
    return windows

def run_predictions(network, time_windows):
    """Run predictions for all windows"""
    cumulative_evidence_roads = set()
    results = []
    
    for window in time_windows:
        # Add evidence from current window
        new_evidence = {r['street'] for r in window['records']}
        cumulative_evidence_roads.update(new_evidence)
        
        # Filter to network nodes
        network_evidence = [road for road in cumulative_evidence_roads if road in network.nodes]
        evidence = {road: 1 for road in network_evidence}
        
        # Predict all network nodes
        predictions = []
        for node in sorted(network.nodes):
            result = network.infer_w_evidence(node, evidence)
            predictions.append({
                'road': node,
                'probability': result['flooded'],
                'is_evidence': node in network_evidence
            })
        
        # Calculate summary stats
        non_evidence = [p for p in predictions if not p['is_evidence']]
        avg_prob = sum(p['probability'] for p in non_evidence) / len(non_evidence) if non_evidence else 0
        high_risk = sum(1 for p in non_evidence if p['probability'] > 0.5)
        
        window_result = {
            'window_id': window['window_id'],
            'window_label': f"{window['window_start'].strftime('%H:%M')}-{window['window_end'].strftime('%H:%M')}",
            'time_range': {
                'window_start': window['window_start'].isoformat(),
                'window_end': window['window_end'].isoformat()
            },
            'evidence': {
                'cumulative_evidence_roads': sorted(list(cumulative_evidence_roads)),
                'network_evidence_roads': sorted(network_evidence),
                'evidence_count': len(cumulative_evidence_roads),
                'network_evidence_count': len(network_evidence)
            },
            'predictions': predictions,
            'summary_stats': {
                'average_prediction_probability': round(avg_prob, 3),
                'high_risk_roads_count': high_risk,
                'total_non_evidence_roads': len(non_evidence)
            }
        }
        
        results.append(window_result)
        print(f"Window {window['window_id']}: {len(network_evidence)} evidence, avg prob {avg_prob:.3f}")
    
    return results

def save_results(window_results, network, training_count):
    """Save results as JSON files"""
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for i, window_result in enumerate(window_results):
        # Create complete result structure
        full_result = {
            'experiment_metadata': {
                'experiment_name': 'Real-Time Cumulative Flood Prediction (Pure Python)',
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'description': f'Using {len(network.nodes)}-node reliable network, testing 10-minute windows on 2025 data',
                'random_seed': RANDOM_SEED
            },
            'training_data_info': {
                'total_records': training_count,
                'unique_streets': len(network.nodes),
                'data_source': 'Road_Closures_2024.csv'
            },
            'test_data_info': {
                'data_source': 'archive/old_results/2025_flood_processed.csv'
            },
            'bayesian_network': {
                'parameters': {
                    'occ_thr': network.occ_thr,
                    'edge_thr': network.edge_thr,
                    'weight_thr': network.weight_thr
                },
                'statistics': {
                    'total_nodes': len(network.nodes),
                    'total_edges': sum(len(probs) for probs in network.conditional_probs.values())
                },
                'all_nodes': sorted(list(network.nodes))
            },
            'current_window': window_result,
            'window_summary': {
                'window_number': i + 1,
                'total_windows': len(window_results),
                'window_label': window_result['window_label'],
                'completion_status': 'success'
            }
        }
        
        # Save file
        filename = f"realtime_window_{i+1:02d}_{window_result['window_label'].replace(':', '').replace('-', '_')}_PM_{timestamp}.json"
        filepath = os.path.join(results_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(full_result, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved {len(window_results)} JSON files to {results_dir}/")
    return results_dir

def main():
    """Main function"""
    print("🌊 Pure Python Real-Time Flood Prediction")
    print("🎯 Reliable Bayesian Network (No Pandas Dependency)")
    print("="*60)
    
    try:
        # Load data
        print("\n📚 Loading training data...")
        flood_records = load_training_data()
        
        print("\n🌊 Loading 2025 test data...")
        test_records = load_2025_test_data()
        
        # Build network
        print(f"\n🧠 Building Bayesian network (occ_thr=5, edge_thr=3, weight_thr=0.4)...")
        network = SimpleBayesianNetwork(occ_thr=5, edge_thr=3, weight_thr=0.4)
        network.fit(flood_records)
        
        # Create time windows
        print(f"\n⏱️ Creating 10-minute time windows...")
        time_windows = create_time_windows(test_records)
        
        # Run predictions
        print(f"\n🔮 Running cumulative predictions...")
        window_results = run_predictions(network, time_windows)
        
        # Save results
        print(f"\n💾 Saving JSON results...")
        results_dir = save_results(window_results, network, len(flood_records))
        
        print(f"\n🎉 SUCCESS!")
        print(f"Generated {len(window_results)} JSON files")
        print(f"Results saved to: {results_dir}/")
        print(f"Network: {len(network.nodes)} nodes")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()