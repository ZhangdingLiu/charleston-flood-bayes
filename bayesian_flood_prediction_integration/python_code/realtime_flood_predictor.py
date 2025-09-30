#!/usr/bin/env python3
"""
Real-Time Cumulative Flood Prediction System - Integration Package
- Train on Road_Closures_2024.csv (2015-2024 historical data) to build 88-node Bayesian Network
- Test on 2025 flood data with cumulative evidence windows (10-minute intervals)
- Each time window uses all previous evidence to predict all 88 network nodes
- Simulates real-time flood prediction scenario with accumulating evidence

This is the integration-ready version for full-stack systems.
"""

import json
import os
import sys
import random
import csv
from datetime import datetime, timedelta
from collections import defaultdict, Counter

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

class RealTimeFloodPredictor:
    """Real-time flood prediction with cumulative evidence"""
    
    def __init__(self, data_dir=None):
        # Use aggressive strategy parameters (88-121 nodes)
        self.network_params = {
            'name': 'Aggressive Strategy (88 Nodes)',
            'occ_thr': 1,      # Include roads appearing >= 1 time
            'edge_thr': 1,     # Create edge if co-occurrence >= 1
            'weight_thr': 0.05, # Include edges with conditional probability >= 0.05
        }
        
        # Data containers
        self.training_data = []
        self.test_data = {}
        
        # Set data directory (for integration flexibility)
        if data_dir:
            self.data_dir = data_dir
        else:
            # Try to find data directory relative to current location
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.data_dir = os.path.join(os.path.dirname(current_dir), 'data')
        
    def load_training_data(self, file_path=None):
        """Load historical training data from Road_Closures_2024.csv"""
        print("📚 Loading training data from Road_Closures_2024.csv...")
        
        if file_path:
            training_paths = [file_path]
        else:
            # Try different path locations for integration flexibility
            training_paths = [
                os.path.join(self.data_dir, "Road_Closures_2024.csv"),
                "Road_Closures_2024.csv",
                "data/Road_Closures_2024.csv",
                "../data/Road_Closures_2024.csv"
            ]
        
        training_file = None
        for path in training_paths:
            if os.path.exists(path):
                training_file = path
                break
                
        if not training_file:
            print(f"❌ Road_Closures_2024.csv not found in expected locations: {training_paths}")
            return False
        
        print(f"   📍 Loading from: {training_file}")
        
        with open(training_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('REASON') and 'flood' in row['REASON'].lower():
                    street = row.get('STREET', '').replace(' ', '_').upper()
                    date = row.get('created_date', row.get('time_create', row.get('DATE', '')))
                    if street and date:
                        self.training_data.append({
                            'street': street,
                            'date': date,
                            'reason': row.get('REASON', '')
                        })
        
        unique_dates = len(set(r['date'] for r in self.training_data))
        unique_streets = len(set(r['street'] for r in self.training_data))
        
        print(f"✅ Loaded {len(self.training_data)} flood records")
        print(f"📊 Training data: {unique_dates} dates, {unique_streets} unique streets")
        
        return len(self.training_data) > 0
        
    def parse_time(self, time_str):
        """Parse time string like 'Fri, Aug 22, 2025 12:19 PM' to datetime"""
        try:
            return datetime.strptime(time_str, "%a, %b %d, %Y %I:%M %p")
        except ValueError:
            return None
    
    def load_test_data_from_csv(self, file_path=None):
        """Load flood test data from CSV file"""
        print("🌊 Loading flood test data...")
        
        if file_path:
            data_paths = [file_path]
        else:
            # Try different path locations
            data_paths = [
                os.path.join(self.data_dir, "2025_flood_processed.csv"),
                "2025_flood_processed.csv",
                "data/2025_flood_processed.csv",
                "../data/2025_flood_processed.csv"
            ]
        
        data_file = None
        for path in data_paths:
            if os.path.exists(path):
                data_file = path
                break
                
        if not data_file:
            print(f"❌ Test data CSV not found in expected locations: {data_paths}")
            return False
        
        print(f"   📍 Loading from: {data_file}")
        
        all_records = []
        with open(data_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                parsed_time = self.parse_time(row['start_time'])
                if parsed_time:
                    row['parsed_time'] = parsed_time
                    all_records.append(row)
        
        if not all_records:
            print("❌ No valid test records found")
            return False
        
        # Sort by time
        all_records.sort(key=lambda x: x['parsed_time'])
        
        print(f"✅ Loaded test data:")
        print(f"   Total records: {len(all_records)}")
        print(f"   Time range: {all_records[0]['start_time']} to {all_records[-1]['start_time']}")
        print(f"   Unique streets: {len(set(r['street'] for r in all_records))}")
        
        self.test_data = {
            'records': all_records,
            'start_time': all_records[0]['parsed_time'],
            'end_time': all_records[-1]['parsed_time'],
            'all_streets': list(set(r['street'] for r in all_records))
        }
        
        return True
    
    def load_test_data_from_events(self, flood_events):
        """Load flood test data from list of events (for API integration)"""
        print(f"🌊 Loading {len(flood_events)} flood events...")
        
        all_records = []
        for event in flood_events:
            # Expected format: {'street': 'KING_ST', 'start_time': 'datetime_obj', 'reason': 'FLOOD'}
            if isinstance(event.get('start_time'), str):
                parsed_time = self.parse_time(event['start_time'])
            else:
                parsed_time = event.get('start_time')
            
            if parsed_time:
                record = {
                    'street': event['street'].replace(' ', '_').upper(),
                    'start_time': event['start_time'] if isinstance(event['start_time'], str) else event['start_time'].strftime("%a, %b %d, %Y %I:%M %p"),
                    'reason': event.get('reason', 'FLOOD'),
                    'parsed_time': parsed_time
                }
                all_records.append(record)
        
        if not all_records:
            print("❌ No valid flood events provided")
            return False
        
        # Sort by time
        all_records.sort(key=lambda x: x['parsed_time'])
        
        print(f"✅ Processed flood events:")
        print(f"   Total events: {len(all_records)}")
        print(f"   Time range: {all_records[0]['start_time']} to {all_records[-1]['start_time']}")
        print(f"   Unique streets: {len(set(r['street'] for r in all_records))}")
        
        self.test_data = {
            'records': all_records,
            'start_time': all_records[0]['parsed_time'],
            'end_time': all_records[-1]['parsed_time'],
            'all_streets': list(set(r['street'] for r in all_records))
        }
        
        return True
        
    def get_network_nodes(self):
        """Get list of all Bayesian network nodes (for API/frontend)"""
        if not self.training_data:
            return []
        
        road_freq = Counter(r['street'] for r in self.training_data)
        network_roads = [road for road, freq in road_freq.items() 
                       if freq >= self.network_params['occ_thr']]
        
        return sorted(network_roads)
    
    def create_time_windows(self, window_minutes=10):
        """Create cumulative time windows from test data"""
        if not self.test_data:
            return []
            
        records = self.test_data['records']
        start_time = self.test_data['start_time']
        end_time = self.test_data['end_time']
        
        windows = []
        current_time = start_time
        
        while current_time < end_time:
            window_end = current_time + timedelta(minutes=window_minutes)
            
            # Get all records from start to current window end (cumulative)
            cumulative_records = [r for r in records if r['parsed_time'] <= window_end]
            evidence_streets = list(set(r['street'] for r in cumulative_records))
            
            window = {
                'window_start': current_time,
                'window_end': window_end, 
                'cumulative_records': cumulative_records,
                'evidence_streets': evidence_streets,
                'window_label': f"{current_time.strftime('%I:%M')}-{window_end.strftime('%I:%M %p')}"
            }
            windows.append(window)
            
            current_time = window_end
            
        return windows
    
    def build_bayesian_network(self):
        """Build Bayesian network from training data"""
        print(f"\n🏗️ Building Bayesian network with parameters...")
        print(f"   Parameters: occ_thr={self.network_params['occ_thr']}, edge_thr={self.network_params['edge_thr']}, weight_thr={self.network_params['weight_thr']}")
        
        try:
            # Apply occurrence threshold filter
            road_freq = Counter(r['street'] for r in self.training_data)
            network_roads = [road for road, freq in road_freq.items() 
                           if freq >= self.network_params['occ_thr']]
            
            if len(network_roads) < 3:
                print(f"❌ Network roads insufficient (<3), found {len(network_roads)}")
                return None, False
                
            print(f"✅ Network nodes: {len(network_roads)} (after occ_thr={self.network_params['occ_thr']} filter)")
            
            # Build co-occurrence matrix
            print("   Building co-occurrence relationships...")
            cooccurrence = defaultdict(lambda: defaultdict(int))
            
            # Group records by date
            date_groups = defaultdict(list)
            for record in self.training_data:
                date_groups[record['date']].append(record['street'])
            
            # Calculate co-occurrences
            edge_count = 0
            for date, streets in date_groups.items():
                unique_streets = list(set(streets))
                for i, street1 in enumerate(unique_streets):
                    for street2 in unique_streets[i+1:]:
                        if street1 in network_roads and street2 in network_roads:
                            cooccurrence[street1][street2] += 1
                            cooccurrence[street2][street1] += 1
                            edge_count += 1
            
            print(f"   Network edges: {edge_count} total co-occurrences")
            
            # Create enhanced network
            class EnhancedBayesianNetwork:
                def __init__(self, nodes, training_data, params):
                    self.nodes = set(nodes)
                    self.road_freq = Counter(r['street'] for r in training_data)
                    self.cooccurrence = cooccurrence
                    self.params = params
                    
                def number_of_nodes(self):
                    return len(self.nodes)
                    
                def number_of_edges(self):
                    edges = 0
                    for road1 in self.nodes:
                        for road2 in self.nodes:
                            if road1 != road2 and self.cooccurrence[road1][road2] >= self.params['edge_thr']:
                                edges += 1
                    return edges // 2  # Undirected edges
                    
                def infer_w_evidence(self, road, evidence):
                    """Enhanced inference with evidence"""
                    if road not in self.nodes:
                        return {'flooded': 0.1}  # Low probability for unknown roads
                    
                    if road in evidence:
                        return {'flooded': 1.0}  # Evidence roads are certain
                    
                    # Base probability from historical frequency
                    base_prob = min(0.8, self.road_freq.get(road, 1) / 100.0)
                    
                    # Evidence boost calculation
                    evidence_boost = 0.0
                    evidence_count = 0
                    
                    for ev_road, ev_value in evidence.items():
                        if ev_value == 1 and ev_road in self.nodes and ev_road != road:
                            cooc_count = self.cooccurrence[ev_road][road]
                            ev_freq = self.road_freq.get(ev_road, 0)
                            
                            if ev_freq > 0 and cooc_count >= self.params['edge_thr']:
                                cond_prob = cooc_count / ev_freq
                                if cond_prob >= self.params['weight_thr']:
                                    evidence_boost += cond_prob * 0.5
                                    evidence_count += 1
                    
                    # Final probability calculation
                    if evidence_count > 0:
                        evidence_avg = evidence_boost / evidence_count
                        final_prob = min(1.0, base_prob * 0.3 + evidence_avg * 0.7)
                    else:
                        final_prob = base_prob * 0.5
                    
                    return {'flooded': final_prob}
            
            enhanced_network = EnhancedBayesianNetwork(network_roads, self.training_data, self.network_params)
            
            # Create wrapper for compatibility
            class NetworkWrapper:
                def __init__(self, enhanced_net):
                    self.network = enhanced_net
                    
                def infer_w_evidence(self, road, evidence):
                    return self.network.infer_w_evidence(road, evidence)
            
            flood_net = NetworkWrapper(enhanced_network)
            
            print(f"   Network statistics: {enhanced_network.number_of_nodes()} nodes, {enhanced_network.number_of_edges()} edges")
            
            return flood_net, True
            
        except Exception as e:
            print(f"❌ Network construction failed: {str(e)}")
            return None, False

    def predict_single_window(self, evidence_roads, output_dir=None, window_label=None):
        """Make predictions for a single time window (for API integration)"""
        if not self.training_data:
            print("❌ No training data loaded")
            return None
        
        # Build network if not already done
        flood_net, success = self.build_bayesian_network()
        if not success:
            return None
        
        network_roads = list(flood_net.network.nodes)
        
        # Set up evidence
        evidence = {}
        for road in evidence_roads:
            road_standardized = road.replace(' ', '_').upper()
            if road_standardized in network_roads:
                evidence[road_standardized] = 1
        
        if not evidence:
            print("⚠️ No evidence roads found in network")
            return None
        
        # Make predictions
        predictions = []
        for road in sorted(network_roads):
            try:
                if road in evidence:
                    prob = 1.000  # Evidence roads have probability 1
                else:
                    result = flood_net.infer_w_evidence(road, evidence)
                    prob = result.get('flooded', result.get(1, 0.0))
                
                predictions.append({
                    'road': road, 
                    'probability': prob, 
                    'is_evidence': road in evidence
                })
                
            except Exception as e:
                predictions.append({
                    'road': road, 
                    'probability': None, 
                    'is_evidence': False, 
                    'error': str(e)
                })
        
        # Calculate summary statistics
        non_evidence_preds = [p['probability'] for p in predictions if not p['is_evidence'] and p['probability'] is not None]
        if non_evidence_preds:
            avg_pred = sum(non_evidence_preds) / len(non_evidence_preds)
            high_risk_count = len([p for p in non_evidence_preds if p > 0.5])
        else:
            avg_pred = 0.0
            high_risk_count = 0
        
        # Prepare result
        result = {
            "experiment_metadata": {
                "experiment_name": "Real-Time Flood Prediction",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "description": f"Single window prediction using {len(evidence_roads)} evidence roads",
                "random_seed": RANDOM_SEED
            },
            "bayesian_network": {
                "parameters": self.network_params,
                "statistics": {
                    "total_nodes": flood_net.network.number_of_nodes(),
                    "total_edges": flood_net.network.number_of_edges()
                },
                "all_nodes": sorted(network_roads)
            },
            "current_window": {
                "window_label": window_label or f"Single_Window_{datetime.now().strftime('%H%M')}",
                "evidence": {
                    "evidence_roads": sorted(evidence_roads),
                    "network_evidence_roads": sorted(evidence.keys()),
                    "evidence_count": len(evidence_roads),
                    "network_evidence_count": len(evidence)
                },
                "predictions": predictions,
                "summary_stats": {
                    "average_prediction_probability": round(avg_pred, 3),
                    "high_risk_roads_count": high_risk_count,
                    "total_non_evidence_roads": len(non_evidence_preds)
                }
            }
        }
        
        # Save to file if output directory specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"flood_prediction_{window_label or 'single'}_{timestamp}.json"
            filepath = os.path.join(output_dir, filename)
            
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print(f"💾 Prediction saved to: {filepath}")
            except Exception as e:
                print(f"❌ Failed to save prediction: {str(e)}")
        
        return result

def run_full_prediction_pipeline(data_dir=None, test_data_path=None, output_dir="results", window_minutes=10):
    """Complete prediction pipeline (for standalone usage)"""
    print("🌊 Real-Time Cumulative Flood Prediction System - Integration Package")
    print("🎯 Using Bayesian network trained on 2015-2024 data")
    print("⏱️  Testing with cumulative time windows")
    print("="*80)
    
    predictor = RealTimeFloodPredictor(data_dir=data_dir)
    
    # Step 1: Load training data
    print("\n📚 Loading training data...")
    if not predictor.load_training_data():
        print("❌ Failed to load training data")
        return None
    
    # Step 2: Load test data
    print("\n🌊 Loading test data...")  
    if not predictor.load_test_data_from_csv(test_data_path):
        print("❌ Failed to load test data")
        return None
    
    # Step 3: Build Bayesian network
    print("\n🏗️ Building Bayesian network...")
    flood_net, success = predictor.build_bayesian_network()
    if not success:
        print("❌ Failed to build Bayesian network")
        return None
    
    network_roads = list(flood_net.network.nodes)
    print(f"\n🧠 BAYESIAN NETWORK: {len(network_roads)} nodes, {flood_net.network.number_of_edges()} edges")
    
    # Step 4: Create time windows and predict
    print(f"\n⏱️ Creating {window_minutes}-minute cumulative time windows...")
    windows = predictor.create_time_windows(window_minutes=window_minutes)
    print(f"✅ Created {len(windows)} time windows")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("🔮 PROCESSING TIME WINDOWS")
    print(f"{'='*80}")
    
    generated_files = []
    
    for i, window in enumerate(windows, 1):
        print(f"\n📅 Window {i}: {window['window_label']}")
        print(f"   🔑 Evidence Roads ({len(window['evidence_streets'])}): {', '.join(sorted(window['evidence_streets'])[:5])}{'...' if len(window['evidence_streets']) > 5 else ''}")
        
        # Use the single window prediction method
        result = predictor.predict_single_window(
            evidence_roads=window['evidence_streets'],
            output_dir=output_dir,
            window_label=f"window_{i:02d}_{window['window_label'].replace(':', '').replace('-', '_').replace(' ', '_')}"
        )
        
        if result:
            avg_prob = result['current_window']['summary_stats']['average_prediction_probability']
            high_risk = result['current_window']['summary_stats']['high_risk_roads_count']
            print(f"   📊 Avg Prob: {avg_prob:.3f}, High Risk: {high_risk}")
            generated_files.append(result)
        
    print(f"\n🎉 Prediction pipeline completed!")
    print(f"📊 Processed {len(windows)} time windows")
    print(f"💾 Generated {len(generated_files)} prediction files")
    print(f"📁 Files saved in: {output_dir}/")
    
    return generated_files

# For standalone usage
if __name__ == "__main__":
    # Example usage
    run_full_prediction_pipeline()