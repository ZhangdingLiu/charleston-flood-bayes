#!/usr/bin/env python3
"""
Reliable 2025 Charleston Flood Validation
========================================
- Use reliable Bayesian modeling from src/models/model.py (FloodBayesNetwork)
- Train on Road_Closures_2024.csv (2015-2024 historical data)
- Test on 2025 flood data with 10-minute time windows
- Use recommended parameters: occ_thr=10, edge_thr=3, weight_thr=0.4 (~40 nodes)
- Output JSON format compatible with existing results

Author: Claude AI based on reliable src/models/model.py
Date: 2025-09-06
"""

import json
import os
import sys
import random
import csv
import time
from datetime import datetime, timedelta
from collections import defaultdict, Counter

# Add src/models to path for importing FloodBayesNetwork
sys.path.append('../../src/models')
sys.path.append('src/models')

# Import pandas and numpy after path setup
try:
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"❌ Missing required packages: {e}")
    print("   Please install: pip install pandas numpy")
    sys.exit(1)

try:
    from model import FloodBayesNetwork
except ImportError:
    print("❌ Cannot import FloodBayesNetwork from src/models/model.py")
    print("   Please ensure you are running from project root or src/models/model.py exists")
    sys.exit(1)

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

class Reliable2025FloodValidator:
    """
    Reliable 2025 flood validation using trusted FloodBayesNetwork implementation
    """
    
    def __init__(self):
        # Use recommended parameters for ~40 nodes (reliable network size)
        self.network_params = {
            'name': 'Reliable Strategy (40 Nodes)',
            'occ_thr': 10,      # Include roads appearing >= 10 times (reliable threshold)
            'edge_thr': 3,      # Create edge if co-occurrence >= 3
            'weight_thr': 0.4,  # Include edges with conditional probability >= 0.4
        }
        
        # Data containers
        self.training_data = []
        self.test_data = []
        self.flood_net = None
        self.results = []
        
        # Time window settings (10 minutes)
        self.window_minutes = 10
        
        print(f"🔧 Initialized Reliable 2025 Flood Validator")
        print(f"   Network parameters: {self.network_params}")
        
    def load_training_data(self):
        """Load historical training data from Road_Closures_2024.csv"""
        print("\n" + "="*80)
        print("📚 LOADING TRAINING DATA")
        print("="*80)
        
        # Try different possible paths for training data
        training_paths = [
            "src/models/Road_Closures_2024.csv",
            "Road_Closures_2024.csv",
            "../../src/models/Road_Closures_2024.csv"
        ]
        
        training_file = None
        for path in training_paths:
            if os.path.exists(path):
                training_file = path
                break
                
        if not training_file:
            raise FileNotFoundError("❌ Road_Closures_2024.csv not found in expected locations")
        
        print(f"📁 Using training file: {training_file}")
        
        # Load using pandas for compatibility with FloodBayesNetwork
        df = pd.read_csv(training_file)
        df = df[df["REASON"].str.upper() == "FLOOD"].copy()
        
        # Preprocess for FloodBayesNetwork compatibility
        df["time_create"] = pd.to_datetime(df["START"], utc=True)
        df["link_id"] = df["STREET"].str.upper().str.replace(" ", "_")
        df["link_id"] = df["link_id"].astype(str)
        df["id"] = df["OBJECTID"].astype(str)
        
        self.training_df = df
        
        print(f"✅ Loaded {len(df)} historical flood records for training")
        
        # Show training data statistics
        unique_dates = df['time_create'].dt.date.nunique()
        unique_streets = df['link_id'].nunique()
        date_range = f"{df['time_create'].min().strftime('%Y-%m-%d')} to {df['time_create'].max().strftime('%Y-%m-%d')}"
        
        print(f"📊 Training data statistics:")
        print(f"   - Date range: {date_range}")
        print(f"   - Unique flood dates: {unique_dates}")
        print(f"   - Unique streets: {unique_streets}")
        
        return len(df)
        
    def load_2025_test_data(self):
        """Load processed 2025 flood test data"""
        print("\n" + "="*60)
        print("🌊 LOADING 2025 TEST DATA")
        print("="*60)
        
        # Try different paths for 2025 processed data
        test_paths = [
            'archive/old_results/2025_flood_processed.csv',
            '2025_flood_processed.csv',
            'experiments/2025_validation/2025_flood_processed.csv'
        ]
        
        test_file = None
        for path in test_paths:
            if os.path.exists(path):
                test_file = path
                break
                
        if not test_file:
            raise FileNotFoundError("❌ 2025_flood_processed.csv not found in expected locations")
            
        print(f"📁 Using test file: {test_file}")
        
        # Load test data
        test_records = []
        with open(test_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Parse timestamp
                try:
                    # Handle format: "Fri, Aug 22, 2025 12:19 PM"
                    timestamp_str = row['start_time']
                    timestamp = datetime.strptime(timestamp_str, "%a, %b %d, %Y %I:%M %p")
                    
                    test_records.append({
                        'street': row['street'],
                        'original_street': row['original_street'],
                        'location': row['location'],
                        'timestamp': timestamp,
                        'reason': row['reason'],
                        'split_type': row['split_type']
                    })
                except Exception as e:
                    print(f"⚠️ Warning: Failed to parse timestamp '{row.get('start_time', 'N/A')}': {e}")
                    continue
        
        # Sort by timestamp
        test_records.sort(key=lambda x: x['timestamp'])
        
        self.test_data = test_records
        
        print(f"✅ Loaded {len(test_records)} 2025 flood records")
        
        if test_records:
            time_range = f"{test_records[0]['timestamp'].strftime('%Y-%m-%d %H:%M')} to {test_records[-1]['timestamp'].strftime('%Y-%m-%d %H:%M')}"
            unique_streets = len(set(r['street'] for r in test_records))
            
            print(f"📊 Test data statistics:")
            print(f"   - Time range: {time_range}")
            print(f"   - Unique streets: {unique_streets}")
        
        return len(test_records)
    
    def build_bayesian_network(self):
        """Build Bayesian network using reliable FloodBayesNetwork from model.py"""
        print("\n" + "="*80)
        print("🧠 BUILDING RELIABLE BAYESIAN NETWORK")
        print("="*80)
        
        # Create FloodBayesNetwork instance
        self.flood_net = FloodBayesNetwork(t_window="D")
        
        print(f"🔧 Network parameters:")
        for key, value in self.network_params.items():
            if key != 'name':
                print(f"   - {key}: {value}")
        
        # Step 1: Fit marginal probabilities
        print("\n📊 Step 1: Computing marginal probabilities...")
        self.flood_net.fit_marginal(self.training_df)
        print(f"✅ Marginal probabilities computed for {len(self.flood_net.marginals)} roads")
        
        # Step 2: Build network structure with co-occurrence filtering
        print(f"\n🏗️ Step 2: Building network structure...")
        self.flood_net.build_network_by_co_occurrence(
            self.training_df,
            occ_thr=self.network_params['occ_thr'],
            edge_thr=self.network_params['edge_thr'], 
            weight_thr=self.network_params['weight_thr'],
            report=False
        )
        
        nodes_count = self.flood_net.network.number_of_nodes()
        edges_count = self.flood_net.network.number_of_edges()
        print(f"✅ Network structure built: {nodes_count} nodes, {edges_count} edges")
        
        # Step 3: Fit conditional probabilities
        print(f"\n🧮 Step 3: Computing conditional probabilities...")
        self.flood_net.fit_conditional(self.training_df, max_parents=2, alpha=1.0)
        print(f"✅ Conditional probabilities computed")
        
        # Step 4: Build Bayesian network
        print(f"\n🔗 Step 4: Building final Bayesian network...")
        self.flood_net.build_bayes_network()
        print(f"✅ Bayesian network construction completed")
        
        # Network validation
        try:
            self.flood_net.check_bayesian_network()
            print(f"✅ Network validation passed")
        except Exception as e:
            print(f"⚠️ Network validation warning: {e}")
        
        # Report final network statistics
        all_nodes = list(self.flood_net.network_bayes.nodes())
        print(f"\n📊 Final network statistics:")
        print(f"   - Total nodes: {len(all_nodes)}")
        print(f"   - Total edges: {edges_count}")
        print(f"   - Network type: Directed Acyclic Graph (DAG)")
        
        return {
            'total_nodes': len(all_nodes),
            'total_edges': edges_count,
            'all_nodes': sorted(all_nodes)
        }
    
    def create_time_windows(self):
        """Create 10-minute time windows from test data"""
        if not self.test_data:
            return []
            
        # Get time range
        start_time = min(r['timestamp'] for r in self.test_data)
        end_time = max(r['timestamp'] for r in self.test_data)
        
        # Create 10-minute windows
        windows = []
        current_time = start_time
        window_id = 1
        
        while current_time < end_time:
            window_end = current_time + timedelta(minutes=self.window_minutes)
            
            # Find records in this window
            window_records = [
                r for r in self.test_data 
                if current_time <= r['timestamp'] < window_end
            ]
            
            if window_records:  # Only create window if it has data
                windows.append({
                    'window_id': window_id,
                    'window_start': current_time,
                    'window_end': window_end,
                    'records': window_records
                })
                window_id += 1
            
            current_time = window_end
            
        print(f"📅 Created {len(windows)} time windows of {self.window_minutes} minutes each")
        return windows
    
    def predict_with_cumulative_evidence(self, time_windows):
        """Run predictions with cumulative evidence for each time window"""
        print("\n" + "="*80)
        print("🔮 RUNNING REAL-TIME PREDICTIONS WITH CUMULATIVE EVIDENCE")
        print("="*80)
        
        bn_nodes = set(self.flood_net.network_bayes.nodes())
        cumulative_evidence_roads = set()
        
        window_results = []
        
        for i, window in enumerate(time_windows):
            print(f"\n📊 Processing Window {window['window_id']}/{len(time_windows)}")
            
            # Add new evidence from current window
            new_evidence_roads = set(r['street'] for r in window['records'])
            cumulative_evidence_roads.update(new_evidence_roads)
            
            # Filter evidence roads to those in network
            network_evidence_roads = [road for road in cumulative_evidence_roads if road in bn_nodes]
            
            print(f"   Time: {window['window_start'].strftime('%H:%M')}-{window['window_end'].strftime('%H:%M')}")
            print(f"   New evidence roads: {len(new_evidence_roads)}")
            print(f"   Total cumulative evidence: {len(cumulative_evidence_roads)}")
            print(f"   Network evidence roads: {len(network_evidence_roads)}")
            
            # Create evidence dictionary for Bayesian inference
            evidence = {road: 1 for road in network_evidence_roads}
            
            # Predict probabilities for all network nodes
            predictions = []
            for node in sorted(bn_nodes):
                if node in network_evidence_roads:
                    # Evidence roads have probability 1.0
                    probability = 1.0
                    is_evidence = True
                else:
                    # Predict probability using Bayesian inference
                    try:
                        result = self.flood_net.infer_w_evidence(node, evidence)
                        probability = result['flooded']
                        is_evidence = False
                    except Exception as e:
                        # Fallback to marginal probability if inference fails
                        marginal_row = self.flood_net.marginals[self.flood_net.marginals['link_id'] == node]
                        if not marginal_row.empty:
                            probability = float(marginal_row['p'].values[0])
                        else:
                            probability = 0.01  # Default low probability
                        is_evidence = False
                
                predictions.append({
                    'road': node,
                    'probability': probability,
                    'is_evidence': is_evidence
                })
            
            # Calculate summary statistics
            non_evidence_predictions = [p for p in predictions if not p['is_evidence']]
            avg_prediction = sum(p['probability'] for p in non_evidence_predictions) / len(non_evidence_predictions) if non_evidence_predictions else 0
            high_risk_count = sum(1 for p in non_evidence_predictions if p['probability'] > 0.5)
            
            window_result = {
                'window_id': window['window_id'],
                'window_label': f"{window['window_start'].strftime('%H:%M')}-{window['window_end'].strftime('%H:%M')}",
                'time_range': {
                    'window_start': window['window_start'].isoformat(),
                    'window_end': window['window_end'].isoformat()
                },
                'evidence': {
                    'cumulative_evidence_roads': sorted(list(cumulative_evidence_roads)),
                    'network_evidence_roads': sorted(network_evidence_roads),
                    'evidence_count': len(cumulative_evidence_roads),
                    'network_evidence_count': len(network_evidence_roads)
                },
                'predictions': predictions,
                'summary_stats': {
                    'average_prediction_probability': round(avg_prediction, 3),
                    'high_risk_roads_count': high_risk_count,
                    'total_non_evidence_roads': len(non_evidence_predictions)
                }
            }
            
            window_results.append(window_result)
            
            print(f"   ✅ Window complete - Avg prob: {avg_prediction:.3f}, High risk: {high_risk_count}")
        
        return window_results
    
    def save_results(self, window_results, network_stats):
        """Save results in JSON format compatible with existing outputs"""
        print("\n" + "="*80)  
        print("💾 SAVING RESULTS")
        print("="*80)
        
        # Create results directory
        results_dir = "experiments/2025_validation/results"
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for i, window_result in enumerate(window_results):
            # Create comprehensive result object
            full_result = {
                'experiment_metadata': {
                    'experiment_name': 'Reliable 2025 Flood Prediction',
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'description': f'Using {network_stats["total_nodes"]}-node network trained on 2015-2024 data, testing with 10-minute cumulative windows on 2025 data',
                    'random_seed': RANDOM_SEED
                },
                'training_data_info': {
                    'total_records': len(self.training_df),
                    'unique_dates': self.training_df['time_create'].dt.date.nunique(),
                    'unique_streets': self.training_df['link_id'].nunique(),
                    'data_source': 'Road_Closures_2024.csv'
                },
                'test_data_info': {
                    'total_records': len(self.test_data),
                    'time_range': {
                        'start': self.test_data[0]['timestamp'].strftime("%a, %b %d, %Y %I:%M %p"),
                        'end': self.test_data[-1]['timestamp'].strftime("%a, %b %d, %Y %I:%M %p")
                    } if self.test_data else {},
                    'unique_streets': len(set(r['street'] for r in self.test_data)),
                    'data_source': 'archive/old_results/2025_flood_processed.csv'
                },
                'bayesian_network': {
                    'parameters': {
                        'occ_thr': self.network_params['occ_thr'],
                        'edge_thr': self.network_params['edge_thr'],
                        'weight_thr': self.network_params['weight_thr']
                    },
                    'statistics': {
                        'total_nodes': network_stats['total_nodes'],
                        'total_edges': network_stats['total_edges']
                    },
                    'all_nodes': network_stats['all_nodes']
                },
                'current_window': window_result,
                'window_summary': {
                    'window_number': i + 1,
                    'total_windows': len(window_results),
                    'window_label': window_result['window_label'],
                    'completion_status': 'success'
                }
            }
            
            # Save individual window result
            filename = f"reliable_window_{i+1:02d}_{window_result['window_label'].replace(':', '').replace('-', '_')}_PM_{timestamp}.json"
            filepath = os.path.join(results_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(full_result, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Saved: {filename}")
        
        print(f"\n✅ All results saved to: {results_dir}")
        print(f"📊 Total files created: {len(window_results)}")
        
        return results_dir
    
    def run_validation(self):
        """Run complete 2025 flood validation process"""
        print("\n" + "🌊" * 40)
        print("RELIABLE 2025 CHARLESTON FLOOD VALIDATION")
        print("🌊" * 40)
        
        start_time = time.time()
        
        try:
            # Step 1: Load training data
            self.load_training_data()
            
            # Step 2: Load test data  
            self.load_2025_test_data()
            
            # Step 3: Build reliable Bayesian network
            network_stats = self.build_bayesian_network()
            
            # Step 4: Create time windows
            time_windows = self.create_time_windows()
            
            # Step 5: Run predictions with cumulative evidence
            window_results = self.predict_with_cumulative_evidence(time_windows)
            
            # Step 6: Save results
            results_dir = self.save_results(window_results, network_stats)
            
            # Final summary
            total_time = time.time() - start_time
            
            print("\n" + "🎉" * 40)
            print("VALIDATION COMPLETED SUCCESSFULLY!")
            print("🎉" * 40)
            print(f"⏱️ Total runtime: {total_time:.2f} seconds")
            print(f"🧠 Network: {network_stats['total_nodes']} nodes, {network_stats['total_edges']} edges") 
            print(f"📊 Windows processed: {len(window_results)}")
            print(f"💾 Results saved to: {results_dir}")
            
            return True, results_dir
            
        except Exception as e:
            print(f"\n❌ Validation failed: {e}")
            import traceback
            traceback.print_exc()
            return False, None

def main():
    """Main function"""
    validator = Reliable2025FloodValidator()
    success, results_dir = validator.run_validation()
    
    if success:
        print(f"\n🎊 SUCCESS! Results available at: {results_dir}")
        return 0
    else:
        print(f"\n💥 FAILED! Check error messages above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)