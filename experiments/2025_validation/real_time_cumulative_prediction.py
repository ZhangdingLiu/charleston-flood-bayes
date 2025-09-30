#!/usr/bin/env python3
"""
Real-Time Cumulative Flood Prediction System (Reliable Version)
- Train on Road_Closures_2024.csv (2015-2024 historical data) to build reliable Bayesian Network
- Test on 2025 flood data with cumulative evidence windows (10-minute intervals)
- Each time window uses all previous evidence to predict network nodes
- Simulates real-time flood prediction scenario with accumulating evidence
- Uses reliable FloodBayesNetwork from src/models/model.py
"""

import json
import os
import sys
import random
import csv
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import pandas as pd

# Add src/models to path for importing FloodBayesNetwork
sys.path.append('../../src/models')
sys.path.append('src/models')

# Import FloodBayesNetwork
try:
    from model import FloodBayesNetwork
    print("✅ Successfully imported FloodBayesNetwork")
except ImportError as e:
    print(f"❌ Cannot import FloodBayesNetwork: {e}")
    print("   Please ensure src/models/model.py exists and dependencies are installed")
    sys.exit(1)

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

class RealTimeFloodPredictor:
    """Real-time flood prediction with cumulative evidence using reliable Bayesian modeling"""
    
    def __init__(self):
        # Use reliable strategy parameters (~40 nodes)
        self.network_params = {
            'name': 'Reliable Strategy (40 Nodes)',
            'occ_thr': 10,     # Include roads appearing >= 10 times (reliable)
            'edge_thr': 3,     # Create edge if co-occurrence >= 3
            'weight_thr': 0.4, # Include edges with conditional probability >= 0.4
        }
        
        # Data containers
        self.training_data = []
        self.training_df = None  # For FloodBayesNetwork
        self.test_data = {}
        self.flood_net = None    # FloodBayesNetwork instance
        
    def load_training_data(self):
        """Load historical training data from Road_Closures_2024.csv compatible with FloodBayesNetwork"""
        print("📚 Loading training data from Road_Closures_2024.csv...")
        
        # Try different path locations
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
            print("❌ Road_Closures_2024.csv not found in expected locations")
            return False
        
        # Load data using pandas for FloodBayesNetwork compatibility (like validation_focused_evaluation.py)
        try:
            df = pd.read_csv(training_file)
            df = df[df["REASON"].str.upper() == "FLOOD"].copy()
            
            # Preprocess exactly like validation_focused_evaluation.py
            df["time_create"] = pd.to_datetime(df["START"], utc=True)
            df["link_id"] = df["STREET"].str.upper().str.replace(" ", "_")
            df["link_id"] = df["link_id"].astype(str)
            df["id"] = df["OBJECTID"].astype(str)
            df["year"] = df["time_create"].dt.year
            df['flood_date'] = df['time_create'].dt.floor('D')
            
            # Store both formats for compatibility
            self.training_df = df  # For FloodBayesNetwork
            
            # Also keep original format for any remaining legacy code
            for _, row in df.iterrows():
                self.training_data.append({
                    'street': row['link_id'],
                    'date': row['time_create'].strftime('%Y-%m-%d'),
                    'reason': row['REASON']
                })
            
            unique_dates = df['flood_date'].nunique()
            unique_streets = df['link_id'].nunique()
            date_range = f"{df['time_create'].min().strftime('%Y-%m-%d')} to {df['time_create'].max().strftime('%Y-%m-%d')}"
            
            print(f"✅ Loaded {len(df)} flood records for FloodBayesNetwork")
            print(f"📊 Training data: {date_range}, {unique_dates} unique dates, {unique_streets} unique streets")
            
            return len(df) > 0
            
        except Exception as e:
            print(f"❌ Failed to load training data: {e}")
            return False
        
    def parse_time(self, time_str):
        """Parse time string like 'Fri, Aug 22, 2025 12:19 PM' to datetime"""
        try:
            return datetime.strptime(time_str, "%a, %b %d, %Y %I:%M %p")
        except ValueError:
            return None
    
    def load_2025_test_data(self):
        """Load 2025 flood test data and organize by time windows"""
        print("🌊 Loading 2025 flood test data...")
        
        # Try different path locations
        data_paths = [
            '../../archive/old_results/2025_flood_processed.csv',
            'archive/old_results/2025_flood_processed.csv',
            '2025_flood_processed.csv'
        ]
        
        data_file = None
        for path in data_paths:
            if os.path.exists(path):
                data_file = path
                break
                
        if not data_file:
            print("❌ 2025_flood_processed.csv not found in expected locations")
            return False
        
        all_records = []
        with open(data_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                parsed_time = self.parse_time(row['start_time'])
                if parsed_time:
                    row['parsed_time'] = parsed_time
                    all_records.append(row)
        
        # Sort by time
        all_records.sort(key=lambda x: x['parsed_time'])
        
        print(f"✅ Loaded 2025 test data:")
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
        
    def display_network_nodes(self, network_roads):
        """Display all Bayesian network nodes"""
        print(f"\n{'='*60}")
        print(f"🧠 BAYESIAN NETWORK NODES ({len(network_roads)} roads)")
        print(f"{'='*60}")
        
        for i, road in enumerate(sorted(network_roads), 1):
            print(f"{i:3d}. {road}")
        
        print(f"{'='*60}\n")
        
    def create_time_windows(self, window_minutes=10):
        """Create cumulative time windows from 2025 data"""
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
        """Build reliable Bayesian network using FloodBayesNetwork from src/models/model.py"""
        print(f"\n🏗️ Building reliable Bayesian network...")
        print(f"   Parameters: occ_thr={self.network_params['occ_thr']}, edge_thr={self.network_params['edge_thr']}, weight_thr={self.network_params['weight_thr']}")
        
        if self.training_df is None:
            print("❌ Training data not loaded properly")
            return None, False
        
        try:
            # Step 1: Create FloodBayesNetwork instance
            self.flood_net = FloodBayesNetwork(t_window="D")
            
            # Step 2: Fit marginal probabilities
            print("   📊 Computing marginal probabilities...")
            self.flood_net.fit_marginal(self.training_df)
            print(f"   ✅ Marginal probabilities computed for {len(self.flood_net.marginals)} roads")
            
            # Step 3: Build network structure with co-occurrence filtering
            print(f"   🏗️ Building network structure...")
            self.flood_net.build_network_by_co_occurrence(
                self.training_df,
                occ_thr=self.network_params['occ_thr'],
                edge_thr=self.network_params['edge_thr'], 
                weight_thr=self.network_params['weight_thr'],
                report=False
            )
            
            nodes_count = self.flood_net.network.number_of_nodes()
            edges_count = self.flood_net.network.number_of_edges()
            print(f"   ✅ Network structure built: {nodes_count} nodes, {edges_count} edges")
            
            # Step 4: Fit conditional probabilities
            print(f"   🧮 Computing conditional probabilities...")
            self.flood_net.fit_conditional(self.training_df, max_parents=2, alpha=1.0)
            print(f"   ✅ Conditional probabilities computed")
            
            # Step 5: Build final Bayesian network
            print(f"   🔗 Building final Bayesian network...")
            self.flood_net.build_bayes_network()
            print(f"   ✅ Bayesian network construction completed")
            
            # Step 6: Network validation
            try:
                self.flood_net.check_bayesian_network()
                print(f"   ✅ Network validation passed")
            except Exception as e:
                print(f"   ⚠️ Network validation warning: {e}")
            
            # Get final network statistics
            bn_nodes = set(self.flood_net.network_bayes.nodes())
            all_nodes = sorted(list(bn_nodes))
            
            print(f"   📊 Final network statistics:")
            print(f"       - Total nodes: {len(all_nodes)}")
            print(f"       - Total edges: {edges_count}")
            print(f"       - Network type: Directed Acyclic Graph (DAG)")
            
            # Store network stats for later use
            self.network_stats = {
                'total_nodes': len(all_nodes),
                'total_edges': edges_count,
                'all_nodes': all_nodes
            }
            
            return self.flood_net, True
            
        except Exception as e:
            print(f"❌ Network construction failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, False

def main():
    """Main function for real-time cumulative flood prediction with reliable Bayesian modeling"""
    print("🌊 Real-Time Cumulative Flood Prediction System (Reliable Version)")
    print("🎯 Using reliable Bayesian network trained on 2015-2024 data")
    print("⏱️  Testing with 10-minute cumulative windows on 2025 data")
    print("🧠 Powered by FloodBayesNetwork from src/models/model.py")
    print("="*80)
    
    # Initialize results collection
    experiment_results = {
        "experiment_metadata": {
            "experiment_name": "Real-Time Cumulative Flood Prediction",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": "Using 88-node network trained on 2015-2024 data, testing with 10-minute cumulative windows on 2025 data",
            "random_seed": RANDOM_SEED
        }
    }
    
    predictor = RealTimeFloodPredictor()
    
    # Step 1: Load training data
    print("\n📚 Loading training data...")
    if not predictor.load_training_data():
        print("❌ Failed to load training data")
        return None
    
    # Collect training data info
    experiment_results["training_data_info"] = {
        "total_records": len(predictor.training_data),
        "unique_dates": len(set(r['date'] for r in predictor.training_data)),
        "unique_streets": len(set(r['street'] for r in predictor.training_data)),
        "data_source": "Road_Closures_2024.csv"
    }
        
    # Step 2: Load 2025 test data
    print("\n🌊 Loading 2025 test data...")  
    if not predictor.load_2025_test_data():
        print("❌ Failed to load 2025 test data")
        return None
    
    # Collect test data info
    experiment_results["test_data_info"] = {
        "total_records": len(predictor.test_data['records']),
        "time_range": {
            "start": predictor.test_data['records'][0]['start_time'],
            "end": predictor.test_data['records'][-1]['start_time']
        },
        "unique_streets": len(predictor.test_data['all_streets']),
        "data_source": "archive/old_results/2025_flood_processed.csv"
    }
    
    # Step 3: Build reliable Bayesian network
    print("\n🏗️ Building reliable Bayesian network...")
    flood_net, success = predictor.build_bayesian_network()
    if not success:
        print("❌ Failed to build reliable Bayesian network")
        return None
        
    # Get network nodes from FloodBayesNetwork
    network_roads = list(flood_net.network_bayes.nodes())
    
    # Collect network info using reliable network statistics
    experiment_results["bayesian_network"] = {
        "parameters": {
            "occ_thr": predictor.network_params['occ_thr'],
            "edge_thr": predictor.network_params['edge_thr'], 
            "weight_thr": predictor.network_params['weight_thr']
        },
        "statistics": {
            "total_nodes": predictor.network_stats['total_nodes'],
            "total_edges": predictor.network_stats['total_edges']
        },
        "all_nodes": predictor.network_stats['all_nodes']
    }
    
    # Step 4: Display all network nodes
    predictor.display_network_nodes(network_roads)
    
    # Step 5: Create time windows
    print("⏱️ Creating 10-minute cumulative time windows...")
    windows = predictor.create_time_windows(window_minutes=10)
    print(f"✅ Created {len(windows)} time windows")
    
    # Step 6: Process each time window
    print(f"\n{'='*80}")
    print("🔮 REAL-TIME CUMULATIVE PREDICTION RESULTS")
    print(f"{'='*80}")
    
    # Create base experiment info for each window
    base_experiment_info = {
        "experiment_metadata": experiment_results["experiment_metadata"],
        "training_data_info": experiment_results["training_data_info"],
        "test_data_info": experiment_results["test_data_info"],
        "bayesian_network": experiment_results["bayesian_network"]
    }
    
    # Results directory for window files
    results_dir = "experiments/2025_validation/results"
    timestamp_base = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for i, window in enumerate(windows, 1):
        print(f"\n📅 Window {i}: {window['window_label']}")
        print(f"   📍 Time Range: From start to {window['window_end'].strftime('%I:%M %p')}")
        print(f"   🔑 Cumulative Evidence Roads ({len(window['evidence_streets'])}): {', '.join(sorted(window['evidence_streets']))}")
        
        # Set up evidence for this window
        evidence = {}
        for road in window['evidence_streets']:
            if road in network_roads:
                evidence[road] = 1
        
        if not evidence:
            print("   ⚠️ No evidence roads in network - skipping predictions")
            continue
            
        print(f"   🧠 Network Evidence ({len(evidence)}): {', '.join(sorted(evidence.keys()))}")
        print(f"\n   🔮 Predictions for all {len(network_roads)} network nodes:")
        
        # Initialize window result
        window_result = {
            "window_id": i,
            "window_label": window['window_label'],
            "time_range": {
                "window_start": window['window_start'].isoformat(),
                "window_end": window['window_end'].isoformat()
            },
            "evidence": {
                "cumulative_evidence_roads": sorted(window['evidence_streets']),
                "network_evidence_roads": sorted(evidence.keys()),
                "evidence_count": len(window['evidence_streets']),
                "network_evidence_count": len(evidence)
            },
            "predictions": []
        }
        
        # Make predictions for ALL network nodes
        predictions_summary = []
        
        for road in sorted(network_roads):
            try:
                if road in evidence:
                    prob = 1.000  # Evidence roads have probability 1
                    print(f"      {road:20s}: {prob:.3f} (evidence)")
                else:
                    result = flood_net.infer_w_evidence(road, evidence)
                    prob = result.get('flooded', result.get(1, 0.0))
                    print(f"      {road:20s}: {prob:.3f}")
                
                # Add to both summaries
                prediction_data = {'road': road, 'probability': prob, 'is_evidence': road in evidence}
                predictions_summary.append(prediction_data)
                window_result["predictions"].append(prediction_data)
                
            except Exception as e:
                print(f"      {road:20s}: ERROR ({str(e)})")
                error_data = {'road': road, 'probability': None, 'is_evidence': False, 'error': str(e)}
                predictions_summary.append(error_data)
                window_result["predictions"].append(error_data)
        
        # Quick summary stats for this window
        non_evidence_preds = [p['probability'] for p in predictions_summary if not p['is_evidence'] and p['probability'] is not None]
        if non_evidence_preds:
            avg_pred = sum(non_evidence_preds) / len(non_evidence_preds)
            high_risk_count = len([p for p in non_evidence_preds if p > 0.5])
            print(f"\\n   📊 Window Summary:")
            print(f"      Average prediction probability: {avg_pred:.3f}")
            print(f"      High-risk roads (>0.5): {high_risk_count}/{len(non_evidence_preds)}")
            
            # Add summary stats to window result
            window_result["summary_stats"] = {
                "average_prediction_probability": round(avg_pred, 3),
                "high_risk_roads_count": high_risk_count,
                "total_non_evidence_roads": len(non_evidence_preds)
            }
        else:
            window_result["summary_stats"] = {
                "average_prediction_probability": 0.0,
                "high_risk_roads_count": 0,
                "total_non_evidence_roads": 0
            }
        
        # Create individual window JSON file
        window_json_result = {
            **base_experiment_info,
            "current_window": window_result,
            "window_summary": {
                "window_number": i,
                "total_windows": len(windows),
                "window_label": window['window_label'],
                "completion_status": "success"
            }
        }
        
        # Save individual window result to JSON file
        window_filename = f"realtime_window_{i:02d}_{window['window_label'].replace(':', '').replace('-', '_').replace(' ', '_')}_{timestamp_base}.json"
        window_filepath = os.path.join(results_dir, window_filename)
        
        try:
            with open(window_filepath, 'w', encoding='utf-8') as f:
                json.dump(window_json_result, f, indent=2, ensure_ascii=False)
            print(f"   💾 Window {i} saved to: {window_filename}")
        except Exception as e:
            print(f"   ❌ Failed to save window {i}: {str(e)}")
        
        print(f"   {'-'*60}")
    
    print(f"\\n🎉 Real-time cumulative prediction completed!")
    print(f"📊 Processed {len(windows)} time windows with accumulating evidence")
    print(f"🧠 Made predictions for {len(network_roads)} network nodes in each window")
    print(f"💾 Generated {len(windows)} individual window JSON files")
    print(f"📁 Files saved in: {results_dir}/")
    
    return True

if __name__ == "__main__":
    main()