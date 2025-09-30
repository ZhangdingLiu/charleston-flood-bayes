#!/usr/bin/env python3
"""
Basic Usage Example - Bayesian Flood Prediction
This example shows how to use the flood prediction system for basic scenarios.
"""

import sys
import os

# Add the package to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python_code import RealTimeFloodPredictor, run_full_prediction_pipeline

def example_1_single_prediction():
    """Example 1: Make a single flood prediction"""
    print("=" * 60)
    print("🌊 EXAMPLE 1: Single Flood Prediction")
    print("=" * 60)
    
    # Initialize predictor
    predictor = RealTimeFloodPredictor()
    
    # Load training data (historical 2015-2024)
    if not predictor.load_training_data():
        print("❌ Failed to load training data")
        return
    
    # Current flood evidence (roads currently flooded)
    evidence_roads = ['KING_ST', 'HUGER_ST', 'FISHBURNE_ST', 'OGIER_ST']
    
    print(f"\n🚨 Current Flood Evidence: {', '.join(evidence_roads)}")
    print("📊 Making predictions for all network nodes...")
    
    # Make prediction
    result = predictor.predict_single_window(
        evidence_roads=evidence_roads,
        output_dir="./example_predictions",
        window_label="current_flood_scenario"
    )
    
    if result:
        predictions = result['current_window']['predictions']
        stats = result['current_window']['summary_stats']
        
        print(f"\n✅ Predictions completed!")
        print(f"   📍 Total roads analyzed: {len(predictions)}")
        print(f"   🎯 Average probability: {stats['average_prediction_probability']:.3f}")
        print(f"   🚨 High-risk roads (>50%): {stats['high_risk_roads_count']}")
        
        # Show top 10 highest probability roads
        non_evidence_roads = [p for p in predictions if not p['is_evidence'] and p['probability']]
        top_roads = sorted(non_evidence_roads, key=lambda x: x['probability'], reverse=True)[:10]
        
        print(f"\n🔝 TOP 10 HIGHEST RISK ROADS:")
        for i, road in enumerate(top_roads, 1):
            prob_pct = road['probability'] * 100
            print(f"   {i:2d}. {road['road']:20s}: {prob_pct:5.1f}%")
    
    print(f"\n{'='*60}\n")

def example_2_time_series_prediction():
    """Example 2: Time series prediction with multiple time windows"""
    print("=" * 60)
    print("🌊 EXAMPLE 2: Time Series Flood Prediction")
    print("=" * 60)
    
    # Run full pipeline with sample data
    print("📊 Processing flood event with 10-minute time windows...")
    
    results = run_full_prediction_pipeline(
        data_dir="../data",
        test_data_path="../data/2025_flood_processed.csv",
        output_dir="./example_predictions/time_series",
        window_minutes=10
    )
    
    if results:
        print(f"\n✅ Time series analysis completed!")
        print(f"   📍 Total windows: {len(results)}")
        
        # Analyze progression over time
        print(f"\n📈 FLOOD PROGRESSION ANALYSIS:")
        for i, result in enumerate(results, 1):
            window_data = result['current_window']
            evidence_count = window_data['evidence']['evidence_count']
            avg_prob = window_data['summary_stats']['average_prediction_probability']
            high_risk = window_data['summary_stats']['high_risk_roads_count']
            
            print(f"   Window {i:2d}: {evidence_count:2d} evidence roads → "
                  f"Avg Risk: {avg_prob:.3f}, High Risk: {high_risk:2d}")
    
    print(f"\n{'='*60}\n")

def example_3_api_style_integration():
    """Example 3: API-style integration (simulating web requests)"""
    print("=" * 60)
    print("🌊 EXAMPLE 3: API-Style Integration")
    print("=" * 60)
    
    # Initialize predictor (would be done once at server startup)
    predictor = RealTimeFloodPredictor()
    predictor.load_training_data()
    
    # Simulate multiple API requests
    api_requests = [
        {
            'request_id': 1,
            'evidence_roads': ['KING_ST', 'MARKET_ST'],
            'scenario': 'Early flood detection'
        },
        {
            'request_id': 2, 
            'evidence_roads': ['KING_ST', 'MARKET_ST', 'MEETING_ST', 'BROAD_ST'],
            'scenario': 'Moderate flooding'
        },
        {
            'request_id': 3,
            'evidence_roads': ['KING_ST', 'MARKET_ST', 'MEETING_ST', 'BROAD_ST', 
                             'CALHOUN_ST', 'ASHLEY_AVE', 'RUTLEDGE_AVE'],
            'scenario': 'Severe flooding'
        }
    ]
    
    print("📡 Processing simulated API requests...")
    
    for request in api_requests:
        print(f"\n🔄 Request {request['request_id']}: {request['scenario']}")
        print(f"   Evidence: {', '.join(request['evidence_roads'])}")
        
        # Make prediction (simulating API endpoint)
        result = predictor.predict_single_window(
            evidence_roads=request['evidence_roads'],
            window_label=f"api_request_{request['request_id']}"
        )
        
        if result:
            stats = result['current_window']['summary_stats']
            print(f"   Result: {stats['average_prediction_probability']:.3f} avg risk, "
                  f"{stats['high_risk_roads_count']} high-risk roads")
    
    print(f"\n{'='*60}\n")

def example_4_custom_flood_event():
    """Example 4: Create custom flood event from scratch"""
    print("=" * 60)
    print("🌊 EXAMPLE 4: Custom Flood Event Analysis") 
    print("=" * 60)
    
    predictor = RealTimeFloodPredictor()
    predictor.load_training_data()
    
    # Create custom flood events (simulating real-time data)
    from datetime import datetime, timedelta
    
    custom_flood_events = [
        {
            'street': 'KING_ST',
            'start_time': datetime(2025, 9, 6, 14, 0),
            'reason': 'FLOOD'
        },
        {
            'street': 'MARKET_ST', 
            'start_time': datetime(2025, 9, 6, 14, 5),
            'reason': 'FLOOD'
        },
        {
            'street': 'MEETING_ST',
            'start_time': datetime(2025, 9, 6, 14, 10),
            'reason': 'FLOOD'
        },
        {
            'street': 'BROAD_ST',
            'start_time': datetime(2025, 9, 6, 14, 15), 
            'reason': 'FLOOD'
        }
    ]
    
    # Load custom events
    if predictor.load_test_data_from_events(custom_flood_events):
        # Create time windows
        windows = predictor.create_time_windows(window_minutes=5)  # 5-minute windows
        
        print(f"📊 Created {len(windows)} time windows from custom events")
        
        for i, window in enumerate(windows, 1):
            evidence_roads = window['evidence_streets']
            if evidence_roads:
                print(f"\n🕐 Window {i} ({window['window_label']}):")
                print(f"   Evidence: {', '.join(evidence_roads)}")
                
                result = predictor.predict_single_window(
                    evidence_roads=evidence_roads,
                    window_label=f"custom_window_{i}"
                )
                
                if result:
                    stats = result['current_window']['summary_stats']
                    print(f"   Prediction: {stats['average_prediction_probability']:.3f} avg risk")
    
    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    print("🌊 BAYESIAN FLOOD PREDICTION - BASIC USAGE EXAMPLES")
    print("🔧 This script demonstrates various ways to use the prediction system")
    print("📁 All prediction files will be saved to ./example_predictions/")
    print("⏱️  Estimated runtime: 30-60 seconds")
    print("\n" + "="*80)
    
    # Run all examples
    try:
        example_1_single_prediction()
        example_2_time_series_prediction()
        example_3_api_style_integration()
        example_4_custom_flood_event()
        
        print("🎉 ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("📁 Check ./example_predictions/ folder for generated files")
        print("📖 See README.md for integration instructions")
        
    except Exception as e:
        print(f"❌ Error running examples: {str(e)}")
        print("💡 Make sure you're in the correct directory and data files exist")