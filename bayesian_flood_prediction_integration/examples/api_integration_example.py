#!/usr/bin/env python3
"""
API Integration Example - Bayesian Flood Prediction
This example shows how to integrate the flood prediction system into web APIs.
"""

import sys
import os
import json
from datetime import datetime

# Add the package to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python_code import RealTimeFloodPredictor

# Global predictor instance (initialize once at server startup)
predictor = None

def initialize_predictor():
    """Initialize the predictor (call once at server startup)"""
    global predictor
    print("🏗️ Initializing Bayesian Flood Predictor...")
    
    predictor = RealTimeFloodPredictor()
    success = predictor.load_training_data()
    
    if success:
        print("✅ Predictor initialized successfully")
        print(f"📊 Network ready with {len(predictor.get_network_nodes())} nodes")
        return True
    else:
        print("❌ Failed to initialize predictor")
        return False

# =================== FLASK INTEGRATION EXAMPLE ===================

def create_flask_app():
    """Create Flask application with flood prediction endpoints"""
    try:
        from flask import Flask, request, jsonify, render_template_string
    except ImportError:
        print("❌ Flask not installed. Install with: pip install flask")
        return None
    
    app = Flask(__name__)
    
    @app.route('/api/flood/predict', methods=['POST'])
    def predict_flood():
        """API endpoint for flood prediction"""
        try:
            data = request.json
            evidence_roads = data.get('evidence_roads', [])
            
            if not evidence_roads:
                return jsonify({
                    'error': 'No evidence roads provided',
                    'example': {'evidence_roads': ['KING_ST', 'MARKET_ST']}
                }), 400
            
            # Make prediction
            result = predictor.predict_single_window(
                evidence_roads=evidence_roads,
                window_label=f"api_{int(datetime.now().timestamp())}"
            )
            
            if result:
                return jsonify(result)
            else:
                return jsonify({'error': 'Prediction failed'}), 500
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/flood/nodes', methods=['GET'])
    def get_network_nodes():
        """Get all Bayesian network nodes"""
        try:
            nodes = predictor.get_network_nodes()
            return jsonify({
                'nodes': nodes,
                'count': len(nodes),
                'description': 'All roads in the Bayesian network'
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/flood/batch-predict', methods=['POST'])
    def batch_predict():
        """Batch prediction for multiple scenarios"""
        try:
            data = request.json
            scenarios = data.get('scenarios', [])
            
            results = []
            for i, scenario in enumerate(scenarios):
                evidence_roads = scenario.get('evidence_roads', [])
                scenario_name = scenario.get('name', f'scenario_{i+1}')
                
                if evidence_roads:
                    result = predictor.predict_single_window(
                        evidence_roads=evidence_roads,
                        window_label=scenario_name
                    )
                    
                    if result:
                        results.append({
                            'scenario_name': scenario_name,
                            'evidence_count': len(evidence_roads),
                            'predictions': result['current_window']['predictions'],
                            'summary_stats': result['current_window']['summary_stats']
                        })
            
            return jsonify({
                'batch_results': results,
                'total_scenarios': len(results)
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/')
    def index():
        """Simple web interface for testing"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>🌊 Flood Prediction API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .example { background: #f5f5f5; padding: 20px; margin: 20px 0; border-radius: 5px; }
                code { background: #e8e8e8; padding: 2px 5px; border-radius: 3px; }
                button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
                button:hover { background: #0056b3; }
                #result { background: #f8f9fa; padding: 15px; margin-top: 20px; border-radius: 5px; white-space: pre-wrap; }
            </style>
        </head>
        <body>
            <h1>🌊 Bayesian Flood Prediction API</h1>
            <p>Real-time flood prediction system with 88-node Bayesian network</p>
            
            <div class="example">
                <h3>📡 API Endpoints</h3>
                <ul>
                    <li><code>POST /api/flood/predict</code> - Single flood prediction</li>
                    <li><code>GET /api/flood/nodes</code> - Get all network nodes</li>
                    <li><code>POST /api/flood/batch-predict</code> - Batch predictions</li>
                </ul>
            </div>
            
            <div class="example">
                <h3>🧪 Test Prediction</h3>
                <p>Evidence Roads (comma-separated):</p>
                <input type="text" id="evidence" value="KING_ST,MARKET_ST,MEETING_ST" style="width: 300px; padding: 5px;">
                <br><br>
                <button onclick="testPrediction()">🔮 Make Prediction</button>
                <div id="result"></div>
            </div>
            
            <script>
                async function testPrediction() {
                    const evidence = document.getElementById('evidence').value.split(',').map(s => s.trim());
                    const resultDiv = document.getElementById('result');
                    
                    resultDiv.textContent = 'Making prediction...';
                    
                    try {
                        const response = await fetch('/api/flood/predict', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({evidence_roads: evidence})
                        });
                        
                        const result = await response.json();
                        
                        if (response.ok) {
                            const stats = result.current_window.summary_stats;
                            let output = `✅ PREDICTION RESULTS\\n`;
                            output += `Evidence Roads: ${evidence.join(', ')}\\n`;
                            output += `Average Risk: ${(stats.average_prediction_probability * 100).toFixed(1)}%\\n`;
                            output += `High-Risk Roads: ${stats.high_risk_roads_count}\\n\\n`;
                            
                            output += `TOP 10 HIGHEST RISK ROADS:\\n`;
                            const nonEvidence = result.current_window.predictions
                                .filter(p => !p.is_evidence && p.probability)
                                .sort((a, b) => b.probability - a.probability)
                                .slice(0, 10);
                            
                            nonEvidence.forEach((road, i) => {
                                output += `${i+1}. ${road.road}: ${(road.probability * 100).toFixed(1)}%\\n`;
                            });
                            
                            resultDiv.textContent = output;
                        } else {
                            resultDiv.textContent = `❌ Error: ${result.error}`;
                        }
                    } catch (error) {
                        resultDiv.textContent = `❌ Network Error: ${error.message}`;
                    }
                }
            </script>
        </body>
        </html>
        """
        return html
    
    return app

# =================== FASTAPI INTEGRATION EXAMPLE ===================

def create_fastapi_app():
    """Create FastAPI application with flood prediction endpoints"""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import HTMLResponse
        from pydantic import BaseModel
        from typing import List, Optional
    except ImportError:
        print("❌ FastAPI not installed. Install with: pip install fastapi uvicorn")
        return None
    
    app = FastAPI(
        title="🌊 Bayesian Flood Prediction API",
        description="Real-time flood prediction using Bayesian Networks",
        version="1.0.0"
    )
    
    # Request models
    class PredictionRequest(BaseModel):
        evidence_roads: List[str]
        window_label: Optional[str] = None
    
    class BatchScenario(BaseModel):
        name: str
        evidence_roads: List[str]
    
    class BatchRequest(BaseModel):
        scenarios: List[BatchScenario]
    
    @app.post("/api/flood/predict")
    async def predict_flood(request: PredictionRequest):
        """Make flood prediction for given evidence roads"""
        try:
            result = predictor.predict_single_window(
                evidence_roads=request.evidence_roads,
                window_label=request.window_label or f"api_{int(datetime.now().timestamp())}"
            )
            
            if result:
                return result
            else:
                raise HTTPException(status_code=500, detail="Prediction failed")
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/flood/nodes")
    async def get_network_nodes():
        """Get all Bayesian network nodes"""
        try:
            nodes = predictor.get_network_nodes()
            return {
                "nodes": nodes,
                "count": len(nodes),
                "description": "All roads in the Bayesian network"
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/flood/batch-predict")
    async def batch_predict(request: BatchRequest):
        """Batch prediction for multiple scenarios"""
        try:
            results = []
            
            for scenario in request.scenarios:
                if scenario.evidence_roads:
                    result = predictor.predict_single_window(
                        evidence_roads=scenario.evidence_roads,
                        window_label=scenario.name
                    )
                    
                    if result:
                        results.append({
                            "scenario_name": scenario.name,
                            "evidence_count": len(scenario.evidence_roads),
                            "predictions": result['current_window']['predictions'],
                            "summary_stats": result['current_window']['summary_stats']
                        })
            
            return {
                "batch_results": results,
                "total_scenarios": len(results)
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/", response_class=HTMLResponse)
    async def root():
        return """
        <html>
            <head><title>🌊 Flood Prediction API - FastAPI</title></head>
            <body style="font-family: Arial, sans-serif; margin: 40px;">
                <h1>🌊 Bayesian Flood Prediction API (FastAPI)</h1>
                <p>Visit <a href="/docs">/docs</a> for interactive API documentation</p>
                <p>Visit <a href="/redoc">/redoc</a> for alternative documentation</p>
            </body>
        </html>
        """
    
    return app

# =================== USAGE EXAMPLES ===================

def example_direct_api_calls():
    """Example of using the predictor directly (without web server)"""
    print("🔧 DIRECT API CALLS EXAMPLE")
    print("=" * 50)
    
    if not predictor:
        print("❌ Predictor not initialized")
        return
    
    # Simulate API-style calls
    test_scenarios = [
        {
            'name': 'Downtown Flooding',
            'evidence_roads': ['KING_ST', 'MARKET_ST', 'MEETING_ST']
        },
        {
            'name': 'Coastal Flooding', 
            'evidence_roads': ['E_BAY_ST', 'LOCKWOOD_DR', 'ASHLEY_AVE']
        },
        {
            'name': 'University Area',
            'evidence_roads': ['CALHOUN_ST', 'RUTLEDGE_AVE', 'COMING_ST']
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n🌊 Scenario: {scenario['name']}")
        print(f"   Evidence: {', '.join(scenario['evidence_roads'])}")
        
        result = predictor.predict_single_window(
            evidence_roads=scenario['evidence_roads'],
            window_label=scenario['name'].lower().replace(' ', '_')
        )
        
        if result:
            stats = result['current_window']['summary_stats']
            print(f"   📊 Average Risk: {stats['average_prediction_probability']:.3f}")
            print(f"   🚨 High-Risk Roads: {stats['high_risk_roads_count']}")
            
            # Show top 3 predictions
            predictions = result['current_window']['predictions']
            non_evidence = [p for p in predictions if not p['is_evidence'] and p['probability']]
            top_3 = sorted(non_evidence, key=lambda x: x['probability'], reverse=True)[:3]
            
            print("   🔝 Top 3 Risks:")
            for i, road in enumerate(top_3, 1):
                print(f"      {i}. {road['road']}: {road['probability']*100:.1f}%")

def run_flask_server():
    """Run Flask development server"""
    app = create_flask_app()
    if app:
        print("🚀 Starting Flask server on http://localhost:5000")
        print("📖 Visit http://localhost:5000 for web interface")
        print("📡 API endpoints available at /api/flood/*")
        app.run(debug=True, host='0.0.0.0', port=5000)

def run_fastapi_server():
    """Run FastAPI server"""
    try:
        import uvicorn
        app = create_fastapi_app()
        if app:
            print("🚀 Starting FastAPI server on http://localhost:8000")
            print("📖 Visit http://localhost:8000/docs for Swagger UI")
            print("📡 API endpoints available at /api/flood/*")
            uvicorn.run(app, host="0.0.0.0", port=8000)
    except ImportError:
        print("❌ uvicorn not installed. Install with: pip install uvicorn")

if __name__ == "__main__":
    print("🌊 BAYESIAN FLOOD PREDICTION - API INTEGRATION EXAMPLES")
    print("=" * 70)
    
    # Initialize predictor
    if initialize_predictor():
        print("\\n📋 Available examples:")
        print("1. Direct API calls (no web server)")
        print("2. Flask web server")  
        print("3. FastAPI server")
        
        choice = input("\\n🔢 Choose example (1-3) or press Enter for direct calls: ").strip()
        
        if choice == "2":
            run_flask_server()
        elif choice == "3":
            run_fastapi_server()
        else:
            example_direct_api_calls()
            
    else:
        print("❌ Failed to initialize. Check that data files exist.")
        print("💡 Make sure you're running from the correct directory")