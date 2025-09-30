# 🌊 Bayesian Flood Prediction Integration Package

A complete, self-contained real-time flood prediction system using Bayesian Networks trained on Charleston historical road closure data (2015-2024). This package is designed for easy integration into full-stack web applications.

## 🎯 Key Features

- **88-Node Bayesian Network** trained on 9+ years of historical flood data
- **Real-time Cumulative Prediction** with 10-minute time windows
- **Zero External ML Dependencies** - uses only Python standard library
- **Flexible Input Methods** - CSV files or API events
- **JSON Output Format** - perfect for web APIs
- **Lightweight & Fast** - <100MB memory, sub-second predictions

## 📁 Package Structure

```
bayesian_flood_prediction_integration/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── python_code/                       # Core prediction system
│   ├── realtime_flood_predictor.py    # Main predictor class
│   └── __init__.py                     # Package initialization
├── data/                              # Training and test data
│   ├── Road_Closures_2024.csv        # Historical data (2015-2024)
│   └── 2025_flood_processed.csv       # Sample test data
└── examples/                          # Usage examples
    ├── basic_usage.py                 # Basic usage example
    └── api_integration_example.py     # Flask API integration
```

## 🚀 Quick Start

### 1. Installation

```bash
# Copy this entire package to your project
cp -r bayesian_flood_prediction_integration/ /path/to/your/project/

# Install minimal dependencies
pip install -r bayesian_flood_prediction_integration/requirements.txt
```

### 2. Basic Usage

```python
from bayesian_flood_prediction_integration.python_code import RealTimeFloodPredictor

# Initialize predictor
predictor = RealTimeFloodPredictor()

# Load training data (historical 2015-2024)
predictor.load_training_data()

# Make single prediction
evidence_roads = ['KING_ST', 'HUGER_ST', 'FISHBURNE_ST']
result = predictor.predict_single_window(
    evidence_roads=evidence_roads,
    output_dir="predictions",
    window_label="current_flood"
)

print(f"Predicted {len(result['current_window']['predictions'])} roads")
print(f"High-risk roads: {result['current_window']['summary_stats']['high_risk_roads_count']}")
```

### 3. Full Pipeline (Batch Processing)

```python
from bayesian_flood_prediction_integration.python_code import run_full_prediction_pipeline

# Process entire flood event with time windows
results = run_full_prediction_pipeline(
    data_dir="./data",
    test_data_path="./data/2025_flood_processed.csv",
    output_dir="./predictions",
    window_minutes=10
)

print(f"Generated {len(results)} prediction files")
```

## 🔧 Integration Methods

### Method 1: Standalone Script Integration

Copy the entire package and run as a subprocess:

```python
import subprocess
import json

# Run prediction
result = subprocess.run([
    'python', 'bayesian_flood_prediction_integration/python_code/realtime_flood_predictor.py'
], capture_output=True, text=True)

# Process JSON results
predictions = json.loads(result.stdout)
```

### Method 2: Direct Python Import

Import as a Python module in your backend:

```python
# Add to your Flask/Django/FastAPI backend
import sys
sys.path.append('./bayesian_flood_prediction_integration')

from python_code import RealTimeFloodPredictor

app = Flask(__name__)
predictor = RealTimeFloodPredictor()
predictor.load_training_data()  # Initialize once

@app.route('/api/predict-flood', methods=['POST'])
def predict_flood():
    evidence_roads = request.json.get('evidence_roads', [])
    
    result = predictor.predict_single_window(
        evidence_roads=evidence_roads,
        window_label=f"api_request_{int(time.time())}"
    )
    
    return jsonify(result)
```

### Method 3: Microservice Architecture

Deploy as a separate microservice:

```dockerfile
# Dockerfile
FROM python:3.9-slim
COPY bayesian_flood_prediction_integration/ /app/
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["python", "-m", "http.server", "8000"]
```

## 📊 API Response Format

```json
{
  "experiment_metadata": {
    "experiment_name": "Real-Time Flood Prediction",
    "timestamp": "2025-09-06 15:52:52",
    "description": "Single window prediction using 3 evidence roads"
  },
  "bayesian_network": {
    "parameters": {"occ_thr": 1, "edge_thr": 1, "weight_thr": 0.05},
    "statistics": {"total_nodes": 88, "total_edges": 569},
    "all_nodes": ["AIKEN_ST", "ALEXANDER_ST", "..."]
  },
  "current_window": {
    "window_label": "current_flood",
    "evidence": {
      "evidence_roads": ["KING_ST", "HUGER_ST", "FISHBURNE_ST"],
      "network_evidence_roads": ["KING_ST", "HUGER_ST", "FISHBURNE_ST"],
      "evidence_count": 3
    },
    "predictions": [
      {
        "road": "KING_ST",
        "probability": 1.000,
        "is_evidence": true
      },
      {
        "road": "ASHLEY_AVE", 
        "probability": 0.191,
        "is_evidence": false
      }
    ],
    "summary_stats": {
      "average_prediction_probability": 0.033,
      "high_risk_roads_count": 0,
      "total_non_evidence_roads": 85
    }
  }
}
```

## 🎛️ Configuration Options

### Network Parameters

```python
predictor = RealTimeFloodPredictor()
predictor.network_params = {
    'occ_thr': 1,       # Minimum flood occurrences to include road
    'edge_thr': 1,      # Minimum co-occurrences for network edges  
    'weight_thr': 0.05  # Minimum conditional probability for edges
}
```

### Input Data Formats

**CSV Format (training data):**
```csv
STREET,REASON,created_date
KING ST,FLOOD,2015/08/31 05:00:00+00
HUGER ST,FLOOD,2015/08/31 05:00:00+00
```

**API Event Format:**
```python
flood_events = [
    {
        'street': 'KING_ST',
        'start_time': datetime(2025, 8, 22, 12, 19),
        'reason': 'FLOOD'
    }
]
```

## 🧠 How It Works

1. **Training Phase**:
   - Loads historical road closure data (2015-2024)
   - Filters for flood-related events
   - Builds co-occurrence matrix between roads
   - Creates Bayesian network with conditional probabilities

2. **Prediction Phase**:
   - Receives current flood evidence (flooded roads)
   - Calculates flood probabilities for all network nodes
   - Combines historical frequency with evidence-based inference
   - Returns predictions in JSON format

3. **Network Architecture**:
   - 88 road nodes (Charleston street network)
   - 569+ edges (co-occurrence relationships)
   - Evidence-based inference engine
   - Cumulative time window analysis

## ⚡ Performance Characteristics

- **Memory Usage**: < 100MB
- **Training Time**: ~2-5 seconds
- **Prediction Time**: < 1 second per window
- **Network Size**: 88 nodes, 569 edges
- **Data Size**: ~1.2MB training data

## 🔍 Troubleshooting

### Common Issues

1. **"Training data not found"**
   ```python
   # Solution: Specify absolute path
   predictor.load_training_data('/absolute/path/to/Road_Closures_2024.csv')
   ```

2. **"No valid flood events"**
   ```python
   # Solution: Check date format
   event = {
       'street': 'KING_ST',  # Must be string
       'start_time': 'Fri, Aug 22, 2025 12:19 PM',  # Exact format required
       'reason': 'FLOOD'
   }
   ```

3. **Empty predictions**
   ```python
   # Solution: Check road name format
   evidence_roads = ['KING_ST', 'HUGER_ST']  # Use underscores, UPPERCASE
   ```

### Debug Mode

```python
# Enable detailed logging
predictor = RealTimeFloodPredictor()
predictor.load_training_data()  # Will print detailed loading info
```

## 🏗️ Integration Examples

### Flask API Example

```python
from flask import Flask, request, jsonify
from python_code import RealTimeFloodPredictor

app = Flask(__name__)
predictor = RealTimeFloodPredictor()
predictor.load_training_data()

@app.route('/api/flood/predict', methods=['POST'])
def predict():
    data = request.json
    evidence = data.get('evidence_roads', [])
    
    result = predictor.predict_single_window(evidence)
    return jsonify(result)

@app.route('/api/flood/nodes', methods=['GET'])  
def get_nodes():
    nodes = predictor.get_network_nodes()
    return jsonify({'nodes': nodes, 'count': len(nodes)})
```

### React Frontend Integration

```javascript
// Fetch prediction
const predictFlood = async (evidenceRoads) => {
  const response = await fetch('/api/flood/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({evidence_roads: evidenceRoads})
  });
  
  const result = await response.json();
  return result.current_window.predictions;
};

// Display results
const predictions = await predictFlood(['KING_ST', 'HUGER_ST']);
predictions.forEach(pred => {
  console.log(`${pred.road}: ${(pred.probability * 100).toFixed(1)}%`);
});
```

## 📝 License

MIT License - Free for commercial and personal use.

## 🤝 Support

For integration support, check the `examples/` directory or refer to the inline code documentation.

---

**Ready to integrate!** This package contains everything needed to add real-time flood prediction to your application. 🌊