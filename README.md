# Charleston Flood Prediction Using Bayesian Networks

A comprehensive Bayesian Network analysis project for predicting Charleston flood events using historical road closure data. This system analyzes patterns in flood-related road closures to build predictive networks for flood-prone road segments.

[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-active-brightgreen.svg)]()

## 📋 Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)

## 🌊 Overview

This project implements a sophisticated Bayesian Network model to predict flood events in Charleston using road closure data spanning 2015-2024. The system employs advanced parameter optimization, temporal data splitting to prevent leakage, and comprehensive evaluation metrics.

**Key Highlights:**
- 🧠 Advanced Bayesian Network implementation with DAG structure
- ⚡ Grid search optimization with 4,320 parameter combinations
- 🔬 Strict temporal validation (2015-2021 training, 2022-2024 testing)
- 📊 Comprehensive visualization and analysis tools
- 🎯 Multiple evaluation strategies with constraint filtering

## 📁 Project Structure

```
├── src/                            # Primary source code ⭐
│   ├── models/                     # Core models
│   │   ├── model.py                # FloodBayesNetwork core class ⭐
│   │   ├── main.py                 # Basic network construction entry ⭐
│   │   └── Road_Closures_2024.csv  # Main dataset (2015-2024, 923 records)
│   ├── evaluation/                 # Evaluation scripts
│   │   ├── latest/                 # Latest recommended versions
│   │   │   ├── validation_focused_evaluation.py  # Main evaluation pipeline ⭐
│   │   │   └── pilot_conservative_evaluation.py  # Quick evaluation
│   │   ├── conservative_strategies/ # Conservative strategy evaluation
│   │   ├── precision_focused/      # Precision optimization evaluation
│   │   ├── experimental/           # Experimental evaluation
│   │   └── [test_scripts]          # Various test scripts
│   ├── analysis/                   # Analysis tools
│   │   ├── comprehensive_parameter_grid_search.py ⭐ # Grid search engine
│   │   ├── run_parameter_optimization.py ⭐        # Parameter optimization controller (4,320 combinations)
│   │   ├── test_parameter_optimization.py          # Quick test (128 combinations)
│   │   └── [other_analysis_tools]  # Threshold optimization, data analysis, etc.
│   ├── visualization/              # Visualization tools
│   │   ├── parameter_analysis_visualizer.py ⭐     # Parameter analysis visualization
│   │   ├── network_visualization.py               # Network structure diagrams
│   │   └── [other_visualization_tools] # Chart generation, reports, etc.
│   └── data_processing/            # Data processing (reserved)
├── experiments/                    # Experiments and specialized analysis
│   ├── 2025_validation/            # 2025 new data validation ⭐
│   ├── parameter_tuning/           # Test set evaluation experiments
│   │   ├── evaluate_top_configs_on_test_set.py ⭐  # Best configuration testing
│   │   └── [other_test_evaluations] # Flexible parameters, specified parameter testing
│   ├── flood_specific/             # Specific flood event analysis
│   └── debug/                      # Debug scripts
├── documentation/                  # Project documentation 📚
│   ├── CLAUDE.md                   # Chinese development guide
│   ├── PARAMETER_OPTIMIZATION_GUIDE.md # Parameter optimization guide
│   └── [technical_reports]         # Various analysis reports
├── results/                        # Result files 📊
│   ├── latest/                     # Latest results
│   │   └── validation_focused_results_20250714_190013/ ⭐ # Best results
│   ├── parameter_optimization_*/    # Parameter optimization results
│   └── [historical_results]        # Other historical results
├── archive/                        # Archive area 🗄️
│   ├── deprecated_scripts/         # Deprecated scripts
│   ├── old_results/                # Old result files
│   └── core_backup_*/              # Important file backups
└── .gitignore                      # Git ignore configuration
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/charleston-flood-prediction.git
cd charleston-flood-prediction

# Install dependencies
pip install pgmpy networkx pandas numpy scikit-learn matplotlib seaborn
```

### Core Functions - 3 Steps to Start

```bash
# 1. Basic network construction  
python src/models/main.py

# 2. Parameter optimization (Recommended) ⭐  
python src/analysis/run_parameter_optimization.py

# 3. Main evaluation pipeline
python src/evaluation/latest/validation_focused_evaluation.py
```

## ✨ Key Features

### 🧠 **Bayesian Network Core**
- **FloodBayesNetwork**: Core model class building DAG based on road dependencies
- **Temporal Split Validation**: Strict temporal splitting to prevent data leakage (2015-2021 training, 2022-2024 testing)
- **Conditional Probability Estimation**: Laplace smoothing for handling sparse data

### ⚡ **Parameter Optimization System** 
- **4,320 Parameter Combinations**: Comprehensive grid search with constraint filtering
- **Multi-Strategy Recommendations**: 5 configurations - Best F1, High Precision, High Recall, Balanced, Robust
- **Visualization Analysis**: 3D scatter plots, heatmaps, sensitivity analysis, Pareto frontier

### 🔬 **Comprehensive Evaluation**
- **Test Set Evaluation**: Independent test set validation for best configuration performance  
- **2025 Data Validation**: Model validation with latest flood data
- **Specific Event Analysis**: Case studies of historical major flood events

## 📊 Usage

### Basic Network Construction
```bash
python src/models/main.py
```

### Parameter Optimization
```bash
# Full parameter optimization (4,320 combinations)
python src/analysis/run_parameter_optimization.py

# Quick test (128 combinations)
python src/analysis/test_parameter_optimization.py
```

### Evaluation Pipeline
```bash
# Main evaluation pipeline (recommended)
python src/evaluation/latest/validation_focused_evaluation.py

# Quick evaluation
python src/evaluation/latest/pilot_conservative_evaluation.py
```

### Test Set Evaluation
```bash
# Best configuration test set evaluation (recommended)
python experiments/parameter_tuning/evaluate_top_configs_on_test_set.py

# Specified parameter testing
python experiments/parameter_tuning/evaluate_specified_params_on_test.py

# Flexible parameter testing
python experiments/parameter_tuning/evaluate_focused_flexible_params.py
```

### Data Analysis
```bash
# Data quality analysis
python src/analysis/detailed_analysis_fixed.py

# Threshold optimization
python src/analysis/threshold_analysis_and_optimization.py

# Grid search
python src/analysis/comprehensive_parameter_grid_search.py
```

### Visualization
```bash
# Network structure diagrams
python src/visualization/network_visualization.py

# Parameter analysis plots
python src/visualization/parameter_analysis_visualizer.py
```

### 2025 Data Validation
```bash
python experiments/2025_validation/validate_2025_flood_data_fixed.py
```

## 📈 Model Parameters

**Core Thresholds:**
- `occ_thr`: Minimum flood occurrence count for road inclusion (default: 10)
- `edge_thr`: Minimum co-occurrence count for edge creation (default: 3)
- `weight_thr`: Minimum conditional probability for edges (default: 0.4)
- `prob_thr`: Prediction probability threshold (typically 0.3-0.7)

**Network Constraints:**
- Maximum nodes: ~20-157 (depending on parameters)
- DAG structure: No cycles allowed
- Laplace smoothing: Prevents zero probabilities

## 📊 Results Analysis

**Best Results Location:**
- **`results/latest/validation_focused_results_20250714_190013/`** - Contains complete PDF/PNG visualizations, parameter sensitivity analysis, and performance comparisons

**Result File Types:**
- **Models**: `*.pkl` trained Bayesian networks
- **Metrics**: JSON format evaluation metrics
- **Visualizations**: Network diagrams, confusion matrices, performance charts
- **Data**: CSV format detailed results

## 🔧 Development Tips

1. **Parameter Optimization Priority**: Use the new parameter optimization system to find optimal configurations
2. **Use Latest Scripts**: Scripts in `src/evaluation/latest/` represent the latest best practices
3. **Focus on Latest Results**: Prioritize analysis of outputs in `results/latest/`
4. **Test-Driven**: Use scripts in `src/evaluation/` to verify functionality
5. **Visualization Priority**: Use `src/visualization/` tools to understand model behavior
6. **Constraint Filtering**: Set appropriate performance constraints based on application needs
7. **Temporal Split Validation**: Prioritize using `src/evaluation/latest/validation_focused_evaluation.py` to avoid data leakage
8. **Test Set Evaluation**: Use `experiments/parameter_tuning/evaluate_top_configs_on_test_set.py` for reliable test performance

## 🎯 Summary

This is a **clean, lightweight** Charleston flood prediction Bayesian network project:
- ✅ **Single Source Directory** (`src/`) - Avoids duplication and confusion  
- ✅ **Git-Friendly** - Real files instead of symbolic links
- ✅ **Feature Complete** - Parameter optimization, evaluation validation, 2025 data testing
- ✅ **Well Documented** - Clear usage guides and best practices
- ✅ **Reproducible** - Strict temporal splitting and random seed control

**Core Command Quick Reference:**
```bash
python src/models/main.py                                         # Basic network
python src/analysis/run_parameter_optimization.py                # Parameter optimization  
python src/evaluation/latest/validation_focused_evaluation.py    # Main evaluation
python experiments/parameter_tuning/evaluate_top_configs_on_test_set.py  # Test set evaluation
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📮 Contact

Your Name - your.email@example.com

Project Link: [https://github.com/your-username/charleston-flood-prediction](https://github.com/your-username/charleston-flood-prediction)

---

⭐ **Star this repository if you find it helpful!**