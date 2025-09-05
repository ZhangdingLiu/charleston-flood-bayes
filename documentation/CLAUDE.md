# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Bayesian Network analysis project for Charleston flood prediction using road closure data. The project analyzes patterns in historical flood events to build predictive networks for flood-prone road segments.

## Core Architecture

**Main Components:**
- `model.py`: Core `FloodBayesNetwork` class implementing Bayesian network construction
- `main.py`: Primary entry point for building and evaluating networks
- `Road_Closures_2024.csv`: Charleston road closure data (2015-2024, 923 flood records)

**Current Script Organization:**
- **Root Directory**: Active scripts for current development
  - `validation_focused_evaluation.py`: Main evaluation pipeline (temporal split)
  - `main.py`: Basic network building with parameter grid search
  - `model.py`: Core Bayesian network implementation
  - `test_*.py`: Testing utilities
- **Archive Directory**: Research experiment scripts moved to `archive/`
  - Organized by functionality (analysis, evaluation, building, optimization)

**Analysis Pipeline:**
1. **Data preprocessing**: Time-based aggregation and road co-occurrence analysis
2. **Network construction**: Building directed acyclic graphs based on road dependencies
3. **Parameter estimation**: Computing conditional probability tables (CPTs)
4. **Evaluation**: Testing on held-out data with various probability thresholds

## Development Commands

**Setup Environment:**
```bash
pip install pgmpy networkx pandas numpy scikit-learn matplotlib seaborn
```

**Main Development Workflow:**
```bash
# 1. Test core validation functions first
python test_validation_script.py

# 2. Run main evaluation pipeline (recommended)
python validation_focused_evaluation.py

# 3. Basic network building with parameter search
python main.py

# 4. Generate visualizations from results
python generate_visualizations.py <results_directory>
```

**Testing Commands:**
```bash
python test_validation_script.py  # Test core validation functions
python test_visualizations.py     # Test visualization functions  
python test_improved_eval.py      # Test evaluation improvements
```

**Active Analysis Scripts:**
```bash
python detailed_analysis.py                    # Comprehensive data analysis
python conservative_negative_evaluation.py     # Conservative evaluation approach
python network_visualization.py                # Network structure visualization
python focused_conservative_evaluation.py      # Conservative precision analysis
python precision_focused_evaluation.py         # High-precision evaluation
python threshold_analysis_and_optimization.py  # Parameter optimization
python verify_best_config.py                   # Verify optimal configuration
python ultra_conservative_ratio_analysis.py    # Ultra-conservative analysis
```

**Archived Scripts:**
Most specialized scripts have been moved to `archive/` directory:
- `archive/analysis_scripts/`: Data diagnosis, co-occurrence analysis
- `archive/evaluation_scripts/`: Various evaluation methodologies  
- `archive/build_scripts/`: Network building variants
- `archive/optimization_scripts/`: Parameter optimization utilities

## Data Handling

**Data Split Approaches:**
- **Random Split**: `sklearn.train_test_split()` used in `main.py` (causes temporal leakage)
- **Temporal Split**: Implemented in `validation_focused_evaluation.py` (recommended approach)
  - Training: 2015-2021, Testing: 2022-2024
  - Prevents same-day flood events appearing in both sets

**Data Structure:**
- Time window: Daily aggregation (`t_window="D"`)
- Random seed: 42 (consistent across all scripts)
- Split ratio: 70% train / 30% test
- Key columns: `time_create`, `STREET` (road names), `OBJECTID`, `REASON`

## Model Parameters

**Core Thresholds:**
- `occ_thr`: Minimum flood occurrences for road inclusion (default: 10)
- `edge_thr`: Minimum co-occurrences for edge creation (default: 3) 
- `weight_thr`: Minimum conditional probability for edges (default: 0.4)
- `prob_thr`: Prediction probability threshold (typically 0.3-0.7)

**Network Constraints:**
- Maximum nodes: ~20-157 depending on parameters
- DAG structure: No cycles allowed
- Laplace smoothing: Applied to avoid zero probabilities

## Dependencies

**Required Libraries:**
- pgmpy: Bayesian network implementation
- networkx: Graph operations
- pandas, numpy: Data manipulation
- sklearn: Train-test splitting and metrics
- matplotlib, seaborn: Visualization

**Installation:**
```bash
pip install pgmpy networkx pandas numpy scikit-learn matplotlib seaborn
```

## Output Files

**Models:** `*.pkl` files containing trained Bayesian networks
**Results:** 
- `results/` directory with JSON metrics and CSV statistics
- `validation_focused_results_*/` timestamped directories from main evaluation pipeline
**Visualizations:** 
- `*.png` files in root directory (network diagrams, confusion matrices)
- `figs/` directory for organized figure storage

## Common Issues

**pgmpy Compatibility:** Scripts handle both `BayesianNetwork` and `DiscreteBayesianNetwork` imports
**Data Leakage:** Current random split causes temporal information leakage
**Sparse Data:** Many roads have limited flood occurrences, affecting network quality
**Zero Probabilities:** Laplace smoothing applied to handle edge cases