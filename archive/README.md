# Archive Directory

This directory contains code and files that are not actively being used in the main development workflow but are preserved for reference and potential future use.

## Directory Structure

### analysis_scripts/
Scripts for data analysis and exploration:
- `data_diagnosis.py` - Analyzes data quality and sparsity issues
- `cooccurrence_analysis.py` - Road pair association analysis  
- `train_cooccurrence_analysis.py` - Training-only co-occurrence statistics
- `visualization.py` - Network visualization utilities

### evaluation_scripts/
Scripts for evaluating different network models:
- `evaluate_manual_bn.py` - Evaluate manual 4-road network
- `evaluate_testset.py` - Comprehensive test set evaluation
- `evaluate_threshold_k.py` - Threshold and Top-K optimization
- `evaluate_train_based_bn.py` - Evaluate training-based network
- `evaluate_ultra_core.py` - Evaluate minimal network

### build_scripts/
Scripts for building different network variants:
- `build_manual_bn.py` - 4-road manual network builder
- `build_train_based_bn.py` - Network based on training set statistics
- `build_ultra_core.py` - Minimal network using Chow-Liu algorithm

### optimization_scripts/
Scripts for parameter optimization and testing:
- `parameter_optimization.py` - Grid search for optimal parameters
- `targeted_optimization.py` - Focused optimization experiments
- `fixed_model.py` - Bug fixes and improvements
- `quick_test.py` - Quick testing utilities
- `main_clean.py` - Alternative main script

### results_backup/
Backup of all results, models, and figures:
- `results/` - JSON metrics and CSV statistics
- `figs/` - Confusion matrices and network diagrams  
- `*.pkl` - Trained Bayesian network models
- `*.json` - Analysis results and metrics
- `*.csv` - Statistical data and summaries
- `*.md` - Documentation and reports

## Usage

To use any archived script, you can either:
1. Copy it back to the main directory temporarily
2. Run it from the archive directory with adjusted import paths
3. Reference it for understanding previous approaches

## Note

These files represent the evolution of the project and different experimental approaches. They are preserved for reference but are not part of the current active development workflow.