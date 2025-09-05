# Bayesian Network Flood Prediction - Visualization Guide

## Overview

This guide explains the visualization functions added to `validation_focused_evaluation.py` and the generated charts that provide insights into the Charleston flood prediction Bayesian network model.

## Generated Visualizations

### 1. Bayesian Network Structure (`bayesian_network_structure.png`)

**Purpose**: Visualizes the structure and properties of the learned Bayesian network.

**Components**:
- **Left Panel**: Spring layout showing network topology
  - **Node Color**: Flood probability (red = high risk, blue = low risk)
  - **Node Size**: Network connectivity (larger = more connections)
  - **Edges**: Directed relationships between roads
  - **Colorbar**: Shows flood probability scale

- **Right Panel**: Circular layout with road labels
  - Same color and size encoding
  - Clear road name labels for identification
  - Better visibility of network structure

**Key Insights**:
- Identifies high-risk roads (red nodes)
- Shows flood propagation pathways
- Reveals network topology and connectivity patterns
- 42 nodes representing road segments
- Directed edges showing flood dependencies

### 2. Parameter Sensitivity Analysis (`parameter_sensitivity_analysis.png`)

**Purpose**: Analyzes how evidence count and prediction threshold affect model performance.

**Four Panels**:

#### **Top Left - F1-Score Heatmap**
- **Axes**: Evidence Count (rows) vs Prediction Threshold (columns)
- **Colors**: Green = high F1, Red = low F1
- **Values**: Exact F1-scores for each parameter combination
- **Insight**: Evidence=3, Threshold=0.1 achieves best performance (0.867)

#### **Top Right - Sample Count Heatmap**
- **Axes**: Same as F1-Score heatmap
- **Colors**: Blue intensity shows sample count
- **Values**: Number of evaluation samples
- **Insight**: Lower evidence counts provide more samples but not necessarily better performance

#### **Bottom Left - Performance vs Evidence Count**
- **Lines**: Precision, Recall, F1-Score trends
- **X-axis**: Evidence count (1, 2, 3)
- **Y-axis**: Performance score (0-1)
- **Insight**: Evidence=3 provides best balance of precision and recall

#### **Bottom Right - Performance vs Prediction Threshold**
- **Lines**: Precision, Recall, F1-Score trends
- **X-axis**: Prediction threshold (0.1, 0.3, 0.5)
- **Y-axis**: Performance score (0-1)
- **Insight**: Lower thresholds (0.1) achieve better overall performance

### 3. Performance Comparison (`performance_comparison.png`)

**Purpose**: Compares validation and test performance to assess model generalization.

**Four Panels**:

#### **Top Left - Validation vs Test Metrics**
- **Bars**: Side-by-side comparison of key metrics
- **Blue**: Validation performance (best configuration)
- **Red**: Test performance (final evaluation)
- **Values**: Exact scores labeled on bars
- **Insight**: Minimal performance drop from validation to test (good generalization)

#### **Top Right - Test Set Confusion Matrix**
- **Heatmap**: Actual vs Predicted classifications
- **Axes**: Predicted (columns) vs Actual (rows)
- **Values**: True Positives (35), False Positives (6), True Negatives (36), False Negatives (6)
- **Insight**: Balanced performance with low false alarm rate

#### **Bottom Left - Sample Count Comparison**
- **Bars**: Sample sizes for validation vs test
- **Values**: 86 validation samples, 83 test samples
- **Insight**: Similar sample sizes ensure reliable comparison

#### **Bottom Right - F1-Score Distribution**
- **Histogram**: Distribution of F1-scores across all validation configurations
- **Lines**: Best validation (blue dashed) vs test result (red solid)
- **Insight**: Test performance aligns well with validation results

### 4. Data Distribution Analysis (`data_distribution_analysis.png`)

**Purpose**: Analyzes the temporal and statistical properties of the train/validation/test splits.

**Four Panels**:

#### **Top Left - Temporal Distribution**
- **Lines**: Daily flood records over time
- **Colors**: Train (blue), Validation (green), Test (red)
- **X-axis**: Date timeline
- **Y-axis**: Number of flood records per day
- **Insight**: Clear temporal separation with no overlap between splits

#### **Top Right - Dataset Statistics**
- **Bars**: Comparative statistics across datasets
- **Metrics**: Records (÷10), Flood Days, Unique Roads
- **Insight**: Proportional distribution maintaining data characteristics

#### **Bottom Left - Road Frequency Distribution**
- **Histogram**: Number of roads vs their flood frequency
- **Statistics**: Total roads (73), Mean frequency (12.6), Max frequency (83)
- **Insight**: Most roads have low-to-moderate flood frequency

#### **Bottom Right - Yearly Distribution**
- **Stacked Bars**: Flood records by year and dataset split
- **Colors**: Same as temporal distribution
- **Insight**: Temporal progression from train (2015-2021) to validation (2021-2023) to test (2023-2024)

## Technical Features

### Visualization Functions Added

1. **`visualize_bayesian_network()`**
   - Network topology visualization
   - Node attribute encoding (color, size)
   - Multiple layout options (spring, circular)

2. **`visualize_parameter_sensitivity()`**
   - Heatmaps for parameter combinations
   - Line plots for trend analysis
   - Comprehensive sensitivity analysis

3. **`visualize_performance_comparison()`**
   - Validation vs test comparison
   - Confusion matrix visualization
   - Distribution analysis

4. **`visualize_data_distribution()`**
   - Temporal split visualization
   - Statistical summaries
   - Data quality assessment

5. **`generate_all_visualizations()`**
   - Orchestrates all visualization functions
   - Error handling and file management
   - Batch generation with progress tracking

### Usage Examples

#### Generate Visualizations for Existing Results
```bash
python generate_visualizations.py validation_focused_results_20250714_174125
```

#### Test Visualization Functions
```bash
python test_visualizations.py
```

#### Run Complete Analysis with Visualizations
```bash
python validation_focused_evaluation.py
```

### File Outputs

Each visualization generates two files:
- **PNG format**: High-resolution raster image (300 DPI)
- **PDF format**: Vector graphics for publications

### Key Insights from Visualizations

1. **Network Structure**: 42-node network with clear flood propagation patterns
2. **Optimal Parameters**: Evidence=3, Prediction threshold=0.1
3. **Model Generalization**: Strong validation-to-test performance consistency
4. **Data Quality**: Clean temporal splits with balanced characteristics

### Customization Options

The visualization functions support various customization options:
- **Color schemes**: Adjustable palettes and colormaps
- **Layout algorithms**: Multiple network layout options
- **Figure sizes**: Configurable subplot arrangements
- **Export formats**: PNG and PDF output formats
- **DPI settings**: High-resolution export (300 DPI)

### Dependencies

Required packages for visualizations:
```python
matplotlib >= 3.5.0
seaborn >= 0.11.0
networkx >= 2.8.0
pandas >= 1.5.0
numpy >= 1.21.0
```

### Best Practices

1. **Non-interactive Backend**: Uses 'Agg' backend for server environments
2. **Memory Management**: Proper plot closing with `plt.close()`
3. **Error Handling**: Graceful handling of missing data fields
4. **Consistent Styling**: Unified color schemes and formatting
5. **Comprehensive Labeling**: Clear titles, axes, and legends

The visualizations provide comprehensive insights into model performance, network structure, and data characteristics, supporting both technical analysis and presentation purposes.