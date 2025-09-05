# Specified Parameter Test Set Evaluation Results

## Overview
This evaluation tested the user-specified parameter combinations using train+validation data for training
and evaluating on the held-out test set.

### Parameter Configuration
- **occ_thr**: 2, 4
- **edge_thr**: 3 (fixed)
- **weight_thr**: 0.2 (fixed)
- **evidence_count**: 1 (fixed)
- **pred_threshold**: 0.1, 0.2, 0.3, 0.4
- **neg_pos_ratio**: 1.0 (fixed)
- **marginal_prob_threshold**: 0.08 (fixed)

## Key Findings

### Overall Performance Distribution
- **F1 Score**: 0.382 ± 0.147 (range: 0.067 - 0.480)
- **Precision**: 0.480 ± 0.127 (range: 0.299 - 0.667)
- **Recall**: 0.478 ± 0.353 (range: 0.036 - 1.000)

### Best Configurations by Metric

**Best F1** (Config #6):
- Performance: P=0.500, R=0.462, F1=0.480, Acc=0.701
- Parameters: occ_thr=4, pred_threshold=0.2
- Network: 40 nodes, 50 edges

**Best Precision** (Config #8):
- Performance: P=0.667, R=0.154, F1=0.250, Acc=0.724
- Parameters: occ_thr=4, pred_threshold=0.4
- Network: 40 nodes, 50 edges

**Best Recall** (Config #1):
- Performance: P=0.301, R=1.000, F1=0.463, Acc=0.301
- Parameters: occ_thr=2, pred_threshold=0.1
- Network: 57 nodes, 50 edges

**Best Accuracy** (Config #7):
- Performance: P=0.556, R=0.385, F1=0.455, Acc=0.724
- Parameters: occ_thr=4, pred_threshold=0.3
- Network: 40 nodes, 50 edges

### Parameter Impact Analysis

**Network Occurrence Threshold (occ_thr) Impact:**
- occ_thr=2: F1=0.352, P=0.455, R=0.455, Avg_nodes=57.0
- occ_thr=4: F1=0.411, P=0.505, R=0.500, Avg_nodes=40.0

**Prediction Threshold Impact:**
- pred_threshold=0.1: F1=0.461, P=0.300, R=1.000
- pred_threshold=0.2: F1=0.462, P=0.481, R=0.445
- pred_threshold=0.3: F1=0.445, P=0.556, R=0.371
- pred_threshold=0.4: F1=0.158, P=0.583, R=0.095

### All Configuration Results

**Config #1** (occ_thr=2, pred_thr=0.1):
- Performance: P=0.301, R=1.000, F1=0.463, Acc=0.301
- Network: 57 nodes, 50 edges
- Test samples: 93, Valid test days: 22/29

**Config #2** (occ_thr=2, pred_thr=0.2):
- Performance: P=0.462, R=0.429, F1=0.444, Acc=0.677
- Network: 57 nodes, 50 edges
- Test samples: 93, Valid test days: 22/29

**Config #3** (occ_thr=2, pred_thr=0.3):
- Performance: P=0.556, R=0.357, F1=0.435, Acc=0.720
- Network: 57 nodes, 50 edges
- Test samples: 93, Valid test days: 22/29

**Config #4** (occ_thr=2, pred_thr=0.4):
- Performance: P=0.500, R=0.036, F1=0.067, Acc=0.699
- Network: 57 nodes, 50 edges
- Test samples: 93, Valid test days: 22/29

**Config #5** (occ_thr=4, pred_thr=0.1):
- Performance: P=0.299, R=1.000, F1=0.460, Acc=0.299
- Network: 40 nodes, 50 edges
- Test samples: 87, Valid test days: 22/29

**Config #6** (occ_thr=4, pred_thr=0.2):
- Performance: P=0.500, R=0.462, F1=0.480, Acc=0.701
- Network: 40 nodes, 50 edges
- Test samples: 87, Valid test days: 22/29

**Config #7** (occ_thr=4, pred_thr=0.3):
- Performance: P=0.556, R=0.385, F1=0.455, Acc=0.724
- Network: 40 nodes, 50 edges
- Test samples: 87, Valid test days: 22/29

**Config #8** (occ_thr=4, pred_thr=0.4):
- Performance: P=0.667, R=0.154, F1=0.250, Acc=0.724
- Network: 40 nodes, 50 edges
- Test samples: 87, Valid test days: 22/29


## Training Data Impact
- **Training Data**: Train + Validation combined (766 records)
- **Test Data**: Independent test set (157 records)
- **Temporal Split**: Proper time-based split to avoid data leakage

## Deployment Recommendations

### Based on Test Results:
1. **Best Overall**: Config #6 - F1=0.480
2. **High Precision**: Config #8 - P=0.667
3. **High Recall**: Config #1 - R=1.000

### Key Insights:
- Training on combined train+valid data vs train-only shows model performance with more data
- pred_threshold has significant impact on precision-recall trade-off
- Network size (occ_thr) affects model complexity and performance

## Data Quality Notes
- Test set covers 29 unique flood days
- Average test samples per configuration: 90.0
- Consistent evaluation across all configurations
