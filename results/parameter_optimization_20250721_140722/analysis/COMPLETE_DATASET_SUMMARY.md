# Complete Top Configurations Dataset Summary

## Dataset Overview
This dataset contains the complete information for all top-performing parameter configurations,
including ALL original columns from the parameter optimization results plus metadata.

## Dataset Structure
- **Total Configurations**: 17
- **Total Columns**: 25
- **Categories**: 5

## Configuration Breakdown

- **Highest F1**: 5 configurations
- **Highest Precision**: 3 configurations
- **Highest Recall**: 3 configurations
- **Best Balanced**: 3 configurations
- **Most Robust**: 3 configurations

## Column Information

### Metadata Columns
- `category`: Type of optimization criteria (Highest F1, Highest Precision, etc.)
- `rank`: Rank within the category (1 = best in category)

### Parameter Columns (7)
- `occ_thr`: [2.0, 4.0]
- `edge_thr`: [1.0, 3.0]
- `weight_thr`: [0.2, 0.3, 0.4]
- `evidence_count`: [1.0]
- `pred_threshold`: [0.1]
- `neg_pos_ratio`: [1.0, 1.5, 2.0]
- `marginal_prob_threshold`: [0.03, 0.08]

### Performance Metrics (4)
- `precision`: Range [0.679, 0.902], Mean: 0.801
- `recall`: Range [0.786, 0.967], Mean: 0.885
- `f1_score`: Range [0.728, 0.875], Mean: 0.837
- `accuracy`: Range [0.779, 0.867], Mean: 0.834

### Confusion Matrix (4)
- `tp`: Range [55, 58]
- `fp`: Range [6, 26]
- `tn`: Range [32, 108]
- `fn`: Range [2, 15]

### Sample Information (3)
- `total_samples`: Range [115, 204]
- `positive_samples`: Range [60, 70]
- `negative_samples`: Range [55, 134]

### Model & Data Information
- `valid_days`: Range [13, 13]
- `total_days`: Range [29, 29]
- `network_nodes`: Range [35, 53]
- `negative_candidates_count`: Range [9, 27]
- `runtime_seconds`: Range [0.17s, 3.38s]

## Top Performers Summary

### Best Overall Configuration (Highest F1)
- **F1 Score**: 0.875
- **Precision**: 0.824
- **Recall**: 0.933
- **Accuracy**: 0.861
- **True Positives**: 56
- **False Positives**: 12
- **Total Samples**: 115
- **Network Nodes**: 35
- **Runtime**: 0.29 seconds

## Usage Notes
- All configurations have been filtered to exclude overfitting cases
- Only configurations with negative_candidates_count ≥ 9 are included
- Perfect precision or recall (1.0) configurations were excluded
- This dataset provides complete transparency for parameter selection decisions
