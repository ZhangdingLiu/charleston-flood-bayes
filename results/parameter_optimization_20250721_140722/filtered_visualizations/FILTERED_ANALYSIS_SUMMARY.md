# Filtered Parameter Optimization Results Summary

## Filtering Criteria Applied
- negative_candidates_count >= 9 (sufficient negative samples)
- precision < 1.0 (exclude perfect precision, likely overfitting)
- recall < 1.0 (exclude perfect recall, likely overfitting)

## Overall Statistics
- **Total filtered configurations**: 3165
- **F1 Score range**: 0.000 - 0.875
- **Precision range**: 0.000 - 0.970
- **Recall range**: 0.000 - 0.978

## Best Configuration (Highest F1)
- **F1 Score**: 0.875
- **Precision**: 0.824
- **Recall**: 0.933
- **Parameters**: 
  - occ_thr: 4.0
  - edge_thr: 3.0
  - weight_thr: 0.2
  - evidence_count: 1.0
  - pred_threshold: 0.1
  - neg_pos_ratio: 1.0
  - marginal_prob_threshold: 0.08
- **Negative candidates**: 9.0

## Constraint Satisfaction Analysis
- **Precision ≥ 0.8**: 1096 / 3165 (34.6%)
- **Recall ≥ 0.8**: 663 / 3165 (20.9%)
- **F1 Score ≥ 0.7**: 725 / 3165 (22.9%)
- **All constraints satisfied**: 74 / 3165 (2.3%)

## Generated Visualizations
1. **precision_recall_f1_3d_filtered.png** - 3D performance distribution
2. **parameter_heatmaps_filtered.png** - Parameter combination performance heatmaps
3. **parameter_sensitivity_filtered.png** - Parameter sensitivity analysis
4. **pareto_frontier_filtered.png** - Precision-recall trade-off analysis
5. **constraint_filtering_filtered.png** - Constraint filtering results

This filtered analysis provides a more robust view of parameter performance by excluding 
configurations that may have achieved perfect scores due to insufficient data or 
overly restrictive parameter settings.
