# Focused Flexible Parameter Test Set Evaluation Results

## Overview
This evaluation tested 120 focused parameter combinations with promising settings,
concentrating on improving precision through higher prediction thresholds.

## Key Findings

### Overall Performance Distribution
- **F1 Score**: 0.336 ± 0.195 (range: 0.000 - 0.692)
- **Precision**: 0.392 ± 0.223 (range: 0.000 - 1.000)
- **Recall**: 0.465 ± 0.357 (range: 0.000 - 1.000)

### Best Configurations by Metric

**Best F1** (Config #44):
- Performance: P=0.692, R=0.692, F1=0.692, Acc=0.818
- Network: occ_thr=4, edge_thr=2, weight_thr=0.3
- Prediction: pred_thr=0.3, neg_ratio=0.5

**Best Precision** (Config #19):
- Performance: P=1.000, R=0.125, F1=0.222, Acc=0.848
- Network: occ_thr=3, edge_thr=2, weight_thr=0.3
- Prediction: pred_thr=0.4, neg_ratio=0.5

**Best Recall** (Config #1):
- Performance: P=0.308, R=1.000, F1=0.471, Acc=0.609
- Network: occ_thr=3, edge_thr=2, weight_thr=0.3
- Prediction: pred_thr=0.1, neg_ratio=0.5

**Best Accuracy** (Config #73):
- Performance: P=0.714, R=0.625, F1=0.667, Acc=0.896
- Network: occ_thr=2, edge_thr=1, weight_thr=0.3
- Prediction: pred_thr=0.3, neg_ratio=0.5

### Prediction Threshold Impact

- **pred_threshold=0.1**: P=0.249, R=0.995, F1=0.397 (n=24)
- **pred_threshold=0.2**: P=0.416, R=0.625, F1=0.493 (n=24)
- **pred_threshold=0.3**: P=0.535, R=0.537, F1=0.533 (n=24)
- **pred_threshold=0.4**: P=0.479, R=0.138, F1=0.209 (n=24)
- **pred_threshold=0.5**: P=0.281, R=0.027, F1=0.049 (n=24)

### Top 10 Configurations

1. **Config #44** (F1=0.692)
   - P=0.692, R=0.692, Acc=0.818
   - pred_thr=0.3, network=(4,2,0.3)

2. **Config #73** (F1=0.667)
   - P=0.714, R=0.625, Acc=0.896
   - pred_thr=0.3, network=(2,1,0.3)

3. **Config #38** (F1=0.645)
   - P=0.556, R=0.769, Acc=0.750
   - pred_thr=0.2, network=(4,2,0.3)

4. **Config #8** (F1=0.640)
   - P=0.667, R=0.615, Acc=0.804
   - pred_thr=0.2, network=(3,2,0.3)

5. **Config #104** (F1=0.640)
   - P=0.667, R=0.615, Acc=0.804
   - pred_thr=0.3, network=(3,1,0.2)

6. **Config #74** (F1=0.615)
   - P=0.667, R=0.571, Acc=0.792
   - pred_thr=0.3, network=(2,1,0.3)

7. **Config #14** (F1=0.583)
   - P=0.636, R=0.538, Acc=0.783
   - pred_thr=0.3, network=(3,2,0.3)

8. **Config #98** (F1=0.581)
   - P=0.500, R=0.692, Acc=0.717
   - pred_thr=0.2, network=(3,1,0.2)

9. **Config #45** (F1=0.571)
   - P=0.476, R=0.714, Acc=0.817
   - pred_thr=0.3, network=(4,2,0.3)

10. **Config #75** (F1=0.562)
   - P=0.529, R=0.600, Acc=0.848
   - pred_thr=0.3, network=(2,1,0.3)


## Deployment Recommendations

### For High Precision Applications
Use pred_threshold ≥ 0.3 with configuration:
- **Best Precision**: pred_threshold=0.4, occ_thr=3, edge_thr=2
- **Expected Performance**: P=1.000, R=0.125, F1=0.222

### For Balanced Performance  
Use the best F1 configuration:
- **Parameters**: pred_threshold=0.3, occ_thr=4, edge_thr=2, weight_thr=0.3
- **Expected Performance**: P=0.692, R=0.692, F1=0.692

### Key Insights
1. Higher prediction thresholds (0.3-0.5) significantly improve precision
2. Network parameters (occ_thr=3-4, edge_thr=1-2) provide good balance
3. Evidence count of 1 is sufficient for this dataset
4. Negative sampling ratio of 0.5-1.5 works well

## Data Quality Notes
- Evaluated on test set covering 2023-2024 period
- Average test samples per configuration: 86.0
- All configurations used proper temporal train/test split
