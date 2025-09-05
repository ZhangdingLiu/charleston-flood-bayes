# Validation-Focused Evaluation

## Overview

`validation_focused_evaluation.py` is an improved version of the Bayesian network flood prediction evaluation, based on `pilot_conservative_evaluation.py` with key methodological improvements.

## Key Improvements

### 1. **Proper Temporal Data Split** ✅
- **Before**: Random record-based split causing data leakage
- **After**: Time-based split by flood days (6:2:2 = Train:Valid:Test)
- **Benefit**: Avoids same-day events in different splits, more realistic evaluation

### 2. **Fixed Evidence Strategy** ✅
- **Before**: Proportional evidence selection (evidence_ratio * total_floods)
- **After**: Fixed evidence counts [1, 2, 3]
- **Benefit**: Consistent sample sizes, more realistic application scenario

### 3. **Simplified Parameter Grid** ✅
- **Focus on key parameters**: Evidence count and prediction threshold
- **Fixed network parameters**: occ_thr=3, edge_thr=2, weight_thr=0.3
- **Reduced combinations**: 9 total (3×3) vs hundreds in pilot study

### 4. **Validation-Based Parameter Selection** ✅
- **Proper methodology**: Tune on validation set, evaluate on test set
- **Prevents overfitting**: No parameter tuning on test data
- **Reliable results**: Final test performance is unbiased

## Parameters Configuration

### Network Building (Fixed)
```python
occ_thr = 3        # Road occurrence threshold
edge_thr = 2       # Edge co-occurrence threshold  
weight_thr = 0.3   # Edge weight threshold
max_parents = 2    # Maximum parent nodes
alpha = 1.0        # Laplace smoothing
```

### Parameter Grid (Tunable)
```python
evidence_counts = [1, 2, 3]      # Fixed evidence counts
pred_thresholds = [0.1, 0.3, 0.5] # Prediction thresholds
prob_threshold = 0.05            # Negative sampling threshold (fixed)
neg_pos_ratio = 1.0              # 1:1 negative to positive ratio
```

## Usage

### Quick Test
```bash
# Test core functions
python test_validation_script.py
```

### Full Evaluation
```bash
# Run complete validation experiment
python validation_focused_evaluation.py
```

## Expected Output

### Data Split Information
```
Total flood days: 144
Train: 86 days, 621 records (2015-08-31 to 2021-11-06)
Valid: 29 days, 145 records (2021-11-07 to 2023-07-23)  
Test:  29 days, 157 records (2023-08-02 to 2024-10-19)
```

### Network Information
```
Network parameters: {'occ_thr': 3, 'edge_thr': 2, 'weight_thr': 0.3}
Network built: 42 nodes
Marginal prob range: 0.0349 - 0.4767
```

### Parameter Selection Results
```
Top 5 configurations:
Rank Evid Pred  F1     Prec   Rec    Samples
1    2    0.1   0.834  0.752  0.931  168    
2    1    0.1   0.821  0.723  0.945  186    
3    3    0.1   0.798  0.781  0.816  142    
...
```

## File Structure

```
validation_focused_evaluation.py     # Main evaluation script
test_validation_script.py           # Test script
validation_focused_README.md        # This documentation

# Output files (generated after run)
validation_focused_results_YYYYMMDD_HHMMSS/
├── validation_results.csv          # All validation combinations
├── experiment_summary.json         # Best parameters and final results
└── (other analysis files)
```

## Methodological Advantages

1. **No Data Leakage**: Temporal split ensures no same-day events across splits
2. **Realistic Evidence**: Fixed counts match real monitoring scenarios  
3. **Stable Evaluation**: Consistent sample sizes across different flood scales
4. **Proper Validation**: Unbiased parameter selection methodology
5. **Reproducible**: Fixed random seed and deterministic splits

## Comparison with Pilot Study

| Aspect | Pilot Study | Validation-Focused |
|--------|-------------|-------------------|
| Data Split | Random records | Temporal by days |
| Evidence Strategy | Proportional ratio | Fixed counts |
| Parameter Grid | 100+ combinations | 9 combinations |
| Sample Balance | Highly variable | More consistent |
| Evaluation Bias | Test set tuning | Validation tuning |
| Runtime | 30-60 minutes | ~10-15 minutes |

## Expected Results

Based on pilot study insights, expect:
- **Best evidence count**: 1-2 (most stable)
- **Best prediction threshold**: 0.1-0.3 (good balance)
- **F1-Score range**: 0.7-0.9 (depending on configuration)
- **Sample count**: 150-200 per configuration (more consistent)

## Next Steps

1. Run the validation experiment
2. Analyze validation results for parameter sensitivity
3. Use selected parameters for final test evaluation
4. Compare results with pilot study findings
5. Consider additional evidence selection strategies if needed