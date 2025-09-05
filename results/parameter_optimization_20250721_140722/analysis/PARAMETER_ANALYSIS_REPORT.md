# Parameter Optimization Analysis Report

## Executive Summary
This report analyzes the parameter optimization results for the Charleston Flood Prediction Bayesian Network model. We tested 5,760 parameter combinations and identified the top-performing configurations for different use cases.

## Parameter Definitions

### Occurrence Threshold (`occ_thr`)
**Description**: Minimum number of flood occurrences required for a road to be included in the Bayesian network

**Purpose**: Controls network size and data quality - higher values create smaller, more reliable networks

**Impact**: Lower values include more roads but may introduce noise; higher values focus on frequently flooded roads

**Range Tested**: [2, 3, 4, 5]

**Typically Good Values**: [3, 4, 5]

---

### Edge Threshold (`edge_thr`)
**Description**: Minimum number of co-occurrences required to create an edge between two roads in the network

**Purpose**: Determines the strength of relationships between roads that get modeled

**Impact**: Lower values create more connected networks; higher values focus on strongest relationships

**Range Tested**: [1, 2, 3]

**Typically Good Values**: [2, 3]

---

### Weight Threshold (`weight_thr`)
**Description**: Minimum conditional probability required for an edge to be retained in the network

**Purpose**: Filters out weak probabilistic relationships between roads

**Impact**: Higher values keep only strong dependencies; lower values preserve more relationships

**Range Tested**: [0.2, 0.3, 0.4, 0.5]

**Typically Good Values**: [0.2, 0.3]

---

### Evidence Count (`evidence_count`)
**Description**: Number of roads that must be observed as flooded to trigger predictions for other roads

**Purpose**: Controls prediction sensitivity - how much evidence is needed before making predictions

**Impact**: Lower values make more aggressive predictions; higher values require more confirmation

**Range Tested**: [1, 2, 3, 4]

**Typically Good Values**: [1, 2]

---

### Prediction Threshold (`pred_threshold`)
**Description**: Minimum probability required for the model to predict a road will flood

**Purpose**: Controls the precision-recall trade-off for final predictions

**Impact**: Lower values increase recall but decrease precision; higher values do the opposite

**Range Tested**: [0.1, 0.2, 0.3, 0.4, 0.5]

**Typically Good Values**: [0.1, 0.2]

---

### Negative-to-Positive Ratio (`neg_pos_ratio`)
**Description**: Ratio of negative (non-flood) to positive (flood) samples used in training

**Purpose**: Balances the dataset to prevent bias toward the majority class

**Impact**: Higher ratios include more negative examples; lower ratios focus more on flood patterns

**Range Tested**: [1.0, 1.5, 2.0]

**Typically Good Values**: [1.0, 1.5]

---

### Marginal Probability Threshold (`marginal_prob_threshold`)
**Description**: Minimum marginal probability for a road to be included in negative sampling

**Purpose**: Controls which roads are considered as potential negative examples

**Impact**: Higher values focus on roads with higher baseline flood probability as negatives

**Range Tested**: [0.03, 0.05, 0.08]

**Typically Good Values**: [0.05, 0.08]

---

## Top Performing Configurations

### Highest F1

**Configuration 1:**
- F1 Score: 0.875
- Precision: 0.824
- Recall: 0.933
- Negative Candidates: 9
- Parameters:
  - occ_thr: 4.0
  - edge_thr: 3.0
  - weight_thr: 0.2
  - evidence_count: 1.0
  - pred_threshold: 0.1
  - neg_pos_ratio: 1.0
  - marginal_prob_threshold: 0.08

**Configuration 2:**
- F1 Score: 0.875
- Precision: 0.824
- Recall: 0.933
- Negative Candidates: 9
- Parameters:
  - occ_thr: 4.0
  - edge_thr: 3.0
  - weight_thr: 0.3
  - evidence_count: 1.0
  - pred_threshold: 0.1
  - neg_pos_ratio: 1.0
  - marginal_prob_threshold: 0.08

**Configuration 3:**
- F1 Score: 0.875
- Precision: 0.824
- Recall: 0.933
- Negative Candidates: 9
- Parameters:
  - occ_thr: 4.0
  - edge_thr: 3.0
  - weight_thr: 0.4
  - evidence_count: 1.0
  - pred_threshold: 0.1
  - neg_pos_ratio: 1.0
  - marginal_prob_threshold: 0.08

---

### Highest Precision

**Configuration 1:**
- F1 Score: 0.840
- Precision: 0.902
- Recall: 0.786
- Negative Candidates: 11
- Parameters:
  - occ_thr: 2.0
  - edge_thr: 1.0
  - weight_thr: 0.2
  - evidence_count: 1.0
  - pred_threshold: 0.1
  - neg_pos_ratio: 1.0
  - marginal_prob_threshold: 0.03

**Configuration 2:**
- F1 Score: 0.840
- Precision: 0.902
- Recall: 0.786
- Negative Candidates: 11
- Parameters:
  - occ_thr: 2.0
  - edge_thr: 1.0
  - weight_thr: 0.3
  - evidence_count: 1.0
  - pred_threshold: 0.1
  - neg_pos_ratio: 1.0
  - marginal_prob_threshold: 0.03

**Configuration 3:**
- F1 Score: 0.840
- Precision: 0.902
- Recall: 0.786
- Negative Candidates: 11
- Parameters:
  - occ_thr: 2.0
  - edge_thr: 1.0
  - weight_thr: 0.4
  - evidence_count: 1.0
  - pred_threshold: 0.1
  - neg_pos_ratio: 1.0
  - marginal_prob_threshold: 0.03

---

### Highest Recall

**Configuration 1:**
- F1 Score: 0.823
- Precision: 0.716
- Recall: 0.967
- Negative Candidates: 9
- Parameters:
  - occ_thr: 4.0
  - edge_thr: 1.0
  - weight_thr: 0.2
  - evidence_count: 1.0
  - pred_threshold: 0.1
  - neg_pos_ratio: 1.0
  - marginal_prob_threshold: 0.08

**Configuration 2:**
- F1 Score: 0.832
- Precision: 0.740
- Recall: 0.950
- Negative Candidates: 9
- Parameters:
  - occ_thr: 4.0
  - edge_thr: 1.0
  - weight_thr: 0.3
  - evidence_count: 1.0
  - pred_threshold: 0.1
  - neg_pos_ratio: 1.0
  - marginal_prob_threshold: 0.08

**Configuration 3:**
- F1 Score: 0.809
- Precision: 0.704
- Recall: 0.950
- Negative Candidates: 9
- Parameters:
  - occ_thr: 4.0
  - edge_thr: 1.0
  - weight_thr: 0.3
  - evidence_count: 1.0
  - pred_threshold: 0.1
  - neg_pos_ratio: 1.5
  - marginal_prob_threshold: 0.08

---

### Best Balanced

**Configuration 1:**
- F1 Score: 0.875
- Precision: 0.824
- Recall: 0.933
- Negative Candidates: 9
- Parameters:
  - occ_thr: 4.0
  - edge_thr: 3.0
  - weight_thr: 0.2
  - evidence_count: 1.0
  - pred_threshold: 0.1
  - neg_pos_ratio: 1.0
  - marginal_prob_threshold: 0.08

**Configuration 2:**
- F1 Score: 0.875
- Precision: 0.824
- Recall: 0.933
- Negative Candidates: 9
- Parameters:
  - occ_thr: 4.0
  - edge_thr: 3.0
  - weight_thr: 0.3
  - evidence_count: 1.0
  - pred_threshold: 0.1
  - neg_pos_ratio: 1.0
  - marginal_prob_threshold: 0.08

**Configuration 3:**
- F1 Score: 0.875
- Precision: 0.824
- Recall: 0.933
- Negative Candidates: 9
- Parameters:
  - occ_thr: 4.0
  - edge_thr: 3.0
  - weight_thr: 0.4
  - evidence_count: 1.0
  - pred_threshold: 0.1
  - neg_pos_ratio: 1.0
  - marginal_prob_threshold: 0.08

---

### Most Robust

**Configuration 1:**
- F1 Score: 0.780
- Precision: 0.775
- Recall: 0.786
- Negative Candidates: 27
- Parameters:
  - occ_thr: 2.0
  - edge_thr: 1.0
  - weight_thr: 0.2
  - evidence_count: 1.0
  - pred_threshold: 0.1
  - neg_pos_ratio: 1.0
  - marginal_prob_threshold: 0.08

**Configuration 2:**
- F1 Score: 0.759
- Precision: 0.733
- Recall: 0.786
- Negative Candidates: 27
- Parameters:
  - occ_thr: 2.0
  - edge_thr: 1.0
  - weight_thr: 0.2
  - evidence_count: 1.0
  - pred_threshold: 0.1
  - neg_pos_ratio: 1.5
  - marginal_prob_threshold: 0.08

**Configuration 3:**
- F1 Score: 0.728
- Precision: 0.679
- Recall: 0.786
- Negative Candidates: 27
- Parameters:
  - occ_thr: 2.0
  - edge_thr: 1.0
  - weight_thr: 0.2
  - evidence_count: 1.0
  - pred_threshold: 0.1
  - neg_pos_ratio: 2.0
  - marginal_prob_threshold: 0.08

---

## Parameter Selection Patterns

Based on analysis of top-performing configurations:

### Occurrence Threshold (`occ_thr`)
- Most common value: 2
- Median value: 3.000
- Value distribution: {2: 270, 3: 238, 4: 125}

### Edge Threshold (`edge_thr`)
- Most common value: 3
- Median value: 2.000
- Value distribution: {3: 233, 2: 227, 1: 173}

### Weight Threshold (`weight_thr`)
- Most common value: 0.4
- Median value: 0.400
- Value distribution: {0.4: 163, 0.5: 158, 0.2: 157, 0.3: 155}

### Evidence Count (`evidence_count`)
- Most common value: 1
- Median value: 2.000
- Value distribution: {1: 206, 2: 173, 3: 130, 4: 124}

### Prediction Threshold (`pred_threshold`)
- Most common value: 0.1
- Median value: 0.100
- Value distribution: {0.1: 586, 0.2: 42, 0.3: 5}

### Negative-to-Positive Ratio (`neg_pos_ratio`)
- Most common value: 1.0
- Median value: 1.500
- Value distribution: {1.0: 262, 1.5: 215, 2.0: 156}

### Marginal Probability Threshold (`marginal_prob_threshold`)
- Most common value: 0.08
- Median value: 0.080
- Value distribution: {0.08: 336, 0.05: 207, 0.03: 90}

## Deployment Recommendations

### For Emergency Response (High Recall Priority)
Use configurations that maximize recall to ensure no floods are missed, even if some false alarms occur.
Recommended parameters based on analysis:
- Lower pred_threshold (0.1-0.2)
- Lower evidence_count (1-2)
- Moderate occ_thr (3-4)

### For Planning and Preparedness (High Precision Priority)
Use configurations that minimize false positives for resource allocation and long-term planning.
Recommended parameters:
- Higher pred_threshold (0.3-0.4)
- Higher evidence_count (2-3)
- Higher weight_thr (0.3-0.4)

### For Balanced Operations (High F1 Priority)
Use configurations that balance precision and recall for general-purpose flood prediction.
Recommended parameters from best F1 configurations:
- occ_thr: 3-4
- edge_thr: 2-3
- weight_thr: 0.2-0.3
- evidence_count: 1-2
- pred_threshold: 0.1-0.2

### Robustness Considerations
For production deployment, prioritize configurations with:
- negative_candidates_count ≥ 15 (sufficient validation data)
- Consistent performance across different parameter nearby values
- Reasonable computational requirements (network_nodes < 100)

## Implementation Notes
1. Start with the highest F1 configuration for initial deployment
2. A/B test different thresholds based on operational needs
3. Monitor performance on new data and adjust parameters accordingly
4. Consider ensemble approaches using multiple top configurations
