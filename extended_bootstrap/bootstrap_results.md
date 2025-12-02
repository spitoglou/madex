# Extended Bootstrap Statistical Validation Analysis

**Analysis Date:** 2025-12-02 10:11:38

## Overview

This analysis validates the MADEX metric using extended clinical scenarios with:
- **50 data points per scenario** (vs. 8 in original)
- **Time-series ordered data** resembling realistic glucose patterns
- **Controlled model bias patterns:**
  - Model A: underestimates below euglycemic (<70 mg/dL), overestimates above (>180 mg/dL)
  - Model B: overestimates below euglycemic, underestimates above
  - Both models have identical absolute errors

## Analysis Objectives

1. Generate bootstrap confidence intervals for MADEX across extended clinical scenarios (50 points each)
2. Compare MADEX performance against traditional metrics (RMSE, MAE)
3. Assess clinical risk using Clarke Error Grid Analysis
4. Evaluate statistical significance of model differences with larger sample sizes
5. Analyze hypoglycemia and hyperglycemia detection sensitivity
6. Validate MADEX behavior with controlled model bias patterns

---

## Clinical Scenarios

| Scenario | Name | Glucose Range | Data Points | Risk Level |
|----------|------|---------------|-------------|------------|
| A | Hypoglycemia Detection | 30-85 mg/dL | 50 | CRITICAL |
| B | Hyperglycemia Management | 150-325 mg/dL | 50 | HIGH |
| C | Postprandial Response | 95-220 mg/dL | 50 | MODERATE |
| D | Dawn Phenomenon | 90-160 mg/dL | 50 | LOW |
| E | Exercise Response | 60-140 mg/dL | 50 | MODERATE |
| F | Measurement Noise | 90-131 mg/dL | 50 | LOW |
| G | Mixed Clinical | 35-300 mg/dL | 50 | CRITICAL |
| H | Extreme Cases | 20-450 mg/dL | 50 | CRITICAL |

---

## Bootstrap Results

### MADEX Confidence Intervals (95%)

| Scenario | Model A MADEX (95% CI) | Model B MADEX (95% CI) | Winner |
|----------|------------------------|------------------------|--------|
| A | 345.21 (324.82-368.19) | 1588.45 (1280.89-1920.73) | Model A |
| B | 665.02 (396.51-994.63) | 79117.22 (46154.77-118030.61) | Model A |
| C | 1199.95 (866.22-1634.91) | 5088.76 (3283.23-7038.71) | Model A |
| D | 1060.12 (869.47-1270.65) | 891.85 (757.58-1053.45) | Model B |
| E | 1691.67 (1302.76-2121.65) | 2318.34 (1732.49-2988.49) | Model A |
| F | 1123.99 (985.22-1269.93) | 1165.66 (1025.54-1308.36) | Model A |
| G | 1082.89 (722.93-1500.43) | 23854.25 (14624.09-33896.29) | Model A |
| H | 798.77 (446.74-1251.84) | 307123.53 (135992.80-518215.24) | Model A |


### Traditional Metrics Comparison

| Scenario | RMSE_A (95% CI) | RMSE_B (95% CI) | MAE_A (95% CI) | MAE_B (95% CI) |
|----------|-----------------|-----------------|----------------|----------------|
| A | 26.27 (24.81-27.83) | 26.24 (24.60-27.62) | 25.71 (24.13-27.28) | 25.68 (24.17-27.28) |
| B | 60.01 (55.79-64.41) | 59.91 (55.49-64.28) | 58.43 (54.36-62.61) | 58.26 (54.22-62.22) |
| C | 38.28 (35.34-41.12) | 38.21 (35.51-40.97) | 36.85 (34.07-39.75) | 36.91 (34.09-39.66) |
| D | 29.68 (28.34-30.94) | 29.74 (28.37-31.03) | 29.32 (27.92-30.70) | 29.31 (28.01-30.70) |
| E | 37.45 (36.25-38.64) | 37.42 (36.20-38.59) | 37.13 (35.98-38.38) | 37.15 (35.90-38.36) |
| F | 32.70 (31.72-33.76) | 32.71 (31.72-33.68) | 32.50 (31.43-33.55) | 32.52 (31.45-33.55) |
| G | 48.60 (44.83-52.75) | 48.48 (44.57-52.36) | 46.65 (42.83-50.72) | 46.65 (43.12-50.46) |
| H | 62.37 (53.95-70.40) | 62.50 (54.09-70.90) | 56.23 (48.82-63.76) | 56.02 (48.73-63.49) |


### Clarke Error Grid Risk Scores

| Scenario | Model A Risk | Model B Risk | Lower Risk Model |
|----------|--------------|--------------|------------------|
| A | 11 | 181 | Model A |
| B | 50 | 50 | Tie |
| C | 50 | 50 | Tie |
| D | 49 | 49 | Tie |
| E | 47 | 77 | Model A |
| F | 49 | 49 | Tie |
| G | 37 | 92 | Model A |
| H | 32 | 82 | Model A |

---

## Statistical Significance Testing

### Wilcoxon Signed-Rank Test Results

| Metric | Statistic | p-value | Cohen's d | Interpretation |
|--------|-----------|---------|-----------|----------------|
| MADEX | 2.00 | 0.0234 | -0.49 | Significant difference |
| RMSE | 13.00 | 0.5469 | 0.21 | No significant difference |
| MAE | 14.00 | 0.6406 | 0.39 | No significant difference |


### Model Comparison Summary

| Scenario | MADEX Winner | RMSE Winner | MAE Winner | CI Overlap |
|----------|--------------|-------------|------------|------------|
| A | Model A | Model B | Model B | No |
| B | Model A | Model B | Model B | No |
| C | Model A | Model B | Model A | No |
| D | Model B | Model A | Model B | Yes |
| E | Model A | Model B | Model A | Yes |
| F | Model A | Model A | Model A | Yes |
| G | Model A | Model B | Model B | No |
| H | Model A | Model A | Model B | No |

---

## Clinical Detection Sensitivity Analysis

### Hypoglycemia Detection (<70 mg/dL)

| Scenario | Model A Sensitivity | Model B Sensitivity | Better Model |
|----------|---------------------|---------------------|--------------|
| A | 1.00 | 0.15 | Model A |
| B | N/A | N/A | No hypo points |
| C | N/A | N/A | No hypo points |
| D | N/A | N/A | No hypo points |
| E | 1.00 | 0.00 | Model A |
| F | N/A | N/A | No hypo points |
| G | 1.00 | 0.15 | Model A |
| H | 1.00 | 0.44 | Model A |

### Hyperglycemia Detection (>180 mg/dL)

| Scenario | Model A Sensitivity | Model B Sensitivity | Better Model |
|----------|---------------------|---------------------|--------------|
| A | N/A | N/A | No hyper points |
| B | 1.00 | 0.50 | Model A |
| C | 1.00 | 0.00 | Model A |
| D | N/A | N/A | No hyper points |
| E | N/A | N/A | No hyper points |
| F | N/A | N/A | No hyper points |
| G | 1.00 | 0.61 | Model A |
| H | 1.00 | 0.90 | Model A |

---

## Conclusions

### Key Findings

1. **Bootstrap Validation**: 1000 bootstrap samples provide robust confidence intervals for MADEX
2. **Model Bias Patterns**: The controlled bias patterns (Model A vs Model B) are clearly detected by MADEX
3. **Clinical Relevance**: MADEX appropriately penalizes clinically dangerous prediction errors
4. **Statistical Significance**: Wilcoxon signed-rank tests validate model comparison methodology

### MADEX Advantages Demonstrated

- **Glucose-context-aware**: Different weights for hypo/eu/hyperglycemic ranges
- **Clinically relevant**: Penalizes dangerous errors more heavily
- **Statistically robust**: Bootstrap CIs provide uncertainty quantification

---

*Generated by Extended Bootstrap Analysis - 2025-12-02 10:11:42*