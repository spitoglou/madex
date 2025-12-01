# Extended Bootstrap Statistical Validation Analysis

**Analysis Date:** 2025-12-01 22:14:04

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
| A | 62.22 (56.65-68.10) | 282.66 (195.61-383.18) | Model A |
| B | 84.96 (80.17-90.59) | 1905.09 (1101.71-2873.64) | Model A |
| C | 72.24 (61.72-83.52) | 156.24 (111.03-202.15) | Model A |
| D | 59.42 (51.88-67.62) | 54.99 (49.07-61.46) | Model B |
| E | 128.98 (99.50-160.57) | 198.62 (137.83-268.71) | Model A |
| F | 48.12 (44.26-52.18) | 48.59 (44.65-52.35) | Model A |
| G | 118.28 (103.07-136.75) | 1058.88 (775.66-1376.04) | Model A |
| H | 99.71 (75.39-128.18) | 13775.61 (6201.74-23360.36) | Model A |


### Traditional Metrics Comparison

| Scenario | RMSE_A (95% CI) | RMSE_B (95% CI) | MAE_A (95% CI) | MAE_B (95% CI) |
|----------|-----------------|-----------------|----------------|----------------|
| A | 11.15 (10.11-12.24) | 11.14 (10.03-12.20) | 10.50 (9.48-11.63) | 10.51 (9.51-11.52) |
| B | 17.45 (15.73-19.33) | 17.41 (15.55-19.23) | 16.49 (14.81-18.18) | 16.43 (14.81-18.04) |
| C | 9.77 (8.86-10.62) | 9.75 (8.92-10.59) | 9.26 (8.45-10.15) | 9.28 (8.44-10.10) |
| D | 7.49 (7.10-7.86) | 7.51 (7.11-7.88) | 7.36 (6.96-7.78) | 7.37 (6.96-7.77) |
| E | 11.47 (10.48-12.47) | 11.46 (10.45-12.43) | 10.89 (9.99-11.87) | 10.91 (9.98-11.90) |
| F | 6.94 (6.69-7.20) | 6.94 (6.69-7.18) | 6.87 (6.61-7.14) | 6.88 (6.61-7.14) |
| G | 16.56 (15.33-17.91) | 16.52 (15.26-17.84) | 15.90 (14.57-17.28) | 15.87 (14.51-17.20) |
| H | 22.93 (19.59-26.19) | 22.98 (19.53-26.36) | 20.19 (17.17-23.23) | 20.11 (17.21-23.18) |


### Clarke Error Grid Risk Scores

| Scenario | Model A Risk | Model B Risk | Lower Risk Model |
|----------|--------------|--------------|------------------|
| A | 0 | 0 | Tie |
| B | 0 | 0 | Tie |
| C | 0 | 0 | Tie |
| D | 0 | 0 | Tie |
| E | 6 | 36 | Model A |
| F | 0 | 0 | Tie |
| G | 0 | 20 | Model A |
| H | 1 | 16 | Model A |

---

## Statistical Significance Testing

### Wilcoxon Signed-Rank Test Results

| Metric | Statistic | p-value | Cohen's d | Interpretation |
|--------|-----------|---------|-----------|----------------|
| MADEX | 2.00 | 0.0234 | -0.45 | Significant difference |
| RMSE | 13.00 | 0.5469 | 0.15 | No significant difference |
| MAE | 16.00 | 0.8438 | 0.34 | No significant difference |


### Model Comparison Summary

| Scenario | MADEX Winner | RMSE Winner | MAE Winner | CI Overlap |
|----------|--------------|-------------|------------|------------|
| A | Model A | Model B | Model A | No |
| B | Model A | Model B | Model B | No |
| C | Model A | Model B | Model A | No |
| D | Model B | Model A | Model A | Yes |
| E | Model A | Model B | Model A | Yes |
| F | Model A | Model A | Model A | Yes |
| G | Model A | Model B | Model B | No |
| H | Model A | Model A | Model B | No |

---

## Clinical Detection Sensitivity Analysis

### Hypoglycemia Detection (<70 mg/dL)

| Scenario | Model A Sensitivity | Model B Sensitivity | Better Model |
|----------|---------------------|---------------------|--------------|
| A | 1.00 | 0.83 | Model A |
| B | N/A | N/A | No hypo points |
| C | N/A | N/A | No hypo points |
| D | N/A | N/A | No hypo points |
| E | 1.00 | 0.00 | Model A |
| F | N/A | N/A | No hypo points |
| G | 1.00 | 0.69 | Model A |
| H | 1.00 | 0.67 | Model A |

### Hyperglycemia Detection (>180 mg/dL)

| Scenario | Model A Sensitivity | Model B Sensitivity | Better Model |
|----------|---------------------|---------------------|--------------|
| A | N/A | N/A | No hyper points |
| B | 1.00 | 0.88 | Model A |
| C | 1.00 | 0.67 | Model A |
| D | N/A | N/A | No hyper points |
| E | N/A | N/A | No hyper points |
| F | N/A | N/A | No hyper points |
| G | 1.00 | 0.94 | Model A |
| H | 1.00 | 1.00 | Tie |

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

*Generated by Extended Bootstrap Analysis - 2025-12-01 22:14:08*