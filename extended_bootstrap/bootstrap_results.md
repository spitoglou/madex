# Extended Bootstrap Statistical Validation Analysis

**Analysis Date:** 2025-12-02 10:33:53

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
| A | 288.38 (247.78-327.34) | 1739.54 (1272.90-2211.28) | Model A |
| B | 698.97 (421.99-1036.36) | 62098.14 (39041.34-90762.83) | Model A |
| C | 1046.67 (766.54-1422.17) | 6577.19 (3474.62-9832.91) | Model A |
| D | 1098.39 (864.90-1345.08) | 1138.58 (876.70-1447.41) | Model A |
| E | 1402.68 (1043.13-1856.51) | 2981.56 (1907.22-4366.36) | Model A |
| F | 1327.84 (1064.10-1652.41) | 1464.22 (1194.96-1762.76) | Model A |
| G | 1418.07 (678.14-2422.20) | 25245.40 (15021.65-36716.17) | Model A |
| H | 764.78 (339.83-1466.70) | 354954.69 (157634.14-589349.93) | Model A |


### Traditional Metrics Comparison

| Scenario | RMSE_A (95% CI) | RMSE_B (95% CI) | MAE_A (95% CI) | MAE_B (95% CI) |
|----------|-----------------|-----------------|----------------|----------------|
| A | 25.84 (23.15-28.34) | 25.86 (23.40-28.16) | 23.97 (21.20-26.77) | 23.88 (21.28-26.84) |
| B | 58.66 (54.68-62.88) | 58.59 (54.26-62.67) | 57.07 (53.01-61.24) | 56.86 (52.80-60.87) |
| C | 38.33 (34.54-42.25) | 38.23 (34.44-41.85) | 35.90 (32.48-39.66) | 35.90 (32.37-39.68) |
| D | 31.23 (28.88-33.47) | 31.26 (28.95-33.33) | 29.90 (27.17-32.41) | 29.82 (27.40-32.50) |
| E | 37.83 (35.48-40.39) | 37.72 (35.23-40.12) | 36.68 (34.11-39.29) | 36.71 (34.28-39.16) |
| F | 35.48 (33.23-37.73) | 35.43 (33.11-37.43) | 34.48 (32.25-36.70) | 34.50 (32.29-36.79) |
| G | 49.25 (45.12-53.46) | 49.14 (44.87-53.19) | 46.79 (42.44-50.89) | 46.85 (42.37-51.01) |
| H | 62.75 (54.42-70.83) | 62.78 (54.38-71.76) | 55.71 (48.16-63.98) | 55.51 (48.23-63.55) |


### Clarke Error Grid Risk Scores

| Scenario | Model A Risk | Model B Risk | Lower Risk Model |
|----------|--------------|--------------|------------------|
| A | 1 | 131 | Model A |
| B | 47 | 55 | Model A |
| C | 34 | 34 | Tie |
| D | 37 | 37 | Tie |
| E | 38 | 78 | Model A |
| F | 46 | 46 | Tie |
| G | 36 | 99 | Model A |
| H | 31 | 81 | Model A |

---

## Statistical Significance Testing

### Wilcoxon Signed-Rank Test Results

| Metric | Statistic | p-value | Cohen's d | Interpretation |
|--------|-----------|---------|-----------|----------------|
| MADEX | 0.00 | 0.0078 | -0.46 | Significant difference |
| RMSE | 6.00 | 0.1094 | 0.68 | No significant difference |
| MAE | 10.00 | 0.3125 | 0.55 | No significant difference |


### Model Comparison Summary

| Scenario | MADEX Winner | RMSE Winner | MAE Winner | CI Overlap |
|----------|--------------|-------------|------------|------------|
| A | Model A | Model A | Model B | No |
| B | Model A | Model B | Model B | No |
| C | Model A | Model B | Model A | No |
| D | Model A | Model A | Model B | Yes |
| E | Model A | Model B | Model A | No |
| F | Model A | Model B | Model A | Yes |
| G | Model A | Model B | Model A | No |
| H | Model A | Model A | Model B | No |

---

## Clinical Detection Sensitivity Analysis

### Hypoglycemia Detection (<70 mg/dL)

| Scenario | Model A Sensitivity | Model B Sensitivity | Better Model |
|----------|---------------------|---------------------|--------------|
| A | 1.00 | 0.32 | Model A |
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
| B | 1.00 | 0.48 | Model A |
| C | 1.00 | 0.00 | Model A |
| D | N/A | N/A | No hyper points |
| E | N/A | N/A | No hyper points |
| F | N/A | N/A | No hyper points |
| G | 1.00 | 0.56 | Model A |
| H | 1.00 | 0.95 | Model A |

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

*Generated by Extended Bootstrap Analysis - 2025-12-02 10:33:57*