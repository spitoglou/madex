# Bootstrap Analysis Module

This directory contains the bootstrap statistical validation analysis for the MADEX metric evaluation.

## Files

### Core Analysis

- **`bootstrap.py`** - Main bootstrap validation script that performs comprehensive statistical analysis of MADEX metric across clinical scenarios

### Output Logs

- **`bootstrap_analysis_narrative.log`** - Detailed technical log with timestamps and structured analysis output
- **`bootstrap_llm_narrative.log`** - Clean narrative log optimized for LLM consumption without timestamps

**Note**: Log files are always created in the bootstrap directory, regardless of where the script is executed from.

## Overview

The bootstrap analysis provides statistical confidence intervals for MADEX (Mean Adjusted Exponent Error) and compares its performance against traditional metrics (RMSE, MAE) across clinically relevant glucose prediction scenarios.

## Key Features

- **Bootstrap Confidence Intervals**: 1000 bootstrap samples with 95% confidence intervals
- **Clinical Scenario Analysis**: 8 different glucose prediction scenarios (A-H)
- **Statistical Testing**: Wilcoxon signed-rank tests for model comparison
- **Clinical Risk Assessment**: Clarke Error Grid Analysis with clinical risk scoring
- **Detection Sensitivity**: Hypoglycemia and hyperglycemia detection analysis

## Usage

Run the bootstrap analysis from the project root:

```bash
python bootstrap/bootstrap.py
```

Or from within the bootstrap directory:

```bash
cd bootstrap
python bootstrap.py
```

## Requirements

- Python 3.9+
- Dependencies from project's `pyproject.toml`
- Access to `new_metric.cega` module for Clarke Error Grid Analysis

## Output

The script generates:

1. Console output with summary tables and statistical results
2. Detailed narrative logs for comprehensive analysis review
3. Clinical interpretations and recommendations based on statistical findings

## Clinical Scenarios

- **Scenario A**: Hypoglycemia Detection (45-61 mg/dL) - CRITICAL
- **Scenario B**: Hyperglycemia Management (275-310 mg/dL) - HIGH  
- **Scenario C**: Postprandial Response (115-220 mg/dL) - MODERATE
- **Scenario D**: Dawn Phenomenon (85-155 mg/dL) - LOW
- **Scenario E**: Exercise Response (65-140 mg/dL) - MODERATE
- **Scenario F**: Measurement Noise (97-105 mg/dL) - LOW
- **Scenario G**: Mixed Clinical (45-250 mg/dL) - CRITICAL
- **Scenario H**: Extreme Cases (25-450 mg/dL) - CRITICAL
