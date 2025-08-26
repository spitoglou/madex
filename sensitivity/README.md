# MADEX Sensitivity Analysis

This directory contains the comprehensive sensitivity analysis for the MADEX (Mean Adjusted Exponent Error) metric.

## Files

### Core Analysis

- **`sensitivity_analysis.py`** - Main sensitivity analysis script (formerly `colab1.py`)
  - Comprehensive parameter sensitivity testing for MADEX metric
  - Clinical scenario evaluation across 8 realistic glucose prediction scenarios
  - Comparison with traditional metrics (RMSE, MAE, MAPE)
  - LLM-optimized narrative logging system

### Documentation & Demo

- **`narrative_demo.py`** - Demonstration script showing LLM-friendly logging features
- **`README.md`** - This documentation file

### Generated Outputs

- **`madex_llm_narrative.log`** - Clean narrative log optimized for LLM consumption
- **`madex_analysis_narrative.log`** - Detailed technical log with timestamps

**Note**: Log files are always created in the sensitivity directory, regardless of where the script is executed from.

## Usage

### Run Sensitivity Analysis

```bash
cd sensitivity
python sensitivity_analysis.py
```

### View Logging Demo

```bash
cd sensitivity
python narrative_demo.py
```

## Analysis Features

### Clinical Scenarios

The analysis evaluates 8 clinically realistic scenarios:

- **A_Hypoglycemia_Detection** - Critical low glucose events (CRITICAL)
- **B_Hyperglycemia_Management** - Sustained high glucose (HIGH)
- **C_Postprandial_Response** - Rapid meal-induced changes (MODERATE)
- **D_Dawn_Phenomenon** - Early morning glucose rise (LOW)
- **E_Exercise_Response** - Activity-induced glucose drop (MODERATE)
- **F_Measurement_Noise** - Sensor accuracy challenges (LOW)
- **G_Mixed_Clinical** - Combined challenging scenarios (CRITICAL)
- **H_Extreme_Cases** - Life-threatening glucose extremes (CRITICAL)

### Parameter Ranges

- **Parameter 'a'** (euglycemic center): [110, 125, 140] mg/dL
- **Parameter 'b'** (critical range): [40, 55, 70, 80]
- **Parameter 'c'** (slope modifier): [20, 30, 40, 50, 60, 80]

### Key Outputs

- Parameter stability assessment across 72 combinations
- Clinical significance annotations for each scenario
- Robustness recommendations for different clinical contexts
- Comparative analysis with traditional error metrics

## LLM-Friendly Logging

The analysis generates structured narrative logs optimized for Large Language Model consumption:

- **Comprehensive context** for each analysis step
- **Clinical interpretations** of parameter effects
- **Progress narratives** with explanations
- **Structured sections** with clear headers
- **ASCII-safe formatting** for cross-platform compatibility

## Dependencies

Requires the parent `new_metric` package to be available. The script automatically adds the parent directory to the Python path for module imports.
