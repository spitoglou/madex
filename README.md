# MADEX - Mean Adjusted Exponent Error

A clinically-aware error metric for blood glucose prediction model evaluation.

## Overview

MADEX (Mean Adjusted Exponent Error) is an adaptive error function designed specifically for evaluating glucose prediction models in diabetes care. Unlike traditional metrics (RMSE, MAE), MADEX weights prediction errors based on their clinical significance - penalizing errors more heavily in dangerous glucose ranges (hypoglycemia and hyperglycemia) where prediction accuracy is critical for patient safety.

## Installation

### Prerequisites
- Python 3.7+
- [UV](https://github.com/astral-sh/uv) package manager (recommended)

### Install with UV

```bash
# Clone the repository
git clone <repository-url>
cd new_metric

# Install dependencies
uv sync

# Verify installation
uv run madex --help
```

### Install with pip

```bash
pip install -e .
```

## CLI Usage

MADEX provides a command-line interface for running various analyses:

### Available Commands

```bash
madex --help              # Show all available commands
madex --version           # Show version
madex sandbox             # Run extended scenario analysis
madex narrative           # Run clinical narrative analysis
madex bootstrap           # Run bootstrap statistical validation
madex sensitivity         # Run parameter sensitivity analysis
madex analyze-all         # Run all analyses sequentially
```

### Sandbox Analysis

Compares Model A and Model B predictions across 8 clinical scenarios with 50 data points each.

```bash
# Run with default output directory (extended_outcome/)
uv run madex sandbox

# Run with custom output directory
uv run madex sandbox --output /path/to/output
```

**Output:**
- Excel files with scenario data and results
- Clarke Error Grid plots for each model
- Comparison visualizations (MSE vs MADEX, model comparisons)

### Narrative Analysis

Generates detailed clinical narrative reports explaining model behavior patterns and their implications.

```bash
# Run with default output directory (extended_narrative/)
uv run madex narrative

# Run with custom output directory
uv run madex narrative --output /path/to/output
```

**Output:**
- `narrative_results.md` - Comprehensive clinical narrative report

### Bootstrap Analysis

Performs bootstrap statistical validation with confidence intervals.

```bash
# Run with defaults (1000 samples, 95% CI)
uv run madex bootstrap

# Custom bootstrap samples and confidence level
uv run madex bootstrap --samples 2000 --confidence 99

# Custom output directory
uv run madex bootstrap --output /path/to/output
```

**Output:**
- `bootstrap_results.md` - Statistical validation report with confidence intervals

### Sensitivity Analysis

Tests MADEX parameter robustness across clinical parameter ranges.

```bash
# Run with default parameter ranges
uv run madex sensitivity

# Custom parameter ranges (a: euglycemic center, b: critical range)
uv run madex sensitivity --a-range 100,150 --b-range 30,90

# Custom output directory
uv run madex sensitivity --output /path/to/output
```

**Output:**
- `sensitivity_results.md` - Parameter sensitivity analysis report

### Run All Analyses

Execute all four analyses sequentially.

```bash
# Run all with default directories
uv run madex analyze-all

# Run all with shared output directory
uv run madex analyze-all --output /path/to/results
```

## Python API Usage

### Basic MADEX Calculation

```python
from new_metric import mean_adjusted_exponent_error

y_true = [100, 80, 150, 200]  # Reference glucose values (mg/dL)
y_pred = [105, 75, 160, 190]  # Predicted glucose values (mg/dL)

# Calculate MADEX with default parameters (a=125, b=55, c=100)
madex_score = mean_adjusted_exponent_error(y_true, y_pred)
print(f"MADEX Score: {madex_score:.2f}")

# Calculate with custom parameters
madex_score = mean_adjusted_exponent_error(
    y_true, y_pred,
    center=125,        # Euglycemic center (mg/dL)
    critical_range=55, # Critical range parameter
    slope=100          # Slope modifier
)
```

### Using MADEXCalculator Class

```python
from new_metric.madex import MADEXCalculator, MADEXParameters

# Create calculator with custom parameters
params = MADEXParameters(center=125, critical_range=55, slope=100)
calculator = MADEXCalculator(params)

# Calculate MADEX
score = calculator.calculate(y_true, y_pred)
```

### Clarke Error Grid Analysis

```python
from new_metric.cega import clarke_error_grid

y_true = [100, 80, 150, 200]
y_pred = [105, 75, 160, 190]

# Generate Clarke Error Grid plot
plot, zones = clarke_error_grid(y_true, y_pred, "Model A")
plot.savefig("clarke_grid.png")

# zones contains percentages: [Zone_A, Zone_B, Zone_C, Zone_D, Zone_E]
print(f"Zone A (Accurate): {zones[0]}%")
```

## MADEX Formula

MADEX weights prediction errors based on glucose context:

```
MADEX = mean(exp(|glucose - a| / b) * |error| / c)

Where:
  a = 125 mg/dL (euglycemic center)
  b = 55 (critical range parameter)
  c = 100 (slope modifier)
```

This means:
- Errors near euglycemic range (around 125 mg/dL) receive baseline weighting
- Errors in hypoglycemic range (<70 mg/dL) receive exponentially higher weighting
- Errors in hyperglycemic range (>180 mg/dL) receive exponentially higher weighting

## Clinical Scenarios

The analysis uses 8 clinical scenarios with 50 data points each:

| Scenario | Name | Glucose Range | Risk Level |
|----------|------|---------------|------------|
| A | Hypoglycemia Detection | 30-85 mg/dL | CRITICAL |
| B | Hyperglycemia Management | 150-325 mg/dL | HIGH |
| C | Postprandial Response | 95-220 mg/dL | MODERATE |
| D | Dawn Phenomenon | 90-160 mg/dL | LOW |
| E | Exercise Response | 60-140 mg/dL | MODERATE |
| F | Measurement Noise | 90-131 mg/dL | LOW |
| G | Mixed Clinical | 35-300 mg/dL | CRITICAL |
| H | Extreme Cases | 20-450 mg/dL | CRITICAL |

## Development

### Running Tests

```bash
uv run pytest
```

### Code Formatting

```bash
uv run autopep8 --in-place --recursive src/
uv run flake8 src/
```

## License

See LICENSE file for details.
