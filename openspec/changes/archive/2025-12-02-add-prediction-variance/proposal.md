# Change: Add Random Variance to Model Predictions and Improve Timeseries Visualization

## Why

1. The current model predictions have deterministic error patterns which may not reflect real-world prediction variability. Adding small random variance (0-15 mg/dL) makes the predictions more realistic while maintaining equal traditional metric scores.

2. The timeseries graphs currently show all data points, making them cluttered. Showing prediction markers only when reference values are outside the euglycemic range (70-180 mg/dL) focuses attention on the clinically critical regions where bias patterns matter most.

## What Changes

- Add random variance (±0-15 mg/dL) to model predictions
- Ensure absolute differences from reference remain equal between models (equal RMSE/MAE)
- Modify timeseries plots to show prediction markers only outside euglycemic range
- Reference values continue to show all data points

## Impact

- Affected specs: clinical-data, cli-analysis
- Affected code: 
  - `src/new_metric/common/clinical_data.py` - `generate_model_predictions()`
  - `src/new_metric/cli/sandbox.py` - `generate_timeseries_plot()`
