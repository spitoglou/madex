# Change: Add Timeseries Graphs for Reference and Model Predictions

## Why

The sandbox analysis currently generates Clarke Error Grid plots and metric comparison charts, but lacks visualization of the actual time-series data. Timeseries graphs showing reference glucose values alongside model predictions would help users visually understand the bias patterns and prediction errors over time.

## What Changes

- Add timeseries graph generation to the sandbox CLI module
- Each scenario will produce a graph showing:
  - Reference glucose values (y_true) as the baseline
  - Model A predictions overlaid
  - Model B predictions overlaid
- Graphs will be saved to the output folder alongside existing plots

## Impact

- Affected specs: cli-analysis
- Affected code: `src/new_metric/cli/sandbox.py`
- New output files: `{scenario_id}_timeseries.png` for each scenario
