## ADDED Requirements

### Requirement: Timeseries Visualization

The sandbox analysis SHALL generate timeseries graphs showing reference glucose values and model predictions over time for each clinical scenario.

#### Scenario: Generate timeseries graph for scenario
- **WHEN** sandbox analysis is run
- **THEN** system SHALL generate a timeseries plot for each scenario
- **AND** plot SHALL display reference glucose values as a line
- **AND** plot SHALL overlay Model A predictions
- **AND** plot SHALL overlay Model B predictions
- **AND** plot SHALL include a legend identifying each series
- **AND** plot SHALL be saved to output folder as `{scenario_id}_timeseries.png`

#### Scenario: Timeseries graph axes and labels
- **WHEN** timeseries graph is generated
- **THEN** x-axis SHALL represent time points (data point index)
- **AND** y-axis SHALL represent glucose concentration in mg/dL
- **AND** graph SHALL include a descriptive title with scenario name
