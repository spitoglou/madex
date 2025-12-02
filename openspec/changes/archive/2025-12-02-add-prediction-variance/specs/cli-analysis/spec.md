## MODIFIED Requirements

### Requirement: Timeseries Visualization

The sandbox analysis SHALL generate timeseries graphs showing reference glucose values and model predictions over time for each clinical scenario.

#### Scenario: Generate timeseries graph for scenario
- **WHEN** sandbox analysis is run
- **THEN** system SHALL generate a timeseries plot for each scenario
- **AND** plot SHALL display reference glucose values as a continuous line with markers
- **AND** plot SHALL overlay Model A predictions as a line
- **AND** plot SHALL overlay Model B predictions as a line
- **AND** plot SHALL include a legend identifying each series
- **AND** plot SHALL be saved to output folder as `{scenario_id}_timeseries.png`

#### Scenario: Timeseries graph axes and labels
- **WHEN** timeseries graph is generated
- **THEN** x-axis SHALL represent time points (data point index)
- **AND** y-axis SHALL represent glucose concentration in mg/dL
- **AND** graph SHALL include a descriptive title with scenario name

#### Scenario: Prediction markers shown only outside euglycemia
- **WHEN** timeseries graph is generated
- **THEN** prediction data point markers SHALL only be visible when reference glucose is outside euglycemic range (below 70 or above 180 mg/dL)
- **AND** prediction lines SHALL remain continuous across all time points
- **AND** reference value markers SHALL be visible at all time points
