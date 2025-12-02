# madex-metric Specification

## Purpose
TBD - created by archiving change refactor-core-metric-api. Update Purpose after archive.
## Requirements
### Requirement: Unified MADEX Calculator
The system SHALL provide a `MADEXCalculator` class as the single canonical implementation of MADEX metric calculation.

#### Scenario: Calculate MADEX with default parameters
- **WHEN** user creates `MADEXCalculator()` without parameters
- **THEN** calculator uses clinically validated defaults (center=125, critical_range=55, slope=TBD)

#### Scenario: Calculate MADEX with custom parameters
- **WHEN** user creates `MADEXCalculator(params=MADEXParameters(a=110, b=40, c=30))`
- **THEN** calculator uses provided parameters for all calculations

#### Scenario: Calculate MADEX score
- **WHEN** user calls `calculator.calculate(y_true, y_pred)`
- **THEN** system returns single float MADEX score
- **AND** calculation matches the formula: mean(|y_pred - y_true|^exponent) where exponent = 2 - tanh((y_true - center) / critical_range) * ((y_pred - y_true) / slope)

### Requirement: Parameter Validation
The system SHALL validate MADEX parameters with clinically meaningful error messages.

#### Scenario: Invalid center parameter
- **WHEN** user provides center value outside 80-180 mg/dL
- **THEN** system raises ValueError with message explaining valid clinical range

#### Scenario: Invalid critical_range parameter  
- **WHEN** user provides critical_range outside 20-100
- **THEN** system raises ValueError with message explaining parameter meaning

### Requirement: Backward Compatible Convenience Function
The system SHALL maintain `mean_adjusted_exponent_error()` function for backward compatibility.

#### Scenario: Existing code compatibility
- **WHEN** user calls `mean_adjusted_exponent_error(y_true, y_pred, center=125, critical_range=55, slope=100)`
- **THEN** system returns identical result to `MADEXCalculator` with same parameters

#### Scenario: Default parameter alignment
- **WHEN** user calls `mean_adjusted_exponent_error(y_true, y_pred)` without parameters
- **THEN** system uses same defaults as `MADEXCalculator()`

### Requirement: Consistent MetricCalculator Integration
The `MetricCalculator.compute_madex()` method SHALL delegate to `MADEXCalculator` to ensure consistent results.

#### Scenario: MetricCalculator matches core implementation
- **WHEN** user calculates MADEX via `MetricCalculator().compute_madex(y_true, y_pred, params)`
- **THEN** result equals `MADEXCalculator(params).calculate(y_true, y_pred)`

### Requirement: Type-Safe Public API
The package SHALL export type-annotated public API from `__init__.py`.

#### Scenario: IDE autocompletion support
- **WHEN** user imports `from new_metric import MADEXCalculator, MADEXParameters`
- **THEN** IDE can provide type hints and autocompletion for all methods

#### Scenario: Package exports include new classes
- **WHEN** user imports `new_metric`
- **THEN** `MADEXCalculator`, `MADEXParameters`, and `mean_adjusted_exponent_error` are accessible

