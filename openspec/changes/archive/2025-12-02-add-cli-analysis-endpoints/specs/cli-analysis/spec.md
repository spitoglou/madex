## ADDED Requirements

### Requirement: CLI Entry Point
The system SHALL provide a `madex` CLI command as the main entry point for all analysis operations.

#### Scenario: CLI help displayed
- **WHEN** user runs `madex --help`
- **THEN** available subcommands are listed (sandbox, narrative, bootstrap, sensitivity, analyze-all)

#### Scenario: Version displayed
- **WHEN** user runs `madex --version`
- **THEN** the package version is displayed

### Requirement: Sandbox Analysis Command
The system SHALL provide a `madex sandbox` command that runs extended scenario analysis comparing Model A and Model B predictions across clinical scenarios.

#### Scenario: Run sandbox analysis with defaults
- **WHEN** user runs `madex sandbox`
- **THEN** analysis runs with default parameters (a=125, b=55, c=100)
- **AND** results are saved to `extended_outcome/` directory
- **AND** Excel and PNG files are generated

#### Scenario: Run sandbox with custom output directory
- **WHEN** user runs `madex sandbox --output /custom/path`
- **THEN** results are saved to the specified directory

### Requirement: Narrative Analysis Command
The system SHALL provide a `madex narrative` command that runs clinical narrative analysis with bias pattern assessment.

#### Scenario: Run narrative analysis with defaults
- **WHEN** user runs `madex narrative`
- **THEN** analysis runs and generates clinical narrative report
- **AND** results are saved to `extended_narrative/narrative_results.md`

#### Scenario: Run narrative with custom output
- **WHEN** user runs `madex narrative --output /custom/path`
- **THEN** results are saved to the specified directory

### Requirement: Bootstrap Analysis Command
The system SHALL provide a `madex bootstrap` command that runs bootstrap statistical validation with configurable parameters.

#### Scenario: Run bootstrap analysis with defaults
- **WHEN** user runs `madex bootstrap`
- **THEN** analysis runs with 1000 bootstrap samples and 95% confidence interval
- **AND** results are saved to `extended_bootstrap/bootstrap_results.md`

#### Scenario: Run bootstrap with custom parameters
- **WHEN** user runs `madex bootstrap --samples 2000 --confidence 99`
- **THEN** analysis runs with 2000 bootstrap samples and 99% confidence interval

### Requirement: Sensitivity Analysis Command
The system SHALL provide a `madex sensitivity` command that runs parameter sensitivity testing across ranges.

#### Scenario: Run sensitivity analysis with defaults
- **WHEN** user runs `madex sensitivity`
- **THEN** analysis tests default parameter ranges (a: 110-140, b: 40-80, c: 20-80)
- **AND** results are saved to `extended_sensitivity/sensitivity_results.md`

#### Scenario: Run sensitivity with custom ranges
- **WHEN** user runs `madex sensitivity --a-range 100,150 --b-range 30,90`
- **THEN** analysis tests the specified parameter ranges

### Requirement: Combined Analysis Command
The system SHALL provide a `madex analyze-all` command that runs all four analyses sequentially.

#### Scenario: Run all analyses
- **WHEN** user runs `madex analyze-all`
- **THEN** sandbox, narrative, bootstrap, and sensitivity analyses run in sequence
- **AND** progress is displayed for each analysis
- **AND** summary of all results is displayed at completion

#### Scenario: Run all with shared output directory
- **WHEN** user runs `madex analyze-all --output /results`
- **THEN** all analysis outputs are saved under the specified directory
