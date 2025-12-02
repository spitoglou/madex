## ADDED Requirements

### Requirement: Extended Clinical Scenario Error Magnitudes

The system SHALL generate model predictions with error magnitudes sufficient to produce Clarke Error Grid zone differentiation beyond Zone A.

#### Scenario: Error magnitudes produce Zone B/C/D/E classifications
- **WHEN** extended clinical scenarios are generated
- **THEN** model predictions SHALL have error magnitudes large enough that a significant portion of predictions fall outside Clarke Zone A
- **AND** the Zone A percentage SHALL be less than 100% for scenarios with critical glucose ranges

#### Scenario: Boundary controls prevent invalid predictions
- **WHEN** error magnitudes are applied to reference glucose values
- **THEN** model predictions SHALL remain within physiologically valid ranges
- **AND** predictions SHALL NOT go below 1 mg/dL
- **AND** predictions SHALL NOT exceed reasonable maximum values for the scenario context

### Requirement: Controlled Model Bias Patterns

The system SHALL maintain opposite bias patterns between Model A and Model B regardless of error magnitude.

#### Scenario: Hypoglycemic bias pattern preserved
- **WHEN** reference glucose is below euglycemic range (<70 mg/dL)
- **THEN** Model A SHALL underestimate (predict lower)
- **AND** Model B SHALL overestimate (predict higher)

#### Scenario: Hyperglycemic bias pattern preserved
- **WHEN** reference glucose is above euglycemic range (>180 mg/dL)
- **THEN** Model A SHALL overestimate (predict higher)
- **AND** Model B SHALL underestimate (predict lower)

#### Scenario: Euglycemic random direction preserved
- **WHEN** reference glucose is within euglycemic range (70-180 mg/dL)
- **THEN** Model A and Model B SHALL have opposite error directions
- **AND** the direction SHALL be randomly determined per data point
