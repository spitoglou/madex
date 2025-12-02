# Change: Increase Model Error Magnitude for Clarke Zone Differentiation

## Why

The current error magnitudes in the extended clinical scenarios are too small, causing most model predictions to fall within Clarke Error Grid Zone A classification. This makes it difficult to demonstrate the clinical differentiation capabilities of MADEX vs traditional metrics when models have similar Zone A percentages.

## What Changes

- Increase error magnitude values in all 8 extended clinical scenarios (A-H) so model predictions fall outside Zone A more frequently
- Retain boundary controls ensuring predictions stay within physiologically valid ranges (minimum glucose > 0, maximum reasonable values)
- Maintain the controlled bias patterns (Model A vs Model B) with the increased error magnitudes

## Impact

- Affected specs: clinical-data (new capability spec)
- Affected code: `src/new_metric/common/clinical_data.py` - `ExtendedScenarioRepository._initialize_extended_scenarios()`
- CLI analysis outputs will show more differentiation between zones
- Bootstrap and sensitivity analyses will have more variance to analyze
