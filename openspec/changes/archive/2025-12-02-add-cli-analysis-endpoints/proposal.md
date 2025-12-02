# Change: Add CLI Analysis Endpoints

## Why
The MADEX analysis scripts (sandbox, narrative, bootstrap, sensitivity) are currently standalone Python files that must be run directly. Adding CLI endpoints will provide a unified interface for running analyses, improve discoverability, and enable easier integration into workflows.

## What Changes
- Add CLI command `madex sandbox` to run extended scenario analysis
- Add CLI command `madex narrative` to run clinical narrative analysis
- Add CLI command `madex bootstrap` to run bootstrap statistical validation
- Add CLI command `madex sensitivity` to run parameter sensitivity analysis
- Add CLI command `madex analyze-all` to run all four analyses sequentially
- Rename/refactor script files to be importable modules
- Create project README.md with installation and usage instructions
- Move old standalone scripts to Archive folder for cleanup

## Impact
- Affected specs: cli-analysis (new capability)
- Affected code:
  - `sandbox_extended.py` → `src/new_metric/cli/sandbox.py` (then archive original)
  - `extended_narrative/narrative_extended.py` → `src/new_metric/cli/narrative.py` (then archive original)
  - `extended_bootstrap/bootstrap_extended.py` → `src/new_metric/cli/bootstrap.py` (then archive original)
  - `extended_sensitivity/sensitivity_extended.py` → `src/new_metric/cli/sensitivity.py` (then archive original)
  - `src/new_metric/cli/__init__.py` (new CLI entry point)
  - `pyproject.toml` (add CLI script entry points)
  - `README.md` (new project documentation)
