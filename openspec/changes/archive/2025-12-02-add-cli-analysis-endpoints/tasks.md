## 1. Setup CLI Infrastructure
- [x] 1.1 Create `src/new_metric/cli/` directory structure
- [x] 1.2 Add typer dependency to pyproject.toml
- [x] 1.3 Create `src/new_metric/cli/__init__.py` with main CLI group

## 2. Migrate Analysis Scripts
- [x] 2.1 Refactor `sandbox_extended.py` into `src/new_metric/cli/sandbox.py`
- [x] 2.2 Refactor `extended_narrative/narrative_extended.py` into `src/new_metric/cli/narrative.py`
- [x] 2.3 Refactor `extended_bootstrap/bootstrap_extended.py` into `src/new_metric/cli/bootstrap.py`
- [x] 2.4 Refactor `extended_sensitivity/sensitivity_extended.py` into `src/new_metric/cli/sensitivity.py`

## 3. Implement CLI Commands
- [x] 3.1 Add `madex sandbox` command with output directory option
- [x] 3.2 Add `madex narrative` command with output directory option
- [x] 3.3 Add `madex bootstrap` command with bootstrap count and confidence options
- [x] 3.4 Add `madex sensitivity` command with parameter range options
- [x] 3.5 Add `madex analyze-all` command to run all analyses

## 4. Configure Entry Points
- [x] 4.1 Add `[project.scripts]` entry point in pyproject.toml
- [x] 4.2 Verify `uv run madex --help` works

## 5. Testing and Documentation
- [x] 5.1 Verify all commands produce expected output

## 6. Documentation
- [x] 6.1 Create project README.md with installation and usage instructions

## 7. Cleanup
- [x] 7.1 Move `sandbox_extended.py` to Archive folder
- [x] 7.2 Move `extended_narrative/` folder to Archive
- [x] 7.3 Move `extended_bootstrap/` folder to Archive
- [x] 7.4 Move `extended_sensitivity/` folder to Archive
