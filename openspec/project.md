# Project Context

## Purpose
MADEX (Mean Adjusted Exponent Error) is a custom adaptive error metric library for machine learning evaluation, specifically designed for blood glucose prediction in clinical/diabetes management contexts. The library provides:

- **MADEX metric**: An adaptive error function that adjusts the exponent in error calculations based on clinical significance (hypoglycemic vs. hyperglycemic ranges)
- **Clarke Error Grid Analysis**: Implementation of the standard clinical tool for evaluating blood glucose prediction accuracy across five clinical significance zones (A-E)
- **Statistical analysis utilities**: Bootstrap analysis, sensitivity testing, and clinical scenario evaluation frameworks

## Tech Stack
- **Language**: Python 3.13+
- **Package Manager**: UV (with uv.lock for reproducible builds)
- **Build System**: Hatchling
- **Core Dependencies**:
  - NumPy, SciPy - numerical computing
  - Pandas - data manipulation
  - Matplotlib, Seaborn - visualization
  - scikit-learn - ML utilities and metrics
  - scikit-fuzzy - fuzzy logic support
  - NetworkX - graph analysis
  - OpenPyXL - Excel file handling
- **Dev Dependencies**: pytest, pytest-cov, coverage, flake8, autopep8
- **CI/CD**: GitLab CI with test and lint stages

## Project Conventions

### Code Style
- Line length: 88 characters (flake8 configured)
- Ignored rules: E203, W503
- Use autopep8 for formatting
- Functions use snake_case naming
- Classes use PascalCase naming

### Architecture Patterns
- **Module structure**: `src/new_metric/` contains main library code
  - `madex.py` - Core MADEX error metric implementation
  - `cega.py` - Clarke Error Grid Analysis
  - `common/` - Shared infrastructure (logging, data structures, statistical utilities, analysis frameworks)
- **Analysis framework**: Base classes in `common/analysis_framework.py` for building analysis workflows
- **Clinical data**: Scenario definitions and MADEX parameters in `common/clinical_data.py`

### Testing Strategy
- Test framework: pytest with coverage reporting
- Test location: `tests/` directory
- Test naming: `test_*.py` files with `test_*` functions
- Coverage targets source code in `src/`
- Coverage reports: terminal + XML (cov.xml)

### Git Workflow
- Main branch: `master`
- Feature branches for development (e.g., `new-extended`)
- Conventional commit messages based on recent history (feat, refactor, etc.)

## Domain Context
This library is designed for evaluating machine learning models that predict blood glucose levels:

- **Glucose ranges**: Normal physiological range is 0-400 mg/dl
- **Critical thresholds**: 70 mg/dl (hypoglycemia), 180 mg/dl (hyperglycemia)
- **MADEX parameters**:
  - `center` (default 125): Central glucose value for exponent adjustment
  - `critical_range` (default 55): Range width for clinical significance
  - `slope` (default 100): Steepness of exponent transition
- **Clarke Error Grid zones**: A (clinically accurate), B (acceptable), C (overcorrecting), D (failure to detect), E (erroneous treatment)

## Important Constraints
- Predictions and references should be in mg/dl units
- Input values should be within physiological range (0-400 mg/dl)
- MADEX is specifically tuned for blood glucose prediction - different clinical applications may require parameter adjustment

## External Dependencies
- No external APIs or services required
- Data inputs expected as NumPy arrays or Python lists
- Output visualizations use Matplotlib (can save to files or display interactively)

## Directory Structure Notes
- `archive/` - Old, inactive code (excluded from active development)
- `notebooks/` - Jupyter notebooks for exploration and demos
- `extended_*/` - Extended analysis output directories (bootstrap, narrative, outcome, sensitivity)
- `.venv/` - UV-managed virtual environment
