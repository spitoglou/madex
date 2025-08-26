# new_metric

[![pipeline status](https://gitlab.csl.gr/spitoglou/new_metric/badges/master/pipeline.svg)](https://gitlab.csl.gr/spitoglou/new_metric/-/commits/master)
[![coverage report](https://gitlab.csl.gr/spitoglou/new_metric/badges/master/coverage.svg)](https://gitlab.csl.gr/spitoglou/new_metric/-/commits/master)

A custom adaptive error function for machine learning evaluation that adjusts
the exponent in error calculation based on input and prediction characteristics.

## Features

- **Mean Adjusted Exponent Error (MADE)**: Adaptive error computation using
  hyperbolic tangent modulation
- **Clarke Error Grid Analysis**: Clinical accuracy evaluation for glucose
  prediction
- **Modular Design**: Reusable error functions for ML pipelines
- **Comprehensive Testing**: Unit tests with 92% coverage

## Setting up environment

### Prerequisites

- Python 3.7 or higher
- [uv](https://docs.astral.sh/uv/) - Fast Python package installer and resolver

### Installation

1. **Install uv** (if not already installed):

   ```bash
   pip install uv
   ```

2. **Create virtual environment**:

   ```bash
   uv venv
   ```

3. **Install the package**:

   ```bash
   # Install main dependencies
   uv pip install -e .
   
   # Install with development dependencies
   uv pip install -e ".[dev]"
   ```

## Usage

```python
import new_metric

# Mean Adjusted Exponent Error
y_true = [100, 120, 150]
y_pred = [95, 125, 140]
error = new_metric.mean_adjusted_exponent_error(y_true, y_pred)

# Clarke Error Grid Analysis
plot, zones = new_metric.clarke_error_grid(y_true, y_pred, "My Analysis")
plot.show()
```

## The main error function (Mean Adjusted Exponent Error)

$Mean AdjustedExponentr Error = \frac{1}{N}\sum_{i=1}^n|\hat{y}_i-y_i|^{exp}$

$exp = 2 - tanh(\frac{y_i-a}{b})\times(\frac{\hat{y}_i-y_i}{c})$

$a: center, b:critical range, c:slope$

## Development

### Run Tests

Run tests using uv:

```bash
uv run pytest
```

### Coverage

To see coverage run:

```bash
uv run pytest --cov
```

To create `cov.xml` file (for IDE coverage reporting):

```bash
uv run pytest --cov-report xml:cov.xml --cov
```

### Code Quality

Run linting:

```bash
uv run flake8
```

Format code:

```bash
uv run autopep8 --in-place --recursive src tests
```

### Jupyter Notebooks

Explore experiments:

```bash
uv run jupyter notebook notebooks/experiments.ipynb
```
