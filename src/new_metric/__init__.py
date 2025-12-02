"""
new_metric: MADEX (Mean Adjusted Exponent Error) library

A custom adaptive error metric for machine learning evaluation that adjusts
the exponent in error calculation based on clinical significance of glucose values.

This library is designed for blood glucose prediction evaluation where errors
in different glucose ranges have different clinical implications.
"""

__version__ = "0.1.0"
__author__ = "new_metric team"
__description__ = (
    "MADEX - A custom adaptive error metric for blood glucose prediction evaluation"
)

# Import core MADEX components
# Import Clarke Error Grid
from .cega import clarke_error_grid
from .madex import (
    MADEXCalculator,
    MADEXParameters,
    graph_vs_mse,
    mean_adjusted_exponent_error,
)

__all__ = [
    # Core MADEX API
    "MADEXCalculator",
    "MADEXParameters",
    "mean_adjusted_exponent_error",
    "graph_vs_mse",
    # Clarke Error Grid
    "clarke_error_grid",
]
