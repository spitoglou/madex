"""
new_metric: Mean Adjusted Exponent Error (MADE) library

A custom adaptive error function for machine learning evaluation that adjusts
the exponent in error calculation based on input and prediction characteristics.
"""

__version__ = "0.1.0"
__author__ = "new_metric team"
__description__ = "Mean Adjusted Exponent Error (MADE) - A custom adaptive error function for machine learning evaluation"

# Import main functions
from .cega import clarke_error_grid
from .madex import mean_adjusted_exponent_error, graph_vs_mse

__all__ = [
    "clarke_error_grid",
    "mean_adjusted_exponent_error", 
    "graph_vs_mse"
]