"""
MADEX (Mean Adjusted Exponent Error) - Core Implementation

A custom adaptive error metric for machine learning evaluation that adjusts
the exponent in error calculation based on clinical significance of glucose values.

The metric is particularly designed for blood glucose prediction evaluation where
errors in different glucose ranges have different clinical implications.
"""

from dataclasses import dataclass
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error


@dataclass
class MADEXParameters:
    """
    MADEX parameter configuration with clinical interpretation.

    Parameters control how the adaptive exponent is calculated:
    - center (a): Target glucose value in mg/dL. Errors near this value use exponent ~2.
    - critical_range (b): Controls sensitivity to deviations from center.
      Smaller values = sharper transition between hypo/hyperglycemic penalties.
    - slope (c): Error scaling factor. Controls how much prediction error affects
      the exponent adjustment. Larger values = more gradual exponent changes.

    Clinical Context:
    - For hypoglycemia (glucose < center): underestimation is penalized more heavily
    - For hyperglycemia (glucose > center): overestimation is penalized more heavily
    - The adaptive exponent ensures clinically dangerous errors receive higher penalties

    Attributes:
        center: Euglycemic center point in mg/dL (default: 125, typical target range)
        critical_range: Sensitivity parameter (default: 55)
        slope: Error scaling factor (default: 100)
    """

    center: float = 125.0
    critical_range: float = 55.0
    slope: float = 100.0

    # Aliases for mathematical notation (a, b, c)
    @property
    def a(self) -> float:
        """Alias for center parameter (mathematical notation)."""
        return self.center

    @property
    def b(self) -> float:
        """Alias for critical_range parameter (mathematical notation)."""
        return self.critical_range

    @property
    def c(self) -> float:
        """Alias for slope parameter (mathematical notation)."""
        return self.slope

    def __post_init__(self):
        """Validate parameters after initialization."""
        self.validate()

    def validate(self):
        """
        Validate MADEX parameters are within clinically reasonable ranges.

        Raises:
            ValueError: If parameters are outside valid ranges
        """
        if not (50 <= self.center <= 250):
            raise ValueError(
                f"Parameter 'center' ({self.center}) must be between 50-250 mg/dL. "
                f"This represents the target glucose level for exponent adjustment."
            )

        if not (10 <= self.critical_range <= 150):
            raise ValueError(
                f"Parameter 'critical_range' ({self.critical_range}) must be between 10-150. "
                f"This controls sensitivity to glucose deviations from the center."
            )

        if not (10 <= self.slope <= 200):
            raise ValueError(
                f"Parameter 'slope' ({self.slope}) must be between 10-200. "
                f"This controls how prediction errors affect the adaptive exponent."
            )

    def get_clinical_interpretation(self) -> dict:
        """
        Get clinical interpretation of current parameter values.

        Returns:
            Dictionary mapping parameter names to clinical interpretations
        """
        interpretations = {}

        if self.center < 100:
            interpretations["center"] = (
                f"Tight glycemic control target ({self.center} mg/dL)"
            )
        elif self.center > 140:
            interpretations["center"] = (
                f"Relaxed glycemic control target ({self.center} mg/dL)"
            )
        else:
            interpretations["center"] = (
                f"Standard glycemic control target ({self.center} mg/dL)"
            )

        if self.critical_range < 40:
            interpretations["critical_range"] = (
                f"High sensitivity (b={self.critical_range})"
            )
        elif self.critical_range > 70:
            interpretations["critical_range"] = (
                f"Low sensitivity (b={self.critical_range})"
            )
        else:
            interpretations["critical_range"] = (
                f"Moderate sensitivity (b={self.critical_range})"
            )

        if self.slope < 50:
            interpretations["slope"] = f"Steep error scaling (c={self.slope})"
        elif self.slope > 120:
            interpretations["slope"] = f"Gentle error scaling (c={self.slope})"
        else:
            interpretations["slope"] = f"Standard error scaling (c={self.slope})"

        return interpretations


# Default parameters - clinically validated defaults
DEFAULT_PARAMS = MADEXParameters(center=125.0, critical_range=55.0, slope=100.0)


class MADEXCalculator:
    """
    Calculator for MADEX (Mean Adjusted Exponent Error) metric.

    MADEX adapts the error exponent based on glucose value and prediction direction,
    providing clinically meaningful error penalties for blood glucose prediction.

    Formula:
        MADEX = mean(|y_pred - y_true|^exponent)
        where exponent = 2 - tanh((y_true - center) / critical_range) * ((y_pred - y_true) / slope)

    Example:
        >>> calculator = MADEXCalculator()
        >>> score = calculator.calculate([100, 150, 200], [105, 145, 210])
        >>> print(f"MADEX score: {score:.2f}")

        >>> # With custom parameters
        >>> params = MADEXParameters(center=110, critical_range=40, slope=80)
        >>> calculator = MADEXCalculator(params)
        >>> score = calculator.calculate(y_true, y_pred)
    """

    def __init__(self, params: Optional[MADEXParameters] = None):
        """
        Initialize MADEX calculator with parameters.

        Args:
            params: MADEX parameters. If None, uses clinically validated defaults
                   (center=125, critical_range=55, slope=100)
        """
        self.params = params if params is not None else DEFAULT_PARAMS

    def _compute_exponent(self, y_true: float, y_pred: float) -> float:
        """
        Compute adaptive exponent for a single prediction.

        The exponent adjusts based on:
        - Whether the true value is below/above the euglycemic center
        - Whether the prediction over/underestimates the true value

        Args:
            y_true: True glucose value
            y_pred: Predicted glucose value

        Returns:
            Adaptive exponent value (typically between 1.5 and 2.5)
        """
        tanh_component = np.tanh(
            (y_true - self.params.center) / self.params.critical_range
        )
        error_component = (y_pred - y_true) / self.params.slope
        return 2 - tanh_component * error_component

    def calculate(
        self, y_true: Union[List, np.ndarray], y_pred: Union[List, np.ndarray]
    ) -> float:
        """
        Calculate MADEX score for predictions.

        Args:
            y_true: Array of true glucose values (mg/dL)
            y_pred: Array of predicted glucose values (mg/dL)

        Returns:
            MADEX score (lower is better)

        Raises:
            ValueError: If arrays have different lengths or are empty
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        if len(y_true) != len(y_pred):
            raise ValueError(
                f"Array length mismatch: y_true has {len(y_true)} elements, "
                f"y_pred has {len(y_pred)} elements"
            )

        if len(y_true) == 0:
            raise ValueError("Cannot calculate MADEX on empty arrays")

        # Calculate exponents and powered errors
        total = 0.0
        for i in range(len(y_true)):
            exp = self._compute_exponent(y_true[i], y_pred[i])
            total += abs(y_pred[i] - y_true[i]) ** exp

        return total / len(y_true)

    def calculate_detailed(
        self, y_true: Union[List, np.ndarray], y_pred: Union[List, np.ndarray]
    ) -> dict:
        """
        Calculate MADEX with detailed breakdown.

        Args:
            y_true: Array of true glucose values
            y_pred: Array of predicted glucose values

        Returns:
            Dictionary with:
            - score: Overall MADEX score
            - exponents: Array of adaptive exponents per point
            - errors: Array of absolute errors per point
            - powered_errors: Array of errors raised to adaptive exponent
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        if len(y_true) != len(y_pred):
            raise ValueError(
                f"Array length mismatch: y_true has {len(y_true)} elements, "
                f"y_pred has {len(y_pred)} elements"
            )

        exponents = np.array(
            [self._compute_exponent(yt, yp) for yt, yp in zip(y_true, y_pred)]
        )
        errors = np.abs(y_pred - y_true)
        powered_errors = errors**exponents

        return {
            "score": np.mean(powered_errors),
            "exponents": exponents,
            "errors": errors,
            "powered_errors": powered_errors,
            "params": self.params,
        }


def mean_adjusted_exponent_error(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    center: float = 125,
    critical_range: float = 55,
    slope: float = 100,
    verbose: bool = False,
) -> float:
    """
    Calculate Mean Adjusted Exponent Error (MADEX).

    MADEX is an adaptive error metric designed for blood glucose prediction
    that adjusts penalties based on clinical significance of errors.

    Args:
        y_true: Array of true glucose values (mg/dL)
        y_pred: Array of predicted glucose values (mg/dL)
        center: Euglycemic center point (default: 125 mg/dL)
        critical_range: Sensitivity parameter (default: 55)
        slope: Error scaling factor (default: 100)
        verbose: If True, print exponent for each data point

    Returns:
        MADEX score (lower is better)

    Example:
        >>> y_true = [100, 150, 200]
        >>> y_pred = [105, 145, 210]
        >>> score = mean_adjusted_exponent_error(y_true, y_pred)
        >>> print(f"MADEX: {score:.2f}")
    """
    params = MADEXParameters(center=center, critical_range=critical_range, slope=slope)
    calculator = MADEXCalculator(params)

    if verbose:
        result = calculator.calculate_detailed(y_true, y_pred)
        for i, exp in enumerate(result["exponents"]):
            print(f"Point {i}: exponent = {exp:.4f}")
        return result["score"]

    return calculator.calculate(y_true, y_pred)


def graph_vs_mse(
    value: float,
    value_range: float,
    action: Optional[str] = None,
    save_folder: str = ".",
):
    """
    Plot MADEX vs MSE comparison for a range of predictions.

    Creates a visualization comparing MADEX and MSE error curves for
    predictions around a reference glucose value.

    Args:
        value: Reference glucose value (mg/dL)
        value_range: Range of prediction values to plot (+/- from reference)
        action: If 'save', saves plot to file instead of returning
        save_folder: Directory for saved plots (default: current directory)

    Returns:
        Matplotlib pyplot object (unless action='save')

    Example:
        >>> plot = graph_vs_mse(100, 50)
        >>> plot.show()
    """
    prediction = np.arange(value - value_range, value + value_range)
    errors = []
    mse = []
    for pred in prediction:
        errors.append(mean_adjusted_exponent_error([value], [pred]))
        mse.append(mean_squared_error([value], [pred]))
    plt.plot(prediction, errors, label="madex")
    plt.plot(prediction, mse, label="mse", ls="dotted")
    plt.axvline(value, label="Reference Value", color="k", ls="--")
    plt.xlabel("Predicted Value")
    plt.ylabel("Error")
    plt.title(f"{value} +- {value_range}")
    plt.legend()
    if action == "save":
        plt.savefig(f"{save_folder}/compare_vs_mse({value}+-{value_range}).png")
        plt.clf()
    else:
        return plt
