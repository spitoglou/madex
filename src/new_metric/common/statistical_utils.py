"""
Statistical Utilities

Common statistical calculation functions shared between bootstrap and sensitivity
analysis modules, including MADEX computation, bootstrap analysis, significance
testing, and clinical detection functions.

Features:
- MADEX calculation with logging and validation
- Bootstrap confidence interval analysis
- Statistical significance testing utilities
- Clinical detection sensitivity functions
- Traditional metric calculations (RMSE, MAE, MAPE)
"""

import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.stats import wilcoxon

from .clinical_data import MADEXParameters
from .clinical_logging import ClinicalLogger


class MetricCalculator:
    """
    Calculator for MADEX and traditional metrics with logging capabilities.

    This class delegates MADEX calculation to the canonical MADEXCalculator
    implementation in new_metric.madex to ensure consistency.
    """

    def __init__(self, logger: Optional[ClinicalLogger] = None):
        """
        Initialize metric calculator

        Args:
            logger: Optional clinical logger for detailed output
        """
        self.logger = logger

    def compute_madex(
        self,
        y_true: Union[List, np.ndarray],
        y_pred: Union[List, np.ndarray],
        params: Optional[MADEXParameters] = None,
        enable_logging: bool = False,
    ) -> float:
        """
        Calculate MADEX (Mean Adjusted Exponent Error) metric.

        Delegates to the canonical MADEXCalculator implementation in madex.py
        to ensure consistent results across the library.

        Args:
            y_true: Array of true glucose values
            y_pred: Array of predicted glucose values
            params: MADEX parameters (defaults to standard clinical parameters)
            enable_logging: Whether to enable detailed logging

        Returns:
            MADEX score
        """
        # Import canonical implementation
        from ..madex import MADEXCalculator
        from ..madex import MADEXParameters as CoreMADEXParameters

        y_true = np.array(y_true, dtype=float)
        y_pred = np.array(y_pred, dtype=float)

        if len(y_true) != len(y_pred):
            raise ValueError(
                f"Length mismatch: y_true({len(y_true)}) != y_pred({len(y_pred)})"
            )

        # Convert legacy params to core params, or use defaults
        if params is None:
            core_params = None  # MADEXCalculator will use its defaults
        else:
            # Convert from legacy MADEXParameters (a,b,c) to core (center, critical_range, slope)
            core_params = CoreMADEXParameters(
                center=params.a, critical_range=params.b, slope=params.c
            )

        # Create calculator and compute
        calculator = MADEXCalculator(core_params)

        if enable_logging and self.logger:
            # Get effective parameters for logging
            effective_params = calculator.params
            glucose_range = f"{y_true.min():.1f}-{y_true.max():.1f} mg/dL"
            prediction_range = f"{y_pred.min():.1f}-{y_pred.max():.1f} mg/dL"
            self.logger.log_narrative(
                f"MADEX Calculation: Processing glucose values in range {glucose_range}, "
                f"predictions in range {prediction_range}"
            )

            # Log parameter interpretations
            param_interpretations = effective_params.get_clinical_interpretation()
            for param_name, interpretation in param_interpretations.items():
                value = getattr(effective_params, param_name)
                self.logger.log_parameter_interpretation(
                    param_name, value, interpretation
                )

        # Delegate to canonical implementation
        if enable_logging and self.logger:
            result = calculator.calculate_detailed(y_true, y_pred)
            madex_score = result["score"]

            self.logger.log_narrative(
                "Adaptive exponent calculation: Base exponent=2, "
                "modified by tanh((glucose-center)/critical_range) * (error/slope)"
            )
            self.logger.log_narrative(
                f"Exponent range: {result['exponents'].min():.3f} to {result['exponents'].max():.3f} "
                f"(higher = more penalty for errors)"
            )

            avg_error = np.mean(result["errors"])
            self.logger.log_narrative(
                f"Mean absolute error: {avg_error:.2f} mg/dL, MADEX score: {madex_score:.2f}"
            )

            self.logger.log_clinical_assessment(
                "VERIFIED", "Using canonical MADEXCalculator implementation"
            )

            return madex_score
        else:
            return calculator.calculate(y_true, y_pred)

    def compute_rmse(
        self, y_true: Union[List, np.ndarray], y_pred: Union[List, np.ndarray]
    ) -> float:
        """Calculate Root Mean Square Error"""
        y_true = np.array(y_true, dtype=float)
        y_pred = np.array(y_pred, dtype=float)
        return np.sqrt(np.mean((y_pred - y_true) ** 2))

    def compute_mae(
        self, y_true: Union[List, np.ndarray], y_pred: Union[List, np.ndarray]
    ) -> float:
        """Calculate Mean Absolute Error"""
        y_true = np.array(y_true, dtype=float)
        y_pred = np.array(y_pred, dtype=float)
        return np.mean(np.abs(y_pred - y_true))

    def compute_mape(
        self, y_true: Union[List, np.ndarray], y_pred: Union[List, np.ndarray]
    ) -> float:
        """Calculate Mean Absolute Percentage Error"""
        y_true = np.array(y_true, dtype=float)
        y_pred = np.array(y_pred, dtype=float)

        # Avoid division by zero
        mask = y_true != 0
        if not np.any(mask):
            raise ValueError("Cannot compute MAPE: all true values are zero")

        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    def compute_comparison_metrics(
        self, y_true: Union[List, np.ndarray], y_pred: Union[List, np.ndarray]
    ) -> Dict[str, float]:
        """
        Calculate all comparison metrics

        Returns:
            Dictionary with RMSE, MAE, and MAPE values
        """
        results = {
            "RMSE": self.compute_rmse(y_true, y_pred),
            "MAE": self.compute_mae(y_true, y_pred),
            "MAPE": self.compute_mape(y_true, y_pred),
        }

        if self.logger:
            self.logger.log_narrative(
                f"Comparison Metrics: RMSE={results['RMSE']:.2f}, "
                f"MAE={results['MAE']:.2f}, MAPE={results['MAPE']:.2f}%"
            )

        return results


class BootstrapAnalyzer:
    """
    Bootstrap analysis utilities for confidence interval estimation
    """

    def __init__(self, logger: Optional[ClinicalLogger] = None):
        """
        Initialize bootstrap analyzer

        Args:
            logger: Optional clinical logger for detailed output
        """
        self.logger = logger

    def bootstrap_metric(
        self,
        y_true: Union[List, np.ndarray],
        y_pred: Union[List, np.ndarray],
        metric_func: Callable,
        n_bootstrap: int = 1000,
        ci_level: float = 95,
    ) -> Tuple[float, float, float]:
        """
        Perform bootstrap analysis for any metric function

        Args:
            y_true: Array of true values
            y_pred: Array of predicted values
            metric_func: Function to calculate metric (takes y_true, y_pred)
            n_bootstrap: Number of bootstrap samples
            ci_level: Confidence interval level (e.g., 95 for 95%)

        Returns:
            Tuple of (mean_value, lower_ci, upper_ci)
        """
        y_true = np.array(y_true, dtype=float)
        y_pred = np.array(y_pred, dtype=float)
        n = len(y_true)

        if len(y_pred) != n:
            raise ValueError(f"Length mismatch: y_true({n}) != y_pred({len(y_pred)})")

        # Generate bootstrap samples
        bootstrap_values = []
        for _ in range(n_bootstrap):
            # Sample with replacement, ensuring consistent indexing
            indices = np.random.choice(n, n, replace=True)
            true_boot = y_true[indices]
            pred_boot = y_pred[indices]

            try:
                value = metric_func(true_boot, pred_boot)
                bootstrap_values.append(value)
            except (ValueError, RuntimeWarning):
                # Skip problematic bootstrap samples
                continue

        if len(bootstrap_values) < n_bootstrap * 0.9:
            warnings.warn(
                f"Only {len(bootstrap_values)}/{n_bootstrap} bootstrap samples succeeded"
            )

        bootstrap_values = np.array(bootstrap_values)

        # Calculate confidence intervals
        lower_percentile = (100 - ci_level) / 2
        upper_percentile = 100 - lower_percentile

        lower_ci = np.percentile(bootstrap_values, lower_percentile)
        upper_ci = np.percentile(bootstrap_values, upper_percentile)
        mean_value = np.mean(bootstrap_values)

        if self.logger:
            self.logger.log_narrative(
                f"Bootstrap analysis: {len(bootstrap_values)} samples, "
                f"{ci_level}% CI: {lower_ci:.3f}-{upper_ci:.3f}, mean: {mean_value:.3f}"
            )

        return mean_value, lower_ci, upper_ci

    def bootstrap_confidence_interval(
        self, values: Union[List, np.ndarray], ci_level: float = 95
    ) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence interval from pre-computed values

        Args:
            values: Array of bootstrap values
            ci_level: Confidence interval level

        Returns:
            Tuple of (lower_ci, upper_ci)
        """
        values = np.array(values)
        lower_percentile = (100 - ci_level) / 2
        upper_percentile = 100 - lower_percentile

        return np.percentile(values, lower_percentile), np.percentile(
            values, upper_percentile
        )

    def check_ci_overlap(
        self, ci1: Tuple[float, float], ci2: Tuple[float, float]
    ) -> bool:
        """
        Check if two confidence intervals overlap

        Args:
            ci1: First confidence interval (lower, upper)
            ci2: Second confidence interval (lower, upper)

        Returns:
            True if intervals overlap, False otherwise
        """
        lower1, upper1 = ci1
        lower2, upper2 = ci2

        return not (upper1 < lower2 or upper2 < lower1)


class SignificanceTester:
    """
    Statistical significance testing utilities
    """

    def __init__(self, logger: Optional[ClinicalLogger] = None):
        """
        Initialize significance tester

        Args:
            logger: Optional clinical logger for detailed output
        """
        self.logger = logger

    def extract_wilcoxon_results(self, result) -> Tuple[float, float]:
        """
        Safely extract statistic and p-value from wilcoxon result across scipy versions

        Args:
            result: Result from scipy.stats.wilcoxon

        Returns:
            Tuple of (statistic, p_value)
        """
        try:
            # For newer scipy versions (>=1.7)
            stat = float(result.statistic)
            p_val = float(result.pvalue)
            return stat, p_val
        except AttributeError:
            # For older scipy versions, result is a tuple (statistic, pvalue)
            stat = float(result[0])
            p_val = float(result[1])
            return stat, p_val

    def cohens_d(self, x: Union[List, np.ndarray], y: Union[List, np.ndarray]) -> float:
        """
        Calculate Cohen's d effect size for paired samples

        Args:
            x: First sample
            y: Second sample

        Returns:
            Cohen's d effect size
        """
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)

        if len(x) != len(y):
            raise ValueError(f"Sample size mismatch: {len(x)} != {len(y)}")

        diff = x - y
        return np.mean(diff) / np.std(diff, ddof=1)

    def interpret_p_value(self, p_value: float) -> str:
        """
        Interpret p-value with clinical significance levels

        Args:
            p_value: Statistical p-value

        Returns:
            Clinical interpretation string
        """
        if p_value < 0.001:
            return "highly significant (p < 0.001) - Very strong evidence of difference"
        elif p_value < 0.01:
            return "very significant (p < 0.01) - Strong evidence of difference"
        elif p_value < 0.05:
            return "significant (p < 0.05) - Evidence of difference"
        elif p_value < 0.10:
            return "marginally significant (p < 0.10) - Weak evidence of difference"
        else:
            return "not significant (p >= 0.10) - No evidence of difference"

    def interpret_effect_size(self, effect_size: float) -> str:
        """
        Interpret Cohen's d effect size

        Args:
            effect_size: Cohen's d value

        Returns:
            Effect size interpretation
        """
        abs_effect = abs(effect_size)

        if abs_effect < 0.2:
            return "negligible effect size"
        elif abs_effect < 0.5:
            return "small effect size"
        elif abs_effect < 0.8:
            return "medium effect size"
        else:
            return "large effect size"

    def wilcoxon_test_with_interpretation(
        self,
        x: Union[List, np.ndarray],
        y: Union[List, np.ndarray],
        metric_name: str = "metric",
    ) -> Dict[str, Union[float, str]]:
        """
        Perform Wilcoxon signed-rank test with clinical interpretation

        Args:
            x: First sample (e.g., Model A values)
            y: Second sample (e.g., Model B values)
            metric_name: Name of the metric being tested

        Returns:
            Dictionary with test results and interpretations
        """
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)

        # Perform Wilcoxon test
        result = wilcoxon(x, y)
        statistic, p_value = self.extract_wilcoxon_results(result)

        # Calculate effect size
        effect_size = self.cohens_d(x, y)

        # Interpret results
        p_interpretation = self.interpret_p_value(p_value)
        effect_interpretation = self.interpret_effect_size(effect_size)

        # Determine preferred model
        if x.mean() < y.mean():
            preferred_model = "Model A"
        else:
            preferred_model = "Model B"

        results = {
            "statistic": statistic,
            "p_value": p_value,
            "effect_size": effect_size,
            "p_interpretation": p_interpretation,
            "effect_interpretation": effect_interpretation,
            "preferred_model": preferred_model,
            "metric_name": metric_name,
        }

        if self.logger:
            self.logger.log_narrative(f"\n{metric_name} Statistical Analysis:")
            self.logger.log_narrative(f"  Wilcoxon statistic: {statistic:.2f}")
            self.logger.log_narrative(f"  P-value: {p_value:.4f}")
            self.logger.log_narrative(f"  Cohen's d (effect size): {effect_size:.2f}")
            self.logger.log_narrative(f"  Interpretation: {p_interpretation}")
            self.logger.log_narrative(f"  Effect size: {effect_interpretation}")
            self.logger.log_narrative(f"  Preferred model: {preferred_model}")

        return results


class ClinicalDetection:
    """
    Clinical detection sensitivity functions for glucose monitoring
    """

    def __init__(self, logger: Optional[ClinicalLogger] = None):
        """
        Initialize clinical detection analyzer

        Args:
            logger: Optional clinical logger for detailed output
        """
        self.logger = logger

    def sensitivity_hypoglycemia(
        self,
        y_true: Union[List, np.ndarray],
        y_pred: Union[List, np.ndarray],
        threshold: float = 70.0,
    ) -> float:
        """
        Calculate hypoglycemia detection sensitivity

        Args:
            y_true: True glucose values
            y_pred: Predicted glucose values
            threshold: Hypoglycemia threshold (default: 70 mg/dL)

        Returns:
            Sensitivity (0-1) or NaN if no hypoglycemic events
        """
        y_true = np.array(y_true, dtype=float)
        y_pred = np.array(y_pred, dtype=float)

        # Find true hypoglycemic events
        true_positives = np.sum((y_true < threshold) & (y_pred < threshold))
        total_positives = np.sum(y_true < threshold)

        if total_positives == 0:
            return np.nan

        sensitivity = true_positives / total_positives

        if self.logger:
            self.logger.log_narrative(
                f"Hypoglycemia detection (<{threshold} mg/dL): "
                f"{true_positives}/{total_positives} events detected "
                f"(sensitivity: {sensitivity:.2f})"
            )

        return sensitivity

    def sensitivity_hyperglycemia(
        self,
        y_true: Union[List, np.ndarray],
        y_pred: Union[List, np.ndarray],
        threshold: float = 180.0,
    ) -> float:
        """
        Calculate hyperglycemia detection sensitivity

        Args:
            y_true: True glucose values
            y_pred: Predicted glucose values
            threshold: Hyperglycemia threshold (default: 180 mg/dL)

        Returns:
            Sensitivity (0-1) or NaN if no hyperglycemic events
        """
        y_true = np.array(y_true, dtype=float)
        y_pred = np.array(y_pred, dtype=float)

        # Find true hyperglycemic events
        true_positives = np.sum((y_true > threshold) & (y_pred > threshold))
        total_positives = np.sum(y_true > threshold)

        if total_positives == 0:
            return np.nan

        sensitivity = true_positives / total_positives

        if self.logger:
            self.logger.log_narrative(
                f"Hyperglycemia detection (>{threshold} mg/dL): "
                f"{true_positives}/{total_positives} events detected "
                f"(sensitivity: {sensitivity:.2f})"
            )

        return sensitivity

    def clinical_risk_score(self, zone_counts: List[int]) -> int:
        """
        Calculate clinical risk score from Clarke Error Grid zone counts

        Args:
            zone_counts: List of counts for zones [A, B, C, D, E]

        Returns:
            Clinical risk score (higher = more risky)
        """
        # Zone weights: A=0 (accurate), B=1 (acceptable), C=2 (overcorrecting),
        # D=5 (failure to detect), E=5 (erroneous treatment)
        weights = [0, 1, 2, 5, 5]

        if len(zone_counts) != len(weights):
            raise ValueError(
                f"Expected {len(weights)} zone counts, got {len(zone_counts)}"
            )

        risk_score = sum(count * weight for count, weight in zip(zone_counts, weights))

        if self.logger:
            zone_names = [
                "A (Accurate)",
                "B (Acceptable)",
                "C (Overcorrecting)",
                "D (Failure to Detect)",
                "E (Erroneous)",
            ]
            zone_details = [
                f"{zone_names[i]}: {count}" for i, count in enumerate(zone_counts)
            ]
            self.logger.log_narrative(f"Clarke Error Grid zones: {zone_details}")
            self.logger.log_narrative(f"Clinical risk score: {risk_score}")

        return risk_score


# Convenience functions for backward compatibility
def compute_madex(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    a: float = 125.0,
    b: float = 55.0,
    c: float = 100.0,  # FIXED: was 40.0, now 100.0 to match canonical implementation
) -> float:
    """Backward compatibility function for MADEX calculation.

    Delegates to the canonical MADEXCalculator implementation.
    """
    from ..madex import mean_adjusted_exponent_error

    return mean_adjusted_exponent_error(
        y_true, y_pred, center=a, critical_range=b, slope=c
    )


def compute_rmse(
    y_true: Union[List, np.ndarray], y_pred: Union[List, np.ndarray]
) -> float:
    """Backward compatibility function for RMSE calculation"""
    calculator = MetricCalculator()
    return calculator.compute_rmse(y_true, y_pred)


def compute_mae(
    y_true: Union[List, np.ndarray], y_pred: Union[List, np.ndarray]
) -> float:
    """Backward compatibility function for MAE calculation"""
    calculator = MetricCalculator()
    return calculator.compute_mae(y_true, y_pred)


def bootstrap_metric(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    metric_func: Callable,
    n_bootstrap: int = 1000,
    ci: float = 95,
) -> Tuple[float, float, float]:
    """Backward compatibility function for bootstrap analysis"""
    analyzer = BootstrapAnalyzer()
    return analyzer.bootstrap_metric(y_true, y_pred, metric_func, n_bootstrap, ci)


def extract_wilcoxon_results(result) -> Tuple[float, float]:
    """Backward compatibility function for Wilcoxon result extraction"""
    tester = SignificanceTester()
    return tester.extract_wilcoxon_results(result)


def cohens_d(x: Union[List, np.ndarray], y: Union[List, np.ndarray]) -> float:
    """Backward compatibility function for Cohen's d calculation"""
    tester = SignificanceTester()
    return tester.cohens_d(x, y)


def sensitivity_hypoglycemia(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    threshold: float = 70.0,
) -> float:
    """Backward compatibility function for hypoglycemia sensitivity"""
    detector = ClinicalDetection()
    return detector.sensitivity_hypoglycemia(y_true, y_pred, threshold)


def sensitivity_hyperglycemia(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    threshold: float = 180.0,
) -> float:
    """Backward compatibility function for hyperglycemia sensitivity"""
    detector = ClinicalDetection()
    return detector.sensitivity_hyperglycemia(y_true, y_pred, threshold)


def clinical_risk_score(zone_counts: List[int]) -> int:
    """Backward compatibility function for clinical risk score"""
    detector = ClinicalDetection()
    return detector.clinical_risk_score(zone_counts)
