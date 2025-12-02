import inspect
from unittest import TestCase

import numpy as np

from new_metric import (
    MADEXCalculator,
    MADEXParameters,
    graph_vs_mse,
    mean_adjusted_exponent_error,
)


class TestMadex(TestCase):
    def test_always_passes(self):
        self.assertTrue(True)

    def test_one_pair(self):
        self.assertAlmostEqual(mean_adjusted_exponent_error([50], [40]), 81.71, 2)

    def test_returns_plot(self):
        plot = graph_vs_mse(50, 40)
        assert inspect.ismodule(plot)


class TestMADEXParameters(TestCase):
    """Tests for MADEXParameters dataclass"""

    def test_default_values(self):
        """Test that default parameters match canonical values"""
        params = MADEXParameters()
        self.assertEqual(params.center, 125.0)
        self.assertEqual(params.critical_range, 55.0)
        self.assertEqual(params.slope, 100.0)

    def test_aliases(self):
        """Test that a, b, c aliases work correctly"""
        params = MADEXParameters(center=110, critical_range=40, slope=80)
        self.assertEqual(params.a, 110)
        self.assertEqual(params.b, 40)
        self.assertEqual(params.c, 80)

    def test_validation_center_too_low(self):
        """Test validation rejects center below range"""
        with self.assertRaises(ValueError) as ctx:
            MADEXParameters(center=30)
        self.assertIn("center", str(ctx.exception))

    def test_validation_center_too_high(self):
        """Test validation rejects center above range"""
        with self.assertRaises(ValueError) as ctx:
            MADEXParameters(center=300)
        self.assertIn("center", str(ctx.exception))

    def test_clinical_interpretation(self):
        """Test clinical interpretation output"""
        params = MADEXParameters()
        interp = params.get_clinical_interpretation()
        self.assertIn("center", interp)
        self.assertIn("critical_range", interp)
        self.assertIn("slope", interp)


class TestMADEXCalculator(TestCase):
    """Tests for MADEXCalculator class"""

    def test_calculator_default_params(self):
        """Test calculator uses default parameters"""
        calc = MADEXCalculator()
        self.assertEqual(calc.params.center, 125.0)
        self.assertEqual(calc.params.slope, 100.0)

    def test_calculator_custom_params(self):
        """Test calculator accepts custom parameters"""
        params = MADEXParameters(center=110, critical_range=40, slope=80)
        calc = MADEXCalculator(params)
        self.assertEqual(calc.params.center, 110)

    def test_calculate_matches_function(self):
        """Test calculator.calculate() matches mean_adjusted_exponent_error()"""
        y_true = [50, 100, 150, 200]
        y_pred = [55, 95, 160, 190]

        calc = MADEXCalculator()
        calc_result = calc.calculate(y_true, y_pred)
        func_result = mean_adjusted_exponent_error(y_true, y_pred)

        self.assertAlmostEqual(calc_result, func_result, places=6)

    def test_calculate_detailed_returns_breakdown(self):
        """Test calculate_detailed() returns expected keys"""
        calc = MADEXCalculator()
        result = calc.calculate_detailed([100, 150], [110, 140])

        self.assertIn("score", result)
        self.assertIn("exponents", result)
        self.assertIn("errors", result)
        self.assertIn("powered_errors", result)
        self.assertIn("params", result)

    def test_calculate_empty_array_raises(self):
        """Test that empty arrays raise ValueError"""
        calc = MADEXCalculator()
        with self.assertRaises(ValueError) as ctx:
            calc.calculate([], [])
        self.assertIn("empty", str(ctx.exception).lower())

    def test_calculate_mismatched_lengths_raises(self):
        """Test that mismatched array lengths raise ValueError"""
        calc = MADEXCalculator()
        with self.assertRaises(ValueError) as ctx:
            calc.calculate([1, 2, 3], [1, 2])
        self.assertIn("mismatch", str(ctx.exception).lower())


class TestConsistency(TestCase):
    """Tests ensuring consistency between implementations"""

    def test_defaults_match_across_modules(self):
        """Test that defaults match between madex.py and common/clinical_data.py"""
        from new_metric.common.clinical_data import MADEXParameters as LegacyParams
        from new_metric.madex import MADEXParameters as CoreParams

        core = CoreParams()
        legacy = LegacyParams()

        # Center/a should match
        self.assertEqual(core.center, legacy.a)
        # Critical range/b should match
        self.assertEqual(core.critical_range, legacy.b)
        # Slope/c should match (this was the bug we fixed!)
        self.assertEqual(core.slope, legacy.c)

    def test_metric_calculator_matches_core(self):
        """Test MetricCalculator.compute_madex matches MADEXCalculator"""
        from new_metric.common.statistical_utils import MetricCalculator

        y_true = [50, 100, 150, 200, 250]
        y_pred = [55, 95, 160, 190, 260]

        # Core calculator
        calc = MADEXCalculator()
        core_result = calc.calculate(y_true, y_pred)

        # MetricCalculator (should delegate to core)
        metric_calc = MetricCalculator()
        delegated_result = metric_calc.compute_madex(y_true, y_pred)

        self.assertAlmostEqual(core_result, delegated_result, places=6)

    def test_backward_compat_function_matches(self):
        """Test backward compatibility function matches core"""
        from new_metric.common.statistical_utils import compute_madex

        y_true = [70, 120, 180]
        y_pred = [65, 125, 175]

        core_result = mean_adjusted_exponent_error(y_true, y_pred)
        compat_result = compute_madex(y_true, y_pred)

        self.assertAlmostEqual(core_result, compat_result, places=6)
