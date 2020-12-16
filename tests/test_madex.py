from unittest import TestCase
import inspect
from splib.diabetes.madex import mean_adjusted_exponent_error, graph_vs_mse


class TestMadex(TestCase):
    def test_always_passes(self):
        self.assertTrue(True)

    def test_one_pair(self):
        self.assertAlmostEqual(
            mean_adjusted_exponent_error([50], [40]),
            81.71,
            2
        )

    def test_returns_plot(self):
        plot = graph_vs_mse(50, 40)
        # assert type(plot) == 'test'
        assert inspect.ismodule(plot)
