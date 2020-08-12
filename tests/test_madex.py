from unittest import TestCase
from ..sp_diab.madex import mean_adjusted_exponent_error


class TestMadex(TestCase):
    def test_always_passes(self):
        self.assertTrue(True)

    def test_one_pair(self):
        self.assertAlmostEqual(
            mean_adjusted_exponent_error([50], [40]),
            81.71,
            2
        )
