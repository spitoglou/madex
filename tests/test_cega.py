from unittest import TestCase
from splib.diabetes.cega import clarke_error_grid


class TestCega(TestCase):

    def test_zones(self):
        y_true = [50, 60, 100, 135, 150, 200, 250, 300]
        y_pred1 = [25, 45, 150, 160, 200, 300, 350, 390]
        y_pred2 = [75, 75, 50, 110, 100, 100, 150, 210]
        self.assertEqual(
            clarke_error_grid(y_true, y_pred1, "")[1],
            [3, 5, 0, 0, 0]
        )
        self.assertEqual(
            clarke_error_grid(y_true, y_pred2, "")[1],
            [1, 4, 0, 3, 0]
        )
