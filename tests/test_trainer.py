"""
"""

import unittest

from src.trainer import *


class TestTeacherRatioScheduler(unittest.TestCase):

    def test_1(self):
        scheduler = TeacherRatioScheduler(5, 1.0, 0.0)
        ratios = [1.00, 0.75, 0.50, 0.25, 0.00]
        for r in ratios:
            assert scheduler.ratio == r, f"Got {scheduler.ratio}. Expected {r}."
            scheduler.step()

    def test_2(self):
        scheduler = TeacherRatioScheduler(5, 1.0, 0.5)
        ratios = [1.000, 0.875, 0.750, 0.625, 0.500]
        for r in ratios:
            assert scheduler.ratio == r, f"Got {scheduler.ratio}. Expected {r}."
            scheduler.step()


if __name__ == "__main__":
    unittest.main()
