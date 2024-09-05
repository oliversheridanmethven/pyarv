#!/usr/bin/env python3
import unittest
import numpy as np
from pyarv.gaussian.polynomial import polynomial


class TestBasicProperties(unittest.TestCase):
    def test_median_is_zero(self):
        u = np.array([0.5], dtype=np.float32)
        z = u * np.nan
        polynomial(input=u, output=z)
        self.assertEqual([0], z.tolist(), "The Gaussian should have a median value of zero.")

    def test_below_median_negative(self):
        u = np.array([0.2], dtype=np.float32)
        z = u * np.nan
        polynomial(input=u, output=z)
        self.assertGreater([0], z.tolist(), "The Gaussian should have negative values below the median.")

    def test_above_median_positive(self):
        u = np.array([0.8], dtype=np.float32)
        z = u * np.nan
        polynomial(input=u, output=z)
        self.assertLess([0], z.tolist(), "The Gaussian should have positive values above the median.")


if __name__ == '__main__':
    unittest.main()
