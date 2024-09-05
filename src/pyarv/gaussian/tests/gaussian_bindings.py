#!/usr/bin/env python3
import unittest
import numpy as np
from pyarv.gaussian.polynomial import polynomial


class TestBasicProperties(unittest.TestCase):
    def test_median_is_zero(self):
        for order in [1, 3]:
            u = np.array([0.5], dtype=np.float32)
            z = u * np.nan
            polynomial(input=u, output=z, order=order)
            self.assertEqual([0], z.tolist(), f"The Gaussian approximation {order = } should have a median value of zero.")

    def test_below_median_negative(self):
        for order in [1, 3]:
            u = np.array([0.2], dtype=np.float32)
            z = u * np.nan
            polynomial(input=u, output=z, order=order)
            self.assertGreater([0], z.tolist(), f"The Gaussian approximation {order = } should have negative values below the median.")

    def test_above_median_positive(self):
        for order in [1, 3]:
            u = np.array([0.8], dtype=np.float32)
            z = u * np.nan
            polynomial(input=u, output=z, order=order)
            self.assertLess([0], z.tolist(), f"The Gaussian approximation {order = } should have positive values above the median.")


if __name__ == '__main__':
    unittest.main()
