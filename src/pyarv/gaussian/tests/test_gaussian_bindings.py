#!/usr/bin/env python3
import unittest
import numpy as np
from pyarv.gaussian.approximation import Gaussian
import scipy.integrate as integrate
from scipy.stats import norm


def scalar_approximation(*, u: float, order: int):
    u = np.array([u], dtype=np.float32)
    return Gaussian(order=order).transform(np.array([0.5], dtype=np.float32))


class TestBasicProperties(unittest.TestCase):
    
    def test_median_is_zero(self):
        for order in [1, 3]:
            z = Gaussian(order=order).transform(np.array([0.5], dtype=np.float32))
            self.assertEqual([0], z.tolist(), f"The Gaussian approximation {order = } should have a median value of zero.")

    def test_below_median_negative(self):
        for order in [1, 3]:
            z = Gaussian(order=order).transform(np.array([0.2], dtype=np.float32))
            self.assertGreater([0], z.tolist(), f"The Gaussian approximation {order = } should have negative values below the median.")

    def test_above_median_positive(self):
        for order in [1, 3]:
            z = Gaussian(order=order).transform(np.array([0.8], dtype=np.float32))
            self.assertLess([0], z.tolist(), f"The Gaussian approximation {order = } should have positive values above the median.")

    def test_low_relative_error(self):
        for order in [1, 3]:
            error_integral = integrate.quad(lambda u: norm.ppf(u) - scalar_approximation(u=u, order=order), 0, 1)
            self.assertLessEqual(0, error_integral[0] + error_integral[1])
            self.assertGreaterEqual(0, error_integral[0] - error_integral[1])

    def test_decreasing_relative_error(self):
        orders = [1, 3]
        l2_errors = [integrate.quad(lambda u: (norm.ppf(u) - scalar_approximation(u=u, order=order)) ** 2, 0, 1)[0] for order in orders]
        for order, difference in zip(orders[1:], np.diff(l2_errors)):
            self.assertLessEqual(difference, 0, f"The difference in l2 errors {difference = } should be decreasing for {order = }")


if __name__ == '__main__':
    unittest.main()
