#!/usr/bin/env python3
import itertools
import unittest
import numpy as np
from pyarv.non_central_chi_squared.polynomial import polynomial
import scipy.integrate as integrate
from scipy.stats import ncx2 as non_central_chi_squared


def scalar_approximation(*,
                         u: float,
                         order: int,
                         degree_of_freedom: float,
                         non_centrality: float
                         ):
    u, degree_of_freedom, non_centrality = [np.array([i], dtype=np.float32) for i in [u, degree_of_freedom, non_centrality]]
    z = u * np.nan
    polynomial(input=u, output=z, order=order, degrees_of_freedom=degree_of_freedom, non_centralities=non_centrality)
    return z


class TestBasicProperties(unittest.TestCase):
    def test_low_relative_error(self):
        orders = [1]
        degrees_of_freedom = [1, 5, 10, 50]
        non_centralities = [1, 5, 10]
        for order, degree_of_freedom, non_centrality in itertools.product(orders, degrees_of_freedom, non_centralities):
            error_integral = integrate.quad(lambda u: non_central_chi_squared.ppf(u, df=degree_of_freedom, nc=non_centrality) - scalar_approximation(u=u, order=order, degree_of_freedom=degree_of_freedom, non_centrality=non_centrality), 0, 1, limit=1000)
            self.assertLessEqual(0, error_integral[0] + error_integral[1], f"Expected a near zero error for {order = } {degree_of_freedom = } {non_centrality = }")
            self.assertGreaterEqual(0, error_integral[0] - error_integral[1], f"Expected a near zero error for {order = } {degree_of_freedom = } {non_centrality = }")


if __name__ == '__main__':
    unittest.main()
