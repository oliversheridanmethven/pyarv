#!/usr/bin/env python3
import itertools
import unittest
from idlelib.pyparse import trans

import numpy as np
from pyarv.approximations.polynomial_approximations import PolynomialApproximationTransformer
from pyarv.non_central_chi_squared.approximation import NonCentralChiSquared
import scipy.integrate as integrate
from scipy.stats import ncx2 as non_central_chi_squared


def scalar_approximation(*,
                         transformer: PolynomialApproximationTransformer, 
                         u: float,
                         degree_of_freedom: float,
                         non_centrality: float
                         ):
    u, non_centrality = [np.array([i], dtype=np.float32) for i in [u, non_centrality]]
    z = transformer.transform(u, degrees_of_freedom=degree_of_freedom, non_centralities=non_centrality)
    return z


class TestBasicProperties(unittest.TestCase):
    def test_low_relative_error(self):
        orders = [1]
        degrees_of_freedom = [1.0, 5.0, 10.0, 50.0]
        non_centralities = [1.0, 5.0, 10.0]
        for order, degree_of_freedom, non_centrality in itertools.product(orders, degrees_of_freedom, non_centralities):
            transformer = NonCentralChiSquared(order=order)
            error_integral = integrate.quad(lambda u: non_central_chi_squared.ppf(u, df=degree_of_freedom, nc=non_centrality) - scalar_approximation(transformer=transformer, u=u, degree_of_freedom=degree_of_freedom, non_centrality=non_centrality), 0, 1, limit=1000)
            self.assertLessEqual(0, error_integral[0] + error_integral[1], f"Expected a near zero error for {order = } {degree_of_freedom = } {non_centrality = }")
            self.assertGreaterEqual(0, error_integral[0] - error_integral[1], f"Expected a near zero error for {order = } {degree_of_freedom = } {non_centrality = }")

class TestResultsDiffer(unittest.TestCase):
    def test_degrees_of_freedom(self):
        order = 1
        transformer = NonCentralChiSquared(order=order)
        non_centralities = [1.0, 5.0, 10.0]
        np.random.seed(1)
        uniforms = np.random.uniform(size=10)
        for non_centrality in non_centralities:
            for u in uniforms:
                with self.subTest(f"{non_centralities = }"):
                    degrees_of_freedom = [5.0, 10.0]
                    self.assertNotEqual(scalar_approximation(transformer=transformer, u=u, degree_of_freedom=degrees_of_freedom[0], non_centrality=non_centrality),
                                        scalar_approximation(transformer=transformer, u=u, degree_of_freedom=degrees_of_freedom[1], non_centrality=non_centrality),
                                        f"Results should differ for  {degrees_of_freedom = }")
    def test_non_centralities(self):
        order = 1
        transformer = NonCentralChiSquared(order=order)
        degrees_of_freedom = [5.0, 10.0]
        np.random.seed(1)
        uniforms = np.array([0.3, 0.5, 0.7])
        non_centrality = 5.0
        for u in uniforms:
            with self.subTest(f"{u =}, {non_centrality = }"):
                self.assertNotEqual(scalar_approximation(transformer=transformer, u=u, degree_of_freedom=degrees_of_freedom[0], non_centrality=non_centrality),
                                    scalar_approximation(transformer=transformer, u=u, degree_of_freedom=degrees_of_freedom[1], non_centrality=non_centrality),
                                    f"Results should differ for  {degrees_of_freedom = }")


if __name__ == '__main__':
    unittest.main()
