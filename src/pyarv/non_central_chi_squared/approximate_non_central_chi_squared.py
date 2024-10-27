"""
Author:

    Oliver Sheridan-Methven September 2020.

Description:

    Approximation of the non-central chi-squared.
"""

from pyarv._approximation_utils.approximating_polynomials import piecewise_polynomial_coefficients_in_half_interval, construct_index_of_dyadic_interval
import numpy as np
from scipy.stats import ncx2, norm, chi2
from scipy.optimize import root_scalar
from bisect import bisect
from tqdm import tqdm as progressbar


def dyadic_function_approximation_constructor(func, n_intervals, polynomial_order):
    """
    Constructs a piecewise polynomial approximation which is piecewise
    L2 optimal on dyadic intervals on the intervals [0, 1/2) and [1,2, 1].
    :param func: Function.
    :param n_intervals: Int.
    :param polynomial_order: Int.
    :return: Function.
    """
    f_lower = func
    f_upper = lambda u, *args, **kwargs: func(1.0 - u, *args, **kwargs)
    coeffs_lower = piecewise_polynomial_coefficients_in_half_interval(f_lower, n_intervals, polynomial_order)
    coeffs_upper = piecewise_polynomial_coefficients_in_half_interval(f_upper, n_intervals, polynomial_order)
    index_of_dyadic_interval = construct_index_of_dyadic_interval(n_intervals)

    def inverse_cumulative_distribution_function_approximation(u):
        """
        Polynomial approximation of the inverse cumulative distribution function.
        :param u: Array.
        :return: Array.
        """
        in_lower = (u < 0.5)
        u = u * in_lower + np.logical_not(in_lower) * (1.0 - u)
        interval = index_of_dyadic_interval(u)
        y_lower, y_upper = [sum([coeffs[i][interval] * u ** i for i in range(polynomial_order + 1)]) for coeffs in [coeffs_lower, coeffs_upper]]
        y = y_lower * in_lower + y_upper * np.logical_not(in_lower)
        return y

    return inverse_cumulative_distribution_function_approximation


def construct_inverse_non_central_chi_squared_interpolated_polynomial_approximation(dof, polynomial_order=1, n_intervals=16, n_interpolating_functions=16):
    """
    Computes a polynomial approximation to the inverse cumulative distribution function for the non-central
    chi-squared distribution for a fixed number of degrees of freedom. The approximation is parametrised
    by a non-central parameter
    :param dof: Float.
    :param polynomial_order: Int.
    :param n_intervals: Int.
    :param n_interpolating_functions: Int.
    :return: Function.
    """
    interpolation_function = lambda f: f ** 0.5
    interpolation_function_deriv_first = lambda f: 0.5 * f ** -0.5
    interpolation_function_deriv_second = lambda f: -0.25 * f ** -1.5

    interpolation_function_contour_spacing = 1.0 / (n_interpolating_functions - 1)
    interpolation_values = ([interpolation_function(1.0) - n * interpolation_function_contour_spacing for n in range(n_interpolating_functions - 1)] + [interpolation_function(0)])[::-1]  # interpolation key values
    interpolation_points = [0.0] + [root_scalar(lambda a: interpolation_function(a) - y, x0=0.5, bracket=[0.0, 1.0], fprime=interpolation_function_deriv_first, fprime2=interpolation_function_deriv_second).root for y in interpolation_values[1:-1]] + [1.0]  # non-centrality for interpolating functions
    # We approximate the function P
    functions_exact = [None] * n_interpolating_functions  # The exact functions
    functions_exact[0] = norm.ppf  # Limiting case as y -> 0
    # The following odd syntax with y=... ensures y is evaluated at declaration and not taken by reference:
    functions_exact[1:-1] = [lambda u, y=y_interpolation_points: np.sqrt(dof / (4.0 * y)) * (y / dof * ncx2.ppf(u, df=dof, nc=(1.0 - y) * dof / y) - 1.0) for y_interpolation_points in interpolation_points[1:-1]]
    functions_exact[-1] = lambda u: np.sqrt(dof / 4.0) * (chi2.ppf(u, df=dof) / dof - 1.0)
    functions_approx = [dyadic_function_approximation_constructor(f, n_intervals, polynomial_order) for f in progressbar(functions_exact)]  # By piecewise dyadic construction

    def construct_linear_interpolation(functions, weightings):
        """
        Builds a linear interpolation between two functions.
        :param functions: List.
        :param weightings: List.
        :return: Function.
        """
        f1, f2 = functions
        w1, w2 = weightings
        return lambda u: f1(u) * w1 + f2(u) * w2

    def get_interpolation_functions_and_weightings(non_centrality):
        """
        Determines the interpolation functions to use and their weights.
        :param non_centrality: Float.
        :return: List.
        """
        interpolation_value = interpolation_function(non_centrality)
        insertion_index = bisect(interpolation_values, interpolation_value, lo=0)
        lower_index, upper_index = insertion_index - 1, insertion_index
        assert lower_index >= 0
        assert upper_index <= len(interpolation_values)
        if upper_index == len(interpolation_values):
            return [[functions_approx[lower_index]] * 2, [1.0, 0.0]]
        functions = [functions_approx[i] for i in [lower_index, upper_index]]
        interpolation_lower, interpolation_upper = [interpolation_values[i] for i in [lower_index, upper_index]]
        w_lower = (interpolation_upper - interpolation_value) / (interpolation_upper - interpolation_lower)
        w_upper = 1.0 - w_lower
        weights = [w_lower, w_upper]
        return [functions, weights]

    def inverse_non_central_chi_squared_interpolated_polynomial_approximation(u, non_centrality):
        """
        Polynomial approximation to the inverse cumulative distribution function for the non-central
        chi-squared distribution
        :param u: Array.
        :param non_centrality: Float.
        :return: Array.
        """
        functions, weightings = get_interpolation_functions_and_weightings(dof / (non_centrality + dof))
        interpolated_function = construct_linear_interpolation(functions, weightings)
        return non_centrality + dof + 2.0 * np.sqrt(non_centrality + dof) * interpolated_function(u)

    return inverse_non_central_chi_squared_interpolated_polynomial_approximation
