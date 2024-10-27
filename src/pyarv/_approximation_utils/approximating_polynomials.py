"""
Some useful code to approximate inverse cumulative
distribution functions to produce approximate random
variables by the inverse transform method.
"""
import numpy as np
from numpy import array, zeros
from numpy.linalg import solve
from scipy.integrate import quad
from tqdm import tqdm as progressbar


def integrate(*args, **kwargs):
    """ A wrapper around the quadrature integration routine. """
    return quad(*args, **kwargs)[0]


def expected_value_in_interval(func, a, b, *args, **kwargs):
    """
    Calculates the expected value of a function inside an interval. 
    :param func: Function. 
    :param a: Float.
    :param b: Float.
    :return: Float.
    """
    return integrate(func, a, b, *args, **kwargs) / (b - a)


##### Piecewise constant approximations #####

def build_lookup_table(func, n_table_entries):
    """
    Builds a lookup table. 
    :param func: Function.
    :param n_table_entries: Int.
    :return: Array.
    """
    interval_width = 1.0 / n_table_entries
    lookup_table = zeros(n_table_entries)
    for n in progressbar(range(n_table_entries)):
        a = n * interval_width
        b = a + interval_width
        lookup_table[n] = expected_value_in_interval(func, a, b)
    return lookup_table


def construct_piecewise_constant_approximation(func, n_intervals):
    """
    Constructs a piecewise constant approximation.
    :param func: Function.
    :param n_intervals: Int.  
    :return: Function.
    """
    lookup_table = build_lookup_table(func, n_intervals)

    def piecewise_constant_approximation(u):
        """
        A piecewise constant approximation.
        :param u: Float.
        :return: Float.
        """
        return lookup_table[array(n_intervals * u).astype(int)]

    return piecewise_constant_approximation

##### Piecewise polynomial approximations #####

def optimal_polynomial_coefficients(func, polynomial_order, a, b):
    """
    Calculates the L2 optimal coefficients of a polynomial approximation to a function.
    :param func: Function.
    :param polynomial_order: Int.
    :param a: Float.
    :param b: Float.
    :return: Array.
    """
    B = [integrate(lambda u: u ** i * func(u), a, b) for i in range(polynomial_order + 1)]
    A = [[(b ** (i + j + 1) - a ** (i + j + 1)) / (i + j + 1.0) for i in range(polynomial_order + 1)] for j in range(polynomial_order + 1)]
    return solve(A, B)


def dyadic_intervals_in_half_interval(n_intervals):
    """
    Computed the dyadic intervals between [0, 1/2]
    :param n_intervals: Int.
    :return: List, e.g. [[1/2, 1/2], [1/4, 1/2], [1/8, 1/4], ... [0, 1/16]].
    """
    intervals = [[0.5 ** (i + 1), 0.5 ** i] for i in range(n_intervals)]
    intervals[0] = [0.5, 0.5]
    intervals[-1][0] = 0.0
    return intervals


def piecewise_polynomial_coefficients_in_half_interval(func, n_intervals, polynomial_order):
    """
    Computes the coefficients of a piecewise polynomial approximation
    using dyadic intervals in the interval [0, 1/2]
    :param func: Function.
    :param n_intervals: Int.
    :param polynomial_order: Int.
    :return:
    """
    intervals = dyadic_intervals_in_half_interval(n_intervals)
    coefficients = zeros((polynomial_order + 1, n_intervals))
    for i in range(n_intervals):
        a, b = intervals[i]
        coefficients[:, i] = optimal_polynomial_coefficients(func, polynomial_order, a, b) if a != b else func(b)
    return coefficients


def construct_index_of_dyadic_interval(n_intervals):
    """
    Construct a function which will return the dyadic indices.
    :param n_intervals: Int.
    :return: Function.
    """
    intervals = dyadic_intervals_in_half_interval(n_intervals)
    intervals_lower_values = [interval[0] for interval in intervals]

    def index_of_dyadic_interval(u):
        """
        Computes the index containing the value.
        :param u: Array.
        :return: Array.
        """
        if isinstance(u, float):
            u = array([u])
        # There are faster ways, but a simple search is good enough for Python.
        return array([next((k for k, v in enumerate(intervals_lower_values) if i >= v)) for i in u])

    return index_of_dyadic_interval


def construct_symmetric_piecewise_polynomial_approximation(func, n_intervals, polynomial_order):
    """
    Constructs a symmetric piecewise polynomial approximation.
    :param func: Function.
    :param n_intervals: Int.
    :param polynomial_order: Int.
    :return: Function.
    """

    coefficients = piecewise_polynomial_coefficients_in_half_interval(func, n_intervals, polynomial_order)
    index_of_dyadic_interval = construct_index_of_dyadic_interval(n_intervals)

    def piecewise_polynomial_approximation(u):
        """
        A piecewise polynomial approximation.
        :param u: Array.
        :return: Float.
        """
        if isinstance(u, (float, int)):
            u = array([u], dtype=float)
        requires_reflecting = (u > 0.5)
        u = (1.0 - u) * requires_reflecting + np.logical_not(requires_reflecting) * u
        interval = index_of_dyadic_interval(u)
        approximation = sum([coefficients[i][interval] * u ** i for i in range(polynomial_order + 1)])
        approximation = -approximation * requires_reflecting + approximation * np.logical_not(requires_reflecting)
        return approximation

    return piecewise_polynomial_approximation
