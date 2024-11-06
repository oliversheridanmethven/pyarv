"""
Some useful code to approximate inverse cumulative
distribution functions to produce approximate random
variables by the inverse transform method.
"""
from typing import Callable

import numpy as np
from numpy import array, zeros
from numpy.linalg import solve
from scipy.integrate import quad
from tqdm import tqdm as progressbar
from pyarv.type_hints.arrays import Array


def _integrate(*args, **kwargs):
    """ A wrapper around the quadrature integration routine. """
    return quad(*args, **kwargs)[0]

def optimal_polynomial_coefficients(*,
                                    f: Callable,
                                    polynomial_order:int,
                                    lower_limit:float,
                                    upper_limit:float) -> Array:
    r"""
    Calculates the \( L^2 \) optimal coefficients of a polynomial approximation to a function \( f \colon (a, b) \to \mathbb{R} \).

    Parameters
    ----------
    f
        \( f \)
    polynomial_order
        Order of the polynomial approximation. 
    lower_limit
        \( a \) 
    upper_limit
        \( b \) 

    Returns
    -------
    coefficients
        Polynomial coefficients. 
    """
    B = [_integrate(lambda u: u ** i * f(u), lower_limit, upper_limit) for i in range(polynomial_order + 1)]
    A = [[(upper_limit ** (i + j + 1) - lower_limit ** (i + j + 1)) / (i + j + 1.0) for i in range(polynomial_order + 1)] for j in range(polynomial_order + 1)]
    return solve(A, B)


def dyadic_intervals_in_half_interval(n_intervals: int) -> list[list[float]]:
    """
    Computed the dyadic intervals in [0, 1/2].

    Parameters
    ----------
    n_intervals
        The number of intervals. 

    Returns
    -------
    intervals
        The dyadic intervals. e.g. `[[1/2, 1/2], [1/4, 1/2], [1/8, 1/4], ... [0, 1/16]]`
    """
    intervals = [[0.5 ** (i + 1), 0.5 ** i] for i in range(n_intervals)]
    intervals[0] = [0.5, 0.5]
    intervals[-1][0] = 0.0
    return intervals


def piecewise_polynomial_coefficients_in_half_interval(f: Callable, 
                                                       n_intervals: int, 
                                                       polynomial_order: int) -> Array:
    """
    Computes the coefficients of a piecewise polynomial approximation to a function \( f \)
    using dyadic intervals in \( [0, 1/2] \).
     
    Parameters
    ----------
    f
        \( f \). 
    n_intervals
        The number of intervals. 
    polynomial_order
        The polynomial order. 

    Returns
    -------
    Array
        The polynomial coefficient tables. 
    """
    intervals = dyadic_intervals_in_half_interval(n_intervals)
    coefficients = zeros((polynomial_order + 1, n_intervals))
    for i in range(n_intervals):
        a, b = intervals[i]
        coefficients[:, i] = optimal_polynomial_coefficients(f=f, polynomial_order=polynomial_order, lower_limit=a, upper_limit=b) if a != b else f(b)
    return coefficients
