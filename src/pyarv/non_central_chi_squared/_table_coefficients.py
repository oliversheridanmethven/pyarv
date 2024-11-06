import numpy as np
from scipy.stats import ncx2, norm, chi2
from scipy.optimize import root_scalar

from pyarv._approximation_utils.approximating_polynomials import piecewise_polynomial_coefficients_in_half_interval
from pyarv.type_hints.arrays import Array

def generate_non_central_chi_squared_coefficients(*,
    n_intervals: int,
    n_interpolation_functions: int,
    n_polynomial_orders: int,
    dof: float,
    dtype: type = np.float32
    ) -> Array:
    """
    Generate coefficient tables for the non-central \( \chi^2 \) distribution.
    """
    n_halves=2
    lower_half, upper_half = [0, 1]
    polynomial_order = n_polynomial_orders - 1

    n_entries = n_halves * n_interpolation_functions * n_polynomial_orders * n_intervals
    polynomial_coefficients = np.reshape([None] * n_entries,
                                         shape=(n_halves,
                                                n_interpolation_functions,
                                                n_polynomial_orders,
                                                n_intervals))


    interpolation_function = lambda f: f ** 0.5
    interpolation_function_deriv_first = lambda f: 0.5 * f ** -0.5
    interpolation_function_deriv_second = lambda f: -0.25 * f ** -1.5

    interpolation_function_contour_spacing = 1.0 / (n_interpolation_functions - 1)
    interpolation_values = ([interpolation_function(1.0) - n * interpolation_function_contour_spacing for n in range(n_interpolation_functions - 1)] + [interpolation_function(0)])[::-1]  # interpolation key values
    interpolation_points = [0.0] + [root_scalar(lambda a: interpolation_function(a) - y, x0=0.5, bracket=[0.0, 1.0], fprime=interpolation_function_deriv_first, fprime2=interpolation_function_deriv_second).root for y in interpolation_values[1:-1]] + [1.0]  # non-centrality for interpolating functions
    # We approximate the function P
    functions_exact = [None] * n_interpolation_functions  # The exact functions
    functions_exact[0] = norm.ppf  # Limiting case as y -> 0
    # The following odd syntax with y=... ensures y is evaluated at declaration and not taken by reference:
    functions_exact[1:-1] = [lambda u, y=y_interpolation_points: np.sqrt(dof / (4.0 * y)) * (y / dof * ncx2.ppf(u, df=dof, nc=(1.0 - y) * dof / y) - 1.0) for y_interpolation_points in interpolation_points[1:-1]]
    functions_exact[-1] = lambda u: np.sqrt(dof / 4.0) * (chi2.ppf(u, df=dof) / dof - 1.0)


    for interpolation_function_index, exact_function in enumerate(functions_exact):
        f_lower = exact_function
        f_upper = lambda u, *args, **kwargs: f_lower(1.0 - u, *args, **kwargs)
        coeffs_lower = piecewise_polynomial_coefficients_in_half_interval(f_lower, n_intervals, polynomial_order)
        coeffs_upper = piecewise_polynomial_coefficients_in_half_interval(f_upper, n_intervals, polynomial_order)
        polynomial_coefficients[lower_half][interpolation_function_index] = coeffs_lower
        polynomial_coefficients[upper_half][interpolation_function_index] = coeffs_upper

    assert all([i is not None for i in polynomial_coefficients.flatten()]), f"The polynomial coefficients contain a missing value."
    return np.ascontiguousarray(polynomial_coefficients.flatten(order="C"), dtype=dtype)
