#!/usr/bin/env python3
import textwrap
import numpy as np
from scipy.stats import norm
import io
import numpy.typing as npt


def fit_polynomal_approx_coefficients(*,
                                      poly_order: int,
                                      n_fit: int,
                                      table_size: int) -> npt.NDArray:
    """
    Find the coefficients of the polynomial approximation of the inverse Gaussian CDF by a least squares polynomial fit.
    Parameters
    ----------
    poly_order:
        Order of polynomial.
    n_fit:
        The number of points to sample in dyadic (exponentially distributed) to fit polynomial.
    table_size:
        The number of entries in the resulting table.

    Returns
    -------
    Coefficients:
        The polynomial coefficients.
    """
    n_poly = poly_order + 1  # Accounting for the constant term.
    a = np.zeros((n_poly, table_size), dtype=float)
    for m in range(1, table_size):
        u2 = 0.5 ** m
        u1 = u2 * 0.5 if m != (table_size - 1) else 0
        x = np.linspace(u1, u2, n_fit)
        x = 0.5 * (x[:-1] + x[1:])
        y = norm.ppf(x)
        a[:, m] = np.polyfit(x, y, deg=poly_order)[::-1]  # Finding the coefficients by weighted least squares.
    return a


if __name__ == "__main__":
    # We will write some of the coefficients to a header file for reading in with C code.
    poly_orders = [1, 3]
    table_sizes = [8, 16]
    n_fit = 1000
    for poly_order in poly_orders:
        for table_size in table_sizes:
            a = fit_polynomal_approx_coefficients(poly_order=poly_order,
                                                  n_fit=n_fit,
                                                  table_size=table_size)
            coeffs = u'\n'.join([f'const float poly_order_{poly_order}_table_size_{table_size}_coef_{order}[{table_size}] = {{{', '.join([float.hex(coeff) for coeff in coeffs])}}};' for order, coeffs in enumerate(a)])
            include_guard_def = f"POLY_ORDER_{poly_order}_TABLE_SIZE_{table_size}_H"
            s = textwrap.dedent(f"""\
            #ifndef {include_guard_def}
            #define {include_guard_def}
            
            #define POLYNOMIAL_ORDER {poly_order}
            #define TABLE_SIZE {table_size}
            
            """) + \
                f"{coeffs}" + \
                textwrap.dedent(f"""
                
            #endif // {include_guard_def} 
            """)
            with io.open(f'polynomial_coefficients_order_{poly_order}_table_size_{table_size}.h', 'w') as coefficients_header_file:
                coefficients_header_file.write(s)
