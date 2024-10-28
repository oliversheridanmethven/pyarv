import logging

from pyarv.non_central_chi_squared.non_central_chi_squared_bindings import linear
from pyarv.non_central_chi_squared._table_coefficients import generate_non_central_chi_squared_coefficients
import numpy as np
import numpy.typing as npt
from typing import Annotated, Literal

Array = Annotated[npt.NDArray[np.float32], Literal["N"]]

cached_tables = {}

def reset_cache():
    del cached_tables

def polynomial(*,
               input: Array,
               output: Array,
               order: int = 1,
               non_centralities: Array,
               degrees_of_freedom: float,
               cache_table: bool = True,
               try_cached_table: bool = True,
               ) -> None:
    r"""
    Polynomial approximation to the inverse CDF of the non-central \( \Chi^2 \) distribution.

    Parameters
    ----------
    input:
        Uniform random numbers in the range \( (0, 1) \).
    output:
        Approximate non-central \( \Chi^2 \) random variables. Must be pre-allocated
        and like `input`.
    order:
        The polynomial order to use:

        1 = linear.
    non_centralities:
        The non-centralities, positive (zero values correspond to the central \( chi^2 \) distribution).
    degrees_of_freedom:
        The degrees of freedom, strictly positive, assumed to be fixed for all values.

    Returns
    -------
    output:
        The approximate non-central \( \Chi^2 \) random variables are written into `output`.
    """
    approximations = {1: linear}
    if order not in approximations.keys():
        raise NotImplementedError(f"We have not yet implemented interfaced for higher order approximations. Currently we only support orders: {approximations.keys()}")
    polynomial_coefficients = None
    if try_cached_table:
        try:
            polynomial_coefficients = cached_tables[degrees_of_freedom]
        except KeyError:
            logging.info(f"There were no cached coefficients for {degrees_of_freedom = }")
    if polynomial_coefficients is None:
        logging.info(f"Generating coefficients for {degrees_of_freedom = }")
        polynomial_coefficients = generate_non_central_chi_squared_coefficients(n_intervals=16,
                                                                                n_interpolation_functions=16,
                                                                                n_polynomial_orders=order+1,
                                                                                dof=degrees_of_freedom,
                                                                                dtype=np.float32
                                                                                )
    assert polynomial_coefficients is not None
    if cache_table:
        logging.info(f"Caching the coefficients for {degrees_of_freedom = }")
        cached_tables[degrees_of_freedom] = polynomial_coefficients
    approximations[order](input=input,
                          output=output,
                          non_centralities=non_centralities,
                          degrees_of_freedom=degrees_of_freedom,
                          polynomial_coefficients=polynomial_coefficients)
