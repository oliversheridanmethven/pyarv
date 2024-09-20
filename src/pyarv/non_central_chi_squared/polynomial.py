from pyarv.non_central_chi_squared.non_central_chi_squared_bindings import linear
import numpy as np
import numpy.typing as npt
from typing import Annotated, Literal

Array = Annotated[npt.NDArray[np.float32], Literal["N"]]


def polynomial(*,
               input: Array,
               output: Array,
               order: int = 1,
               non_centralities: Array,
               degrees_of_freedom: Array
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
        The degrees of freedom, strictly positive.

    Returns
    -------
    output:
        The approximate non-central \( \Chi^2 \) random variables are written into `output`.
    """
    approximations = {1: linear}
    approximations[order](input=input, output=output, non_centralities=non_centralities, degrees_of_freedom=degrees_of_freedom)
