from pyarv.gaussian.gaussian_bindings import linear, cubic
import numpy as np
import numpy.typing as npt
from typing import Annotated, Literal

Array = Annotated[npt.NDArray[np.float32], Literal["N"]]


def polynomial(*,
               input: Array,
               output: Array,
               order: int = 1
               ) -> None:
    r"""
    Polynomial approximation to the inverse CDF of the Gaussian distribution.

    Parameters
    ----------
    input:
        Uniform random numbers in the range \( (0, 1) \).
    output:
        Approximate Gaussian random variables. Must be pre-allocated
        and like `input`.
    order:
        The polynomial order to use:

        1 = linear.
        3 = cubic.
        
    Returns
    -------
    output:
        The approximate Gaussian random variables are written into `output`.
    """
    approximations = {1: linear,
                      3: cubic}
    approximations[order](input=input, output=output)
