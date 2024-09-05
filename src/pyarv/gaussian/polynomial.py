from pyarv.gaussian.gaussian_bindings import linear, cubic
import numpy as np
from nptyping import NDArray, Shape


def polynomial(*,
               input: NDArray[Shape["N"], np.float32],
               output: NDArray[Shape["N"], np.float32],
               order: int = 1
               ) -> None:
    """
    Polynomial approximation to the inverse CDF of the Gaussian ditribution.

    Parameters
    ----------
    input:
        Uniform random numbers in the range \( (0, 1) \).
    output:
        Approximate Gaussian random variables.
    order:
        The polynomial order to use:
        1 = linear.
        3 = cubic.
    Returns
    -------
    """
    approximations = {1: linear,
                      3: cubic}
    approximations[order](input=input, output=output)
