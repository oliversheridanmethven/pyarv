from pyarv.gaussian.gaussian_bindings import polynomial as polynomial_
import numpy as np
import numpy.typing as npt


def polynomial(*, input: npt.NDArray[np.float32], output: npt.NDArray[np.float32]) -> None:
    """
    Polynomial approximation to the inverse CDF of the Gaussian ditribution.

    Parameters
    ----------
    input :
        Uniform random numbers in the range \( (0, 1) \).
    output :
        Approximate Gaussian random variables.

    Returns
    -------
    """
    polynomial_(input=input, output=output)
