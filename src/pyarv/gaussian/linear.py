from pyarv.gaussian.gaussian_bindings import linear as linear_
import numpy as np
import numpy.typing as npt


def linear(*, input: npt.NDArray[np.double], output: npt.NDArray[np.double]) -> None:
    """
    Linear approximation to the inverse CDF of the Gaussian ditribution.

    Parameters
    ----------
    input :
        Uniform random numbers in the range (0, 1).
    output :
        Approximate Gaussian random variables.

    Returns
    -------
    """
    linear_(input=input, output=output)
