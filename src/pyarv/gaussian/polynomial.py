from pyarv.gaussian.gaussian_bindings import linear, cubic
from pyarv._type_hints.arrays import Array



def polynomial(*,
               input: Array,
               output: Array,
               order: int = 1
               ) -> None:
    r"""
    Polynomial approximation to the inverse CDF of the Gaussian distribution,  \( \Phi^{-1} \).

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
