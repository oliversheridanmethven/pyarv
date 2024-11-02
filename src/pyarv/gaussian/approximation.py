from pyarv.gaussian.gaussian_bindings import linear, cubic
from pyarv._type_hints.arrays import Array

from pyarv.approximations.polynomial_approximations import PolynomialApproximationTransformer

class Gaussian(PolynomialApproximationTransformer):
    """Polynomial approximation to the inverse CDF of the Gaussian distribution,  \( \Phi^{-1} \)."""
    
    def approximation(self,
                      *,
                      inputs: Array,
                      outputs: Array,
                      order: int = 1,
                      **kwargs
                      ) -> None:
        approximations = {1: linear,
                          3: cubic}
        approximations[order](input=inputs, output=outputs)
