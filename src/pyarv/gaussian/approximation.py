from pyarv.gaussian.gaussian_bindings import linear, cubic
from pyarv.type_hints.arrays import Array

from pyarv.approximations.polynomial_approximations import PolynomialApproximationTransformer

class Gaussian(PolynomialApproximationTransformer):
    """Polynomial approximation to the inverse CDF of the Gaussian distribution,  \( \Phi^{-1} \)."""
    
    def approximation(self,
                      *args, 
                      order: int,  # This will be removed with injection.
                      **kwargs
                      ) -> None:
        approximations = {1: linear,
                          3: cubic}
        if self.order not in approximations.keys():
            raise NotImplementedError(f"We have not yet implemented interfaced for higher order approximations. Currently we only support orders: {approximations.keys()}")
        approximations[self.order](*args, **kwargs)
