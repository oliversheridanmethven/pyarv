import logging

from pyarv.non_central_chi_squared.non_central_chi_squared_bindings import linear
from pyarv.non_central_chi_squared._table_coefficients import generate_non_central_chi_squared_coefficients
import numpy as np
import numpy.typing as npt
from typing import Annotated, Literal
from pyarv.approximations.polynomial_approximations import PolynomialApproximationTransformer

Array = Annotated[npt.NDArray[np.float32], Literal["N"]]



class NonCentralChiSquared(PolynomialApproximationTransformer):
    """Polynomial approximation to the inverse CDF of the non-central \( \chi^2 \) distribution."""    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cached_tables = {}

    def _reset_cache(self):
        del self.cached_tables

    def approximation(self, 
                   *args,
                   order,
                   non_centralities: Array,
                   degrees_of_freedom: float,
                   **kwargs
                   ):
        r"""
        Specialism extending `PolynomialApproximationTransformer.transform`.
    
        Parameters
        ----------
        non_centralities:
            The non-centralities, positive (zero values correspond to the central \( \chi^2 \) distribution).
        degrees_of_freedom:
            The degrees of freedom, strictly positive, assumed to be fixed for all values.    
        """
        approximations = {1: linear}
        if self.order not in approximations.keys():
            raise NotImplementedError(f"We have not yet implemented interfaced for higher order approximations. Currently we only support orders: {approximations.keys()}")
        polynomial_coefficients = None
        if self.try_cached_table:
            try:
                polynomial_coefficients = self.cached_tables[degrees_of_freedom]
            except KeyError:
                logging.info(f"There were no cached coefficients for {degrees_of_freedom = }")
        if polynomial_coefficients is None:
            logging.info(f"Generating coefficients for {degrees_of_freedom = }")
            polynomial_coefficients = generate_non_central_chi_squared_coefficients(n_intervals=16,
                                                                                    n_interpolation_functions=16,
                                                                                    n_polynomial_orders=self.order+1,
                                                                                    dof=degrees_of_freedom,
                                                                                    dtype=np.float32
                                                                                    )
        assert polynomial_coefficients is not None
        if self.cache_table:
            logging.info(f"Caching the coefficients for {degrees_of_freedom = }")
            self.cached_tables[degrees_of_freedom] = polynomial_coefficients
        approximations[self.order](*args,
                               non_centralities=non_centralities,
                               degrees_of_freedom=degrees_of_freedom,
                               polynomial_coefficients=polynomial_coefficients,
                               **kwargs)
