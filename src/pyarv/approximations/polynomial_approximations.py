"""
The interface we specify for our approximations.
"""

import abc
import numpy as np
from typing import Any

from pyarv.type_hints.arrays import Array
import warnings
class PolynomialApproximationTransformer(abc.ABC):
    """
    The general class all polynomial approximations should derive from.
    """
    
    def __init__(self, 
                 *args, 
                 use_preallocated_output_array: bool=False,
                 order: int = 1,
                 cache_table: bool =True, 
                 try_cached_table: bool =True,
                 **kwargs):
        """
        Create the transformer.
        
        Parameters
        ----------
        use_preallocated_output_array
            Whether to use a user provided array with preallocated memory.
        order
            The polynomial order.
            
            0 = constant.  
            1 = linear.  
            2 = quadratic.  
            3 = cubic.  
            ... etc. ...
        cache_table
            Should any coefficient tables be cached?
        try_cached_table
            Should coefficient table caches be tried? 
        """
        if order < 0:
            raise ValueError(f"The polynomial order must be positive, not {order = }")
        if not isinstance(use_preallocated_output_array, bool):
            raise ValueError(f"A boolean value must be specified, not {use_preallocated_output_array = }")
        if not isinstance(cache_table, bool):
            raise ValueError(f"A boolean value must be specified, not {cache_table = }")
        if not isinstance(try_cached_table, bool):
            raise ValueError(f"A boolean value must be specified, not {try_cached_table = }")
        self.use_preallocated_output_array = use_preallocated_output_array
        self.cache_table = cache_table
        self.try_cached_table = try_cached_table
        self.order = order
        
    @abc.abstractmethod
    def approximation(self,
                      *args,
                      **kwargs) -> None:
        """
        The underlying approximation and the interface between the Python
        and C functions. This may have additional parameters beyond those 
        described by `transform`. (Also responsible for generating coefficient tables
        and caching). 
        """
        ...
    
    def transform(self, 
                   inputs: Array, 
                   /, 
                   *, 
                   outputs: Array | None = None, 
                   **kwargs: dict[str, Any])\
            -> Array | None:
        """
        Use a polynomial approximation for the inverse transform method.
         
        Parameters
        ----------
        inputs
            Uniform random numbers in \( [0, 1] \). 
        outputs
            A pre-allocated array containing the outputs if requested. 
        kwargs
            Keyword arguments needed for any specific distribution's `approximation` implementation. 

        Returns
        -------
        outputs:
            Random variables from a desired distribution. 

        """
        if self.use_preallocated_output_array:
            if len(inputs) != len(outputs):
                raise TypeError(f"The {outputs = } must be the same length as the {inputs = }")
            if outputs.shape != inputs.shape:
                raise TypeError(f"The {outputs.shape} must match the {inputs.shape}")
            if not inputs.flags['C_CONTIGUOUS']:
                raise TypeError(f"The {inputs = } must be C-contiguous.")
            if not outputs.flags['C_CONTIGUOUS']:
                raise TypeError(f"The {outputs = } must be C-contiguous.")
            if inputs.dtype != np.float32:
                raise TypeError(f"The {inputs.dtype = } must be {np.float32}.")
            if outputs.dtype != inputs.dtype:
                raise TypeError(f"The {outputs.dtype = } must match {inputs.dtype = }.")
            if outputs is inputs:
                raise ValueError(f"Must use different objects for output, {inputs is outputs = }")
            if outputs.base is inputs or inputs.base is outputs:
                raise ValueError(f"The inputs and output arrays must not share any memory.")
            
        if not self.use_preallocated_output_array:
            if outputs is not None:
                raise ValueError(f"{self.use_preallocated_output_array = }, but {outputs = } has been provided.")
            outputs = np.empty_like(inputs)
            
        self.approximation(inputs=inputs, outputs=outputs, order=self.order, **kwargs)
        return outputs
        