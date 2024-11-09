# An example

The example showcases how to use the PyARV package.

!!! note "NumPy arrays"
    - We use numpy arrays, not lists. 
    - The arrays have a specific datatype, in this case `np.float32`.
    - The function needs a preallocated location to write its results into.

```{.python .annotate title="Example Gaussian transformation." hl_lines="7"}
import numpy as np  #(1)!
from pyarv.gaussian.approximation import Gaussian
from scipy.stats import norm

n_samples = 10_000_000  #(2)!
u = np.random.uniform(size=n_samples).astype(np.float32)  #(3)!
z_approx = Gaussian(order=1).transform(u)  #(4)!
z_exact = norm.ppf(u)  #(5)!
```

1.  NumPy is required for the `array` data type and the 
uniform random numbers which will be transformed.
2. This package is designed for applications which use **lots** 
of random numbers. 
3. Our approximations require 32-bit floats.
4. The PyARV method for transforming uniform random numbers
into approximate random numbers.
5. The regular method for transforming uniform random numbers
into exact random numbers.

To see how much faster PyARV is on your system, run:
```bash
python -m pyarv.gaussian.demos.speed
python -m pyarv.non_central_chi_squared.demos.speed
```