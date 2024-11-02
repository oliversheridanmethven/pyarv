# An example

The example showcases how to use the PyARV package.

!!! note "NumPy arrays"
    - We use numpy arrays, not lists. 
    - The arrays have a specific datatype, in this case `np.float32`.
    - The function needs a preallocated location to write its results into.

```python
import numpy as np
from pyarv.gaussian.approximation import Gaussian

n_samples = 10_000_000
u = np.random.uniform(size=n_samples + 1).astype(np.float32)
z_approx = Gaussian(order=1).transform(u)
print(f"{z_approx = }")
```

To see how much faster PyARV is on your system, run:
```bash
python -m pyarv.gaussian.demos.speed
python -m pyarv.non_central_chi_squared.demos.speed
```