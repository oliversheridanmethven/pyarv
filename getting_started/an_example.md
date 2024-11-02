# An example

The example showcases how to use the PyARV package.

!!! note "NumPy arrays"
    - We use numpy arrays, not lists. 
    - The arrays have a specific datatype, in this case `np.float32`.
    - The function needs a preallocated location to write its results into.

```python
import numpy as np
from pyarv.gaussian.approximation import polynomial

n_samples = 10_000_000
u = np.random.uniform(size=n_samples + 1).astype(np.float32)
z_approx = np.empty_like(u)
polynomial(inputs=u, outputs=z_approx, order=1)

print(f"{z_approx = }")
```

To see how much faster PyARV is on your system, run:
```bash
python -m pyarv.gaussian.demos.speed
```