#!/usr/bin/env python3
import numpy as np
from scipy.stats import norm
import time

from pyarv.gaussian.approximation import Gaussian

if __name__ == "__main__":
    for order in [1, 3]:
        print(f"\n{order = }\n" + "-"*50)
        n_samples = 1_000
        u = np.linspace(0, 1, n_samples + 1, dtype=np.float32)[1:-1]
        z_exact = norm.ppf(u)
        z_approx = np.empty_like(u)
        polynomial = Gaussian(order=1, use_preallocated_output_array=True)
        polynomial.transform(u, outputs=z_approx)

        n_samples = 10_000_000
        u = np.random.uniform(size=n_samples + 1).astype(np.float32)
        scipy_start = time.time()
        z_exact = norm.ppf(u)
        scipy_end = time.time()
        z_approx = np.empty_like(u)
        pyarv_start = time.time()
        polynomial.transform(u, outputs=z_approx)
        pyarv_end = time.time()
        print(f"The inverse transform method: ")
        print(f"scipy = {scipy_end - scipy_start}")
        print(f"pyarv = {pyarv_end - pyarv_start}")

        n_samples = 10_000_000
        numpy_start = time.time()
        np.random.normal(size=n_samples)  # Doesn't take a dtype argument.
        numpy_end = time.time()
        pyarv_numpy_start = time.time()
        u = np.random.uniform(size=n_samples + 1).astype(np.float32)
        z_approx = np.empty_like(u)
        polynomial.transform(u, outputs=z_approx)
        pyarv_numpy_end = time.time()
        pyarv_start = time.time()
        polynomial.transform(u, outputs=z_approx)
        pyarv_end = time.time()
        uniforms_start = time.time()
        u = np.random.uniform(size=n_samples + 1).astype(np.float32)
        uniforms_end = time.time()
        print(f"Random number generation: ")
        print(f"numpy = {numpy_end - numpy_start}")
        print(f"pyarv + uniforms = {pyarv_numpy_end - pyarv_numpy_start}")
        print(f"uniforms = {uniforms_end - uniforms_start}")
        print(f"pyarv without uniforms = {pyarv_end - pyarv_start}")
        print("\n")
