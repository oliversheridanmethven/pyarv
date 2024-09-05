#!/usr/bin/env python3
import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import time

from pyarv.gaussian.polynomial import polynomial

if __name__ == "__main__":
    n_samples = 1_000
    u = np.linspace(0, 1, n_samples + 1, dtype=np.float32)[1:-1]
    z_exact = norm.ppf(u)
    z_approx = np.empty_like(u)
    polynomial(input=u, output=z_approx)
    plt.ion()
    plt.clf()
    plt.plot(u, z_exact, 'k-')
    plt.plot(u, z_approx, 'r-')
    plt.show()

    plt.clf()
    plt.plot(u, z_exact - z_approx)
    plt.show()

    n_samples = 10_000_000
    u = np.random.uniform(size=n_samples + 1).astype(np.float32)
    scipy_start = time.time()
    z_exact = norm.ppf(u)
    scipy_end = time.time()
    z_approx = np.empty_like(u)
    pyarv_start = time.time()
    polynomial(input=u, output=z_approx)
    pyarv_end = time.time()
    print(f"scipy = {scipy_end - scipy_start}")
    print(f"pyarv = {pyarv_end - pyarv_start}")
