#!/usr/bin/env python3
import matplotlib as mpl

# mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import time

from pyarv.gaussian import Gaussian

if __name__ == "__main__":
    n_samples = 1_000
    u = np.linspace(0, 1, n_samples + 1, dtype=np.float32)[1:-1]
    z_exact = norm.ppf(u)
    z_approx_linear = np.empty_like(u)
    z_approx_cubic = np.empty_like(u)
    Gaussian(order=1, use_preallocated_output_array=True).transform(inputs=u, outputs=z_approx_linear)
    Gaussian(order=3, use_preallocated_output_array=True).transform(inputs=u, outputs=z_approx_cubic)
    plt.ion()
    plt.clf()
    plt.plot(u, z_exact, 'k-', label="Exact")
    plt.plot(u, z_approx_cubic, 'r-', label="Cubic")
    plt.plot(u, z_approx_linear, 'b-', label="Linear")
    plt.legend()
    plt.show()
    input("Press enter to exit")
