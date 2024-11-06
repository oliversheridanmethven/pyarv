#!/usr/bin/env python3
import matplotlib
import platform
if platform.system() == 'Darwin':
    matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ncx2

from pyarv.non_central_chi_squared.approximation import NonCentralChiSquared

if __name__ == "__main__":
    n_samples = 1_000
    u = np.linspace(0, 1, n_samples + 1, dtype=np.float32)[1:-1]
    order = 1
    degrees_of_freedom = 10.0
    polynomial = NonCentralChiSquared(order=1)
    plt.ion()
    plt.clf()
    for non_centrality in [1.0, 10.0, 20.0]:
        z_exact = ncx2.ppf(u, df=degrees_of_freedom, nc=non_centrality)
        non_centralities = np.zeros_like(u) + non_centrality
        z_approx_linear = np.empty_like(u)
        z_approx_linear = polynomial.transform(u, non_centralities=non_centralities, degrees_of_freedom=degrees_of_freedom)
        plt.plot(u, z_exact, '--', label=f"Exact, $\\lambda = {non_centrality}$")
        plt.plot(u, z_approx_linear, ':', label="Linear")
    plt.legend()
    plt.show()
    input("Press enter to exit")
