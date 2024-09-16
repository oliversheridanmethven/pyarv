#!/usr/bin/env python3
import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ncx2

from pyarv.non_central_chi_squared.polynomial import polynomial

if __name__ == "__main__":
    n_samples = 1_000
    u = np.linspace(0, 1, n_samples + 1, dtype=np.float32)[1:-1]
    order = 1
    degree_of_freedom = 10
    non_centrality = 1
    z_exact = ncx2.ppf(u, df=degree_of_freedom, nc=non_centrality)
    degrees_of_freedom = np.zeros_like(u) + degree_of_freedom
    non_centralities = np.zeros_like(u) + non_centrality
    z_approx_linear = np.empty_like(u)
    polynomial(input=u, output=z_approx_linear, order=order, non_centralities=non_centralities, degrees_of_freedom=degrees_of_freedom)

    plt.ion()
    plt.clf()
    plt.plot(u, z_exact, 'k-', label="Exact")
    plt.plot(u, z_approx_linear, 'b-', label="Linear")
    plt.legend()
    plt.show()
    input("Press enter to exit")
