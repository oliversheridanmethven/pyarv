#!/usr/bin/env python3
import itertools

import numpy as np
from scipy.odr import polynomial
from scipy.stats import ncx2
import time
import pandas as pd
from tqdm import tqdm as progressbar

from pyarv.non_central_chi_squared.approximation import NonCentralChiSquared

if __name__ == "__main__":
    nus = [1, 5, 10, 50, 100]
    lambdas = [1, 5, 10, 50, 100, 200, 500, 1000]
    order = 1
    polynomial = NonCentralChiSquared(order=order, use_preallocated_output_array=True)
    n_samples = 10
    u = np.random.uniform(size=n_samples).astype(np.float32)
    z_approx = np.empty_like(u)
    # We cache the tables for each DOF so their computation isn't timed.
    for degree_of_freedom in nus:
        polynomial.transform(u, outputs=z_approx, non_centralities=np.ones_like(u), degrees_of_freedom=degree_of_freedom)
    n_samples = 10_000 # Should only take a second or two in release mode.
    u = np.random.uniform(size=n_samples).astype(np.float32)
    z_approx = np.empty_like(u)
    res = {nu: {} for nu in nus}
    for degree_of_freedom, non_centrality in progressbar(list(itertools.product(nus, lambdas)), leave=False):
        start = time.time()
        z_exact = ncx2.ppf(u, df=degree_of_freedom, nc=non_centrality)
        elapsed_ncx2 = (time.time() - start) / n_samples
        z_approx = np.empty_like(u)
        degrees_of_freedom = degree_of_freedom
        non_centralities = np.zeros_like(u) + non_centrality
        start = time.time()
        polynomial.transform(u, outputs=z_approx, non_centralities=non_centralities, degrees_of_freedom=degrees_of_freedom)
        elapsed_approx = (time.time() - start) / n_samples
        res[degree_of_freedom][non_centrality] = int(elapsed_ncx2 / elapsed_approx)

    df = pd.DataFrame(res)
    df.index = df.index.rename('lambda')
    df.columns = df.columns.rename('nu')
    print("Speedup from using the approximation compared to SciPy.")
    print(df)
