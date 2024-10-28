import matplotlib
import platform
if platform.system() == 'Darwin':
    matplotlib.use('MacOSX')
import matplotlib.pylab as plt
from numpy import linspace
from scipy.stats import uniform, ncx2
from pyarv.non_central_chi_squared._approximate_non_central_chi_squared import construct_inverse_non_central_chi_squared_interpolated_polynomial_approximation

uniform_numbers = lambda n_samples: uniform.rvs(size=n_samples)
ncx2_exact = ncx2.ppf


def plot_non_central_chi_squared_polynomial_approximation(save_figure=False):
    """ Plots a polynomial approximation to the non-central chi-squared. """
    u = linspace(0.0, 1.0, 10000)[:-1]  # Excluding the end points.
    dof = 1.0
    non_centralities = [1.0, 10.0, 20.0]
    plt.clf()
    plt.ion()
    for non_centrality in non_centralities:
        ncx2_approx = construct_inverse_non_central_chi_squared_interpolated_polynomial_approximation(dof, n_intervals=4)
        plt.plot(u, ncx2.ppf(u, df=dof, nc=non_centrality), 'k--')
        plt.plot(u, ncx2_approx(u, non_centrality=non_centrality), 'k,')
    plt.show()


if __name__ == '__main__':
    print("Plotting a polynomial approximation to the non-central chi-squared distribution.")
    plot_non_central_chi_squared_polynomial_approximation()

