import numpy as np
from scipy.stats import norm, gamma, nbinom
import matplotlib.pyplot as plt

from cyclic_boosting.quantile_matching import cdf_fit_gaussian, cdf_fit_gamma, cdf_fit_nbinom, cdf_fit_spline


def test_cdf_fit_gaussian(is_plot):
    quantiles = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    mu_exp = 0.3
    sigma_exp = 1.4
    cdf_values = norm.ppf(quantiles, mu_exp, sigma_exp)

    gaussian_fit_quantiles = cdf_fit_gaussian(quantiles, cdf_values)
    np.testing.assert_almost_equal(gaussian_fit_quantiles(0.2), -0.878, 3)
    np.testing.assert_almost_equal(gaussian_fit_quantiles(0.5), 0.3, 3)
    np.testing.assert_almost_equal(gaussian_fit_quantiles(0.8), 1.478, 3)

    gaussian_fit_quantiles_pdf = cdf_fit_gaussian(quantiles, cdf_values, mode="dist")
    np.testing.assert_almost_equal(gaussian_fit_quantiles_pdf.mean(), mu_exp, 3)
    np.testing.assert_almost_equal(gaussian_fit_quantiles_pdf.std(), sigma_exp, 3)

    gaussian_fit_quantiles_cdf = cdf_fit_gaussian(quantiles, cdf_values, mode="cdf")
    np.testing.assert_almost_equal(gaussian_fit_quantiles_cdf(-0.9), 0.196, 3)
    np.testing.assert_almost_equal(gaussian_fit_quantiles_cdf(0.3), 0.5, 3)
    np.testing.assert_almost_equal(gaussian_fit_quantiles_cdf(1.5), 0.804, 3)

    if is_plot:
        plt.plot(cdf_values, quantiles, "ro")
        xs = np.linspace(cdf_values.min(), cdf_values.max(), 100)
        plt.plot(xs, gaussian_fit_quantiles_cdf(xs))
        plt.savefig("gaussian.png")
        plt.clf()


def test_cdf_fit_gamma(is_plot):
    quantiles = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    mu_exp = 2.3
    sigma_exp = 1.4
    alpha_exp = mu_exp * mu_exp / (sigma_exp * sigma_exp)
    beta_exp = mu_exp / (sigma_exp * sigma_exp)
    cdf_values = gamma.ppf(quantiles, alpha_exp, scale=1 / beta_exp)

    gamma_fit_quantiles = cdf_fit_gamma(quantiles, cdf_values)
    np.testing.assert_almost_equal(gamma_fit_quantiles(0.2), 1.12, 3)
    np.testing.assert_almost_equal(gamma_fit_quantiles(0.5), 2.023, 3)
    np.testing.assert_almost_equal(gamma_fit_quantiles(0.8), 3.322, 3)

    gamma_fit_quantiles_pdf = cdf_fit_gamma(quantiles, cdf_values, mode="dist")
    np.testing.assert_almost_equal(gamma_fit_quantiles_pdf.mean(), mu_exp, 3)
    np.testing.assert_almost_equal(gamma_fit_quantiles_pdf.std(), sigma_exp, 3)

    gamma_fit_quantiles_cdf = cdf_fit_gamma(quantiles, cdf_values, mode="cdf")
    np.testing.assert_almost_equal(gamma_fit_quantiles_cdf(1.1), 0.194, 3)
    np.testing.assert_almost_equal(gamma_fit_quantiles_cdf(2.0), 0.493, 3)
    np.testing.assert_almost_equal(gamma_fit_quantiles_cdf(3.3), 0.796, 3)

    if is_plot:
        plt.plot(cdf_values, quantiles, "ro")
        xs = np.linspace(0, cdf_values.max(), 100)
        plt.plot(xs, gamma_fit_quantiles_cdf(xs))
        plt.savefig("gamma.png")
        plt.clf()


def test_cdf_fit_nbinom(is_plot):
    quantiles = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    mu_exp = 5.3
    sigma_exp = 3.1
    n_exp = mu_exp * mu_exp / (sigma_exp * sigma_exp - mu_exp)
    p_exp = mu_exp / (sigma_exp * sigma_exp)
    cdf_values = nbinom.ppf(quantiles, n_exp, p_exp)

    np.random.seed(42)
    nbinom_fit_quantiles = cdf_fit_nbinom(quantiles, cdf_values)
    np.testing.assert_equal(nbinom_fit_quantiles(0.2), 3.0)
    np.testing.assert_equal(nbinom_fit_quantiles(0.5), 5.0)
    np.testing.assert_equal(nbinom_fit_quantiles(0.8), 8.0)

    nbinom_fit_quantiles_pdf = cdf_fit_nbinom(quantiles, cdf_values, mode="dist")
    np.testing.assert_almost_equal(nbinom_fit_quantiles_pdf.mean(), 5.850, 3)
    np.testing.assert_almost_equal(nbinom_fit_quantiles_pdf.std(), 3.234, 3)

    nbinom_fit_quantiles_cdf = cdf_fit_nbinom(quantiles, cdf_values, mode="cdf")
    np.testing.assert_almost_equal(nbinom_fit_quantiles_cdf(3), 0.251, 3)
    np.testing.assert_almost_equal(nbinom_fit_quantiles_cdf(5), 0.51, 3)
    np.testing.assert_almost_equal(nbinom_fit_quantiles_cdf(8), 0.809, 3)
    np.testing.assert_almost_equal(nbinom_fit_quantiles_cdf(8.2), 0.809, 3)

    if is_plot:
        plt.plot(cdf_values, quantiles, "ro")
        xs = np.linspace(0, cdf_values.max(), 100)
        plt.plot(xs, nbinom_fit_quantiles_cdf(xs))
        plt.savefig("nbinom.png")
        plt.clf()


def test_cdf_fit_spline(is_plot):
    quantiles = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    cdf_values = np.array([-1.7, 0.3, 1.5, 2.7, 5.8])

    spl = cdf_fit_spline(quantiles, cdf_values)
    np.testing.assert_almost_equal(spl(0.2), -0.567, 3)
    np.testing.assert_almost_equal(spl(0.5), 1.5, 3)
    np.testing.assert_almost_equal(spl(0.8), 3.877, 3)

    if is_plot:
        plt.plot(quantiles, cdf_values, "ro")
        xs = np.linspace(0, 1, 100)
        plt.plot(xs, spl(xs))
        plt.savefig("spline.png")
        plt.clf()
