import numpy as np
from scipy.stats import norm, gamma, nbinom
import matplotlib.pyplot as plt

from cyclic_boosting.quantile_matching import (
    J_QPD_S,
    J_QPD_B,
    quantile_fit_gaussian,
    quantile_fit_gamma,
    quantile_fit_nbinom,
    quantile_fit_spline,
)


def test_J_QPD_S(is_plot):
    alpha = 0.2
    # Gamma-like
    qv_low = 1.12
    qv_median = 2.023
    qv_high = 3.322

    j_qpd_s = J_QPD_S(alpha, qv_low, qv_median, qv_high)

    np.testing.assert_almost_equal(j_qpd_s.ppf(0.2), qv_low, 3)
    np.testing.assert_almost_equal(j_qpd_s.ppf(0.5), qv_median, 3)
    np.testing.assert_almost_equal(j_qpd_s.ppf(0.8), qv_high, 3)
    # expectation from Gamma
    np.testing.assert_almost_equal(j_qpd_s.ppf(0.1), 0.79, 2)
    np.testing.assert_almost_equal(j_qpd_s.ppf(0.3), 1.42, 2)
    np.testing.assert_almost_equal(j_qpd_s.ppf(0.7), 2.78, 2)
    np.testing.assert_almost_equal(j_qpd_s.ppf(0.9), 4.18, 2)

    np.testing.assert_almost_equal(j_qpd_s.cdf(qv_low), 0.2, 3)
    np.testing.assert_almost_equal(j_qpd_s.cdf(qv_median), 0.5, 3)
    np.testing.assert_almost_equal(j_qpd_s.cdf(qv_high), 0.8, 3)
    # expectation from Gamma
    np.testing.assert_almost_equal(j_qpd_s.cdf(0.79), 0.1, 2)
    np.testing.assert_almost_equal(j_qpd_s.cdf(1.42), 0.3, 2)
    np.testing.assert_almost_equal(j_qpd_s.cdf(2.78), 0.7, 2)
    np.testing.assert_almost_equal(j_qpd_s.cdf(4.18), 0.9, 2)

    if is_plot:
        plt.plot([alpha, 0.5, 1 - alpha], [qv_low, qv_median, qv_high], "ro")
        xs = np.linspace(0.0, 1.0, 100)
        plt.plot(xs, j_qpd_s.ppf(xs))
        plt.savefig("J_QPD_S.png")
        plt.clf()

    j_qpd_s = J_QPD_S(alpha, qv_low, qv_median, qv_high, version="logistic")

    np.testing.assert_almost_equal(j_qpd_s.ppf(0.2), qv_low, 3)
    np.testing.assert_almost_equal(j_qpd_s.ppf(0.5), qv_median, 3)
    np.testing.assert_almost_equal(j_qpd_s.ppf(0.8), qv_high, 3)
    # deviate from Gamma expecation
    np.testing.assert_almost_equal(j_qpd_s.ppf(0.1), 0.76, 2)
    np.testing.assert_almost_equal(j_qpd_s.ppf(0.3), 1.42, 2)
    np.testing.assert_almost_equal(j_qpd_s.ppf(0.7), 2.78, 2)
    np.testing.assert_almost_equal(j_qpd_s.ppf(0.9), 4.28, 2)

    np.testing.assert_almost_equal(j_qpd_s.cdf(qv_low), 0.2, 3)
    np.testing.assert_almost_equal(j_qpd_s.cdf(qv_median), 0.5, 3)
    np.testing.assert_almost_equal(j_qpd_s.cdf(qv_high), 0.8, 3)
    # deviate from Gamma expecation
    np.testing.assert_almost_equal(j_qpd_s.cdf(0.76), 0.1, 2)
    np.testing.assert_almost_equal(j_qpd_s.cdf(1.42), 0.3, 2)
    np.testing.assert_almost_equal(j_qpd_s.cdf(2.78), 0.7, 2)
    np.testing.assert_almost_equal(j_qpd_s.cdf(4.28), 0.9, 2)

    if is_plot:
        plt.plot([alpha, 0.5, 1 - alpha], [qv_low, qv_median, qv_high], "ro")
        xs = np.linspace(0.0, 1.0, 100)
        plt.plot(xs, j_qpd_s.ppf(xs))
        plt.savefig("J_QPD_S_logistic.png")
        plt.clf()


def test_J_QPD_B(is_plot):
    alpha = 0.2
    # Gaussian-like
    qv_low = -0.878
    qv_median = 0.3
    qv_high = 1.478

    j_qpd_b = J_QPD_B(alpha, qv_low, qv_median, qv_high, -5.0, 10.0)

    np.testing.assert_almost_equal(j_qpd_b.ppf(0.2), qv_low, 3)
    np.testing.assert_almost_equal(j_qpd_b.ppf(0.5), qv_median, 3)
    np.testing.assert_almost_equal(j_qpd_b.ppf(0.8), qv_high, 3)
    # expect from Gaussian
    np.testing.assert_almost_equal(j_qpd_b.ppf(0.1), -1.49, 2)
    np.testing.assert_almost_equal(j_qpd_b.ppf(0.3), -0.43, 2)
    np.testing.assert_almost_equal(j_qpd_b.ppf(0.7), 1.03, 2)
    np.testing.assert_almost_equal(j_qpd_b.ppf(0.9), 2.09, 1)

    np.testing.assert_almost_equal(j_qpd_b.cdf(qv_low), 0.2, 3)
    np.testing.assert_almost_equal(j_qpd_b.cdf(qv_median), 0.5, 3)
    np.testing.assert_almost_equal(j_qpd_b.cdf(qv_high), 0.8, 3)
    # expect from Gaussian
    np.testing.assert_almost_equal(j_qpd_b.cdf(-1.49), 0.1, 2)
    np.testing.assert_almost_equal(j_qpd_b.cdf(-0.43), 0.3, 2)
    np.testing.assert_almost_equal(j_qpd_b.cdf(1.03), 0.7, 2)
    np.testing.assert_almost_equal(j_qpd_b.cdf(2.09), 0.9, 2)

    if is_plot:
        plt.plot([alpha, 0.5, 1 - alpha], [qv_low, qv_median, qv_high], "ro")
        xs = np.linspace(0.0, 1.0, 100)
        plt.plot(xs, j_qpd_b.ppf(xs))
        plt.savefig("J_QPD_B.png")
        plt.clf()

    j_qpd_b = J_QPD_B(alpha, qv_low, qv_median, qv_high, -5.0, 10.0, version="logistic")

    np.testing.assert_almost_equal(j_qpd_b.ppf(0.2), qv_low, 3)
    np.testing.assert_almost_equal(j_qpd_b.ppf(0.5), qv_median, 3)
    np.testing.assert_almost_equal(j_qpd_b.ppf(0.8), qv_high, 3)
    # deviate from Gaussian expectation
    np.testing.assert_almost_equal(j_qpd_b.ppf(0.1), -1.57, 2)
    np.testing.assert_almost_equal(j_qpd_b.ppf(0.3), -0.43, 2)
    np.testing.assert_almost_equal(j_qpd_b.ppf(0.7), 1.03, 2)
    np.testing.assert_almost_equal(j_qpd_b.ppf(0.9), 2.19, 2)

    np.testing.assert_almost_equal(j_qpd_b.cdf(qv_low), 0.2, 3)
    np.testing.assert_almost_equal(j_qpd_b.cdf(qv_median), 0.5, 3)
    np.testing.assert_almost_equal(j_qpd_b.cdf(qv_high), 0.8, 3)
    # deviate from Gaussian expectation
    np.testing.assert_almost_equal(j_qpd_b.cdf(-1.57), 0.1, 2)
    np.testing.assert_almost_equal(j_qpd_b.cdf(-0.43), 0.3, 2)
    np.testing.assert_almost_equal(j_qpd_b.cdf(1.03), 0.7, 2)
    np.testing.assert_almost_equal(j_qpd_b.cdf(2.19), 0.9, 2)

    if is_plot:
        plt.plot([alpha, 0.5, 1 - alpha], [qv_low, qv_median, qv_high], "ro")
        xs = np.linspace(0.0, 1.0, 100)
        plt.plot(xs, j_qpd_b.ppf(xs))
        plt.savefig("J_QPD_B_logistic.png")
        plt.clf()


def test_cdf_fit_gaussian(is_plot):
    quantiles = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    mu_exp = 0.3
    sigma_exp = 1.4
    quantile_values = norm.ppf(quantiles, mu_exp, sigma_exp)

    gaussian_fit_quantiles = quantile_fit_gaussian(quantiles, quantile_values)
    np.testing.assert_almost_equal(gaussian_fit_quantiles(0.2), -0.878, 3)
    np.testing.assert_almost_equal(gaussian_fit_quantiles(0.5), 0.3, 3)
    np.testing.assert_almost_equal(gaussian_fit_quantiles(0.8), 1.478, 3)

    gaussian_fit_quantiles_pdf = quantile_fit_gaussian(quantiles, quantile_values, mode="dist")
    np.testing.assert_almost_equal(gaussian_fit_quantiles_pdf.mean(), mu_exp, 3)
    np.testing.assert_almost_equal(gaussian_fit_quantiles_pdf.std(), sigma_exp, 3)

    gaussian_fit_quantiles_cdf = quantile_fit_gaussian(quantiles, quantile_values, mode="cdf")
    np.testing.assert_almost_equal(gaussian_fit_quantiles_cdf(-0.9), 0.196, 3)
    np.testing.assert_almost_equal(gaussian_fit_quantiles_cdf(0.3), 0.5, 3)
    np.testing.assert_almost_equal(gaussian_fit_quantiles_cdf(1.5), 0.804, 3)

    if is_plot:
        plt.plot(quantiles, quantile_values, "ro")
        xs = np.linspace(0.0, 1.0, 100)
        plt.plot(xs, gaussian_fit_quantiles(xs))
        plt.savefig("gaussian.png")
        plt.clf()


def test_cdf_fit_gamma(is_plot):
    quantiles = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    mu_exp = 2.3
    sigma_exp = 1.4
    alpha_exp = mu_exp * mu_exp / (sigma_exp * sigma_exp)
    beta_exp = mu_exp / (sigma_exp * sigma_exp)
    quantile_values = gamma.ppf(quantiles, alpha_exp, scale=1 / beta_exp)

    gamma_fit_quantiles = quantile_fit_gamma(quantiles, quantile_values)
    np.testing.assert_almost_equal(gamma_fit_quantiles(0.2), 1.12, 3)
    np.testing.assert_almost_equal(gamma_fit_quantiles(0.5), 2.023, 3)
    np.testing.assert_almost_equal(gamma_fit_quantiles(0.8), 3.322, 3)

    gamma_fit_quantiles_pdf = quantile_fit_gamma(quantiles, quantile_values, mode="dist")
    np.testing.assert_almost_equal(gamma_fit_quantiles_pdf.mean(), mu_exp, 3)
    np.testing.assert_almost_equal(gamma_fit_quantiles_pdf.std(), sigma_exp, 3)

    gamma_fit_quantiles_cdf = quantile_fit_gamma(quantiles, quantile_values, mode="cdf")
    np.testing.assert_almost_equal(gamma_fit_quantiles_cdf(1.1), 0.194, 3)
    np.testing.assert_almost_equal(gamma_fit_quantiles_cdf(2.0), 0.493, 3)
    np.testing.assert_almost_equal(gamma_fit_quantiles_cdf(3.3), 0.796, 3)

    if is_plot:
        plt.plot(quantiles, quantile_values, "ro")
        xs = np.linspace(0.0, 1.0, 100)
        plt.plot(xs, gamma_fit_quantiles(xs))
        plt.savefig("gamma.png")
        plt.clf()


def test_cdf_fit_nbinom(is_plot):
    quantiles = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    mu_exp = 5.3
    sigma_exp = 3.1
    n_exp = mu_exp * mu_exp / (sigma_exp * sigma_exp - mu_exp)
    p_exp = mu_exp / (sigma_exp * sigma_exp)
    quantile_values = nbinom.ppf(quantiles, n_exp, p_exp)

    np.random.seed(42)
    nbinom_fit_quantiles = quantile_fit_nbinom(quantiles, quantile_values)
    np.testing.assert_equal(nbinom_fit_quantiles(0.2), 3.0)
    np.testing.assert_equal(nbinom_fit_quantiles(0.5), 5.0)
    np.testing.assert_equal(nbinom_fit_quantiles(0.8), 8.0)

    nbinom_fit_quantiles_pdf = quantile_fit_nbinom(quantiles, quantile_values, mode="dist")
    np.testing.assert_almost_equal(nbinom_fit_quantiles_pdf.mean(), 5.850, 3)
    np.testing.assert_almost_equal(nbinom_fit_quantiles_pdf.std(), 3.234, 3)

    nbinom_fit_quantiles_cdf = quantile_fit_nbinom(quantiles, quantile_values, mode="cdf")
    np.testing.assert_almost_equal(nbinom_fit_quantiles_cdf(3), 0.251, 3)
    np.testing.assert_almost_equal(nbinom_fit_quantiles_cdf(5), 0.51, 3)
    np.testing.assert_almost_equal(nbinom_fit_quantiles_cdf(8), 0.809, 3)
    np.testing.assert_almost_equal(nbinom_fit_quantiles_cdf(8.2), 0.809, 3)

    if is_plot:
        plt.plot(quantiles, quantile_values, "ro")
        xs = np.linspace(0.0, 1.0, 100)
        plt.plot(xs, nbinom_fit_quantiles(xs))
        plt.savefig("nbinom.png")
        plt.clf()


def test_quantile_fit_spline(is_plot):
    quantiles = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    quantile_values = np.array([-1.7, 0.3, 1.5, 2.7, 5.8])

    spl = quantile_fit_spline(quantiles, quantile_values)
    np.testing.assert_almost_equal(spl(0.2), -0.567, 3)
    np.testing.assert_almost_equal(spl(0.5), 1.5, 3)
    np.testing.assert_almost_equal(spl(0.8), 3.877, 3)

    if is_plot:
        plt.plot(quantiles, quantile_values, "ro")
        xs = np.linspace(0, 1, 100)
        plt.plot(xs, spl(xs))
        plt.savefig("spline.png")
        plt.clf()
