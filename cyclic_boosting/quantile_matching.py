import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm, gamma, nbinom
from scipy.interpolate import InterpolatedUnivariateSpline

from typing import Optional


def cdf_fit_gaussian(quantiles: np.ndarray, cdf_values: np.ndarray, mode: Optional[str] = "ppf") -> callable:
    """
    Interpolation between pairs of CDF values (potentially estimated by means
    of quantile regression) and corresponding quantiles according to a
    Gaussian distribution as assumed PDF.

    Parameters
    ----------
    quantiles : np.ndarray
        quantile values
    cdf_values : np.ndarray
        CDF values corresponding to quantile values
    mode : str
        decides about kind of returned callable, possible values are:

            - ``ppf``: input quantile, output CDF value (default)
            - ``dist``: fitted Gaussian (scipy function)
            - ``cdf``: input CDF value, output quantile

    Returns
    -------
    callable
        fitted Gaussian function (see mode)
    """

    def f(x, mu, sigma):
        return norm(loc=mu, scale=sigma).cdf(x)

    mu, sigma = curve_fit(f, cdf_values, quantiles)[0]
    if mode == "ppf":
        return norm(mu, sigma).ppf
    elif mode == "dist":
        return norm(mu, sigma)
    elif mode == "cdf":
        return norm(mu, sigma).cdf
    else:
        raise Exception("Invalid mode.")


def cdf_fit_gamma(quantiles: np.ndarray, cdf_values: np.ndarray, mode: Optional[str] = "ppf") -> callable:
    """
    Interpolation between pairs of CDF values (potentially estimated by means
    of quantile regression) and corresponding quantiles according to a
    Gamma distribution as assumed PDF (i.e., continuous, non-negative target
    values).

    Parameters
    ----------
    quantiles : np.ndarray
        quantile values
    cdf_values : np.ndarray
        CDF values corresponding to quantile values
    mode : str
        decides about kind of returned callable, possible values are:

            - ``ppf``: input quantile, output CDF value (default)
            - ``dist``: fitted Gamma (scipy function)
            - ``cdf``: input CDF value, output quantile

    Returns
    -------
    callable
        fitted Gamma function (see mode)
    """

    def f(x, alpha, beta):
        return gamma(alpha, scale=1 / beta).cdf(x)

    alpha, beta = curve_fit(f, cdf_values, quantiles, p0=[2.0, 0.9])[0]
    if mode == "ppf":
        return gamma(alpha, scale=1 / beta).ppf
    elif mode == "dist":
        return gamma(alpha, scale=1 / beta)
    elif mode == "cdf":
        return gamma(alpha, scale=1 / beta).cdf
    else:
        raise Exception("Invalid mode.")


def _nbinom_cdf_mu_var(x: float, mu: float, var: float) -> callable:
    """
    Calculation of negative binomial parameters n and p from given mean and
    variance, and subsequent call of cumulative distribution function.
    Parameters
    ----------
    x : float
        value of random variable following a negative binomial distribution
    mu : float
        mean of negative binomial distribution
    var : float
        variance of negative binomial distribution
    """
    n = mu * mu / (var - mu)
    p = mu / var
    return nbinom(n, p).cdf(x)


def cdf_fit_nbinom(quantiles: np.ndarray, cdf_values: np.ndarray, mode: Optional[str] = "ppf") -> callable:
    """
    Interpolation between pairs of CDF values (potentially estimated by means
    of quantile regression) and corresponding quantiles according to a
    negative binomial distribution as assumed PDF (i.e., discrete, non-negative
    target values).

    Parameters
    ----------
    quantiles : np.ndarray
        quantile values
    cdf_values : np.ndarray
        CDF values corresponding to quantile values
    mode : str
        decides about kind of returned callable, possible values are:

            - ``ppf``: input quantile, output CDF value (default)
            - ``dist``: fitted negative binomial (scipy function)
            - ``cdf``: input CDF value, output quantile

    Returns
    -------
    callable
        fitted negative binomial function (see mode)
    """
    mu, var = curve_fit(_nbinom_cdf_mu_var, cdf_values, quantiles, p0=[1.2, 1.4])[0]
    n = mu * mu / (var - mu)
    p = mu / var
    if mode == "ppf":
        return nbinom(n, p).ppf
    elif mode == "dist":
        return nbinom(n, p)
    elif mode == "cdf":
        return nbinom(n, p).cdf
    else:
        raise Exception("Invalid mode.")


def cdf_fit_spline(quantiles: np.ndarray, cdf_values: np.ndarray) -> callable:
    """
    Interpolation between pairs of CDF values (potentially estimated by means
    of quantile regression) and corresponding quantiles according to a
    smoothing spline (i.e., arbitrary target distribution).

    Parameters
    ----------
    quantiles : np.ndarray
        quantile values
    cdf_values : np.ndarray
        CDF values corresponding to quantile values

    Returns
    -------
    callable
        fitted spline function (input quantile, output CDF value)
    """
    spl = InterpolatedUnivariateSpline(quantiles, cdf_values, k=3, bbox=[0, 1], ext=3)
    return spl
