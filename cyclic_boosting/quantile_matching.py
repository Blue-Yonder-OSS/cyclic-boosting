import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm, gamma, nbinom
from scipy.interpolate import InterpolatedUnivariateSpline

from typing import Optional


def quantile_fit_gaussian(quantiles: np.ndarray, quantile_values: np.ndarray, mode: Optional[str] = "ppf") -> callable:
    """
    Interpolation of a quantile function (with quantiles estimated, e.g., by
    means of quantile regression) according to a Gaussian distribution as
    assumed PDF.

    Parameters
    ----------
    quantiles : np.ndarray
        quantiles (x values of quantile function)
    quantile_values : np.ndarray
        quantile values (y values of quantile function)
    mode : str
        decides about kind of returned callable, possible values are:

            - ``ppf``: quantile (default)
            - ``dist``: fitted negative binomial (scipy function)
            - ``cdf``: CDF function

    Returns
    -------
    callable
        fitted Gaussian function (see mode)
    """

    def f(x, mu, sigma):
        return norm(loc=mu, scale=sigma).ppf(x)

    mu, sigma = curve_fit(f, quantiles, quantile_values)[0]
    if mode == "ppf":
        return norm(mu, sigma).ppf
    elif mode == "dist":
        return norm(mu, sigma)
    elif mode == "cdf":
        return norm(mu, sigma).cdf
    else:
        raise Exception("Invalid mode.")


def quantile_fit_gamma(quantiles: np.ndarray, quantile_values: np.ndarray, mode: Optional[str] = "ppf") -> callable:
    """
    Interpolation of a quantile function (with quantiles estimated, e.g., by
    means of quantile regression) according to a Gamma distribution as assumed
    PDF (i.e., continuous, non-negative target values).

    Parameters
    ----------
    quantiles : np.ndarray
        quantiles (x values of quantile function)
    quantile_values : np.ndarray
        quantile values (y values of quantile function)
    mode : str
        decides about kind of returned callable, possible values are:

            - ``ppf``: quantile (default)
            - ``dist``: fitted negative binomial (scipy function)
            - ``cdf``: CDF function

    Returns
    -------
    callable
        fitted Gamma function (see mode)
    """

    def f(x, alpha, beta):
        return gamma(alpha, scale=1 / beta).ppf(x)

    alpha, beta = curve_fit(f, quantiles, quantile_values, p0=[2.0, 0.9])[0]
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
    variance, and subsequent call of its cumulative distribution function.

    Parameters
    ----------
    x : float
        value of random variable following negative binomial distribution
    mu : float
        mean of negative binomial distribution
    var : float
        variance of negative binomial distribution

    Returns
    -------
    callable
        negative binomial cumulative distribution function
    """
    n = mu * mu / (var - mu)
    p = mu / var
    return nbinom(n, p).cdf(x)


def quantile_fit_nbinom(quantiles: np.ndarray, quantile_values: np.ndarray, mode: Optional[str] = "ppf") -> callable:
    """
    Interpolation of a quantile function (with quantiles estimated, e.g., by
    means of quantile regression) according to a negative binomial distribution
    as assumed PDF (i.e., discrete, non-negative target values).

    Parameters
    ----------
    quantiles : np.ndarray
        quantiles (x values of quantile function)
    quantile_values : np.ndarray
        quantile values (y values of quantile function)
    mode : str
        decides about kind of returned callable, possible values are:

            - ``ppf``: quantile (default)
            - ``dist``: fitted negative binomial (scipy function)
            - ``cdf``: CDF function

    Returns
    -------
    callable
        fitted negative binomial function (see mode)
    """
    mu, var = curve_fit(_nbinom_cdf_mu_var, quantile_values, quantiles, p0=[2.2, 2.4])[0]
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


def quantile_fit_spline(quantiles: np.ndarray, quantile_values: np.ndarray) -> callable:
    """
    Interpolation of a quantile function (with quantiles estimated, e.g., by
    means of quantile regression) according to a smoothing spline (i.e.,
    arbitrary target distribution).

    Parameters
    ----------
    quantiles : np.ndarray
        quantiles (x values of quantile function)
    quantile_values : np.ndarray
        quantile values (y values of quantile function)

    Returns
    -------
    callable
        spline fitted to quantile function
    """
    spl = InterpolatedUnivariateSpline(quantiles, quantile_values, k=3, bbox=[0, 1], ext=3)
    return spl
