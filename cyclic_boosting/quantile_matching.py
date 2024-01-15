import numpy as np
from numpy import exp, log, sinh, arcsinh, arccosh
from scipy.optimize import curve_fit
from scipy.stats import norm, gamma, nbinom, logistic
from scipy.interpolate import InterpolatedUnivariateSpline

from typing import Optional


class J_QPD_S:
    """
    Implementation of the semi-bounded mode of Johnson Quantile-Parameterized
    Distributions (J-QPD), see https://repositories.lib.utexas.edu/bitstream/handle/2152/63037/HADLOCK-DISSERTATION-2017.pdf
    (Due to the Python keyword, the parameter lambda from this reference is named kappa below.).
    A distribution is parameterized by a symmetric-percentile triplet (SPT).

    Parameters
    ----------
    alpha : float
        lower quantile of SPT (upper is ``1 - alpha``)
    qv_low : float
        quantile function value of ``alpha``
    qv_median : float
        quantile function value of quantile 0.5
    qv_high : float
        quantile function value of quantile ``1 - alpha``
    l : float
        lower bound of semi-bounded range (default is 0)
    version: str
        options are ``normal`` (default) or ``logistic``
    """

    def __init__(
        self,
        alpha: float,
        qv_low: float,
        qv_median: float,
        qv_high: float,
        l: Optional[float] = 0,
        version: Optional[str] = "normal",
    ):
        if version == "normal":
            self.phi = norm()
        elif version == "logistic":
            self.phi = logistic()
        else:
            raise Exception("Invalid version.")

        if (qv_low > qv_median) or (qv_high < qv_median):
            raise ValueError("The SPT values need to be monotonically increasing.")

        self.l = l

        self.c = self.phi.ppf(1 - alpha)

        self.L = log(qv_low - l)
        self.H = log(qv_high - l)
        self.B = log(qv_median - l)

        if self.L + self.H - 2 * self.B > 0:
            self.n = 1
            self.theta = qv_low - l
        elif self.L + self.H - 2 * self.B < 0:
            self.n = -1
            self.theta = qv_high - l
        else:
            self.n = 0
            self.theta = qv_median - l

        self.delta = 1.0 / self.c * sinh(arccosh((self.H - self.L) / (2 * min(self.B - self.L, self.H - self.B))))

        self.kappa = 1.0 / (self.delta * self.c) * min(self.H - self.B, self.B - self.L)

    def ppf(self, x):
        return self.l + self.theta * exp(
            self.kappa * sinh(arcsinh(self.delta * self.phi.ppf(x)) + arcsinh(self.n * self.c * self.delta))
        )

    def cdf(self, x):
        return self.phi.cdf(
            1.0
            / self.delta
            * sinh(arcsinh(1.0 / self.kappa * log((x - self.l) / self.theta)) - arcsinh(self.n * self.c * self.delta))
        )


class J_QPD_B:
    """
    Implementation of the bounded mode of Johnson Quantile-Parameterized
    Distributions (J-QPD), see https://repositories.lib.utexas.edu/bitstream/handle/2152/63037/HADLOCK-DISSERTATION-2017.pdf.
    (Due to the Python keyword, the parameter lambda from this reference is named kappa below.)
    A distribution is parameterized by a symmetric-percentile triplet (SPT).

    Parameters
    ----------
    alpha : float
        lower quantile of SPT (upper is ``1 - alpha``)
    qv_low : float
        quantile function value of ``alpha``
    qv_median : float
        quantile function value of quantile 0.5
    qv_high : float
        quantile function value of quantile ``1 - alpha``
    l : float
        lower bound of supported range
    u : float
        upper bound of supported range
    version: str
        options are ``normal`` (default) or ``logistic``
    """

    def __init__(
        self,
        alpha: float,
        qv_low: float,
        qv_median: float,
        qv_high: float,
        l: float,
        u: float,
        version: Optional[str] = "normal",
    ):
        if version == "normal":
            self.phi = norm()
        elif version == "logistic":
            self.phi = logistic()
        else:
            raise Exception("Invalid version.")

        self.l = l
        self.u = u

        self.c = self.phi.ppf(1 - alpha)

        self.L = self.phi.ppf((qv_low - l) / (u - l))
        self.H = self.phi.ppf((qv_high - l) / (u - l))
        self.B = self.phi.ppf((qv_median - l) / (u - l))

        if self.L + self.H - 2 * self.B > 0:
            self.n = 1
            self.xi = self.L
        elif self.L + self.H - 2 * self.B < 0:
            self.n = -1
            self.xi = self.H
        else:
            self.n = 0
            self.xi = self.B

        self.delta = 1.0 / self.c * arccosh((self.H - self.L) / (2 * min(self.B - self.L, self.H - self.B)))

        self.kappa = (self.H - self.L) / sinh(2 * self.delta * self.c)

    def ppf(self, x):
        return self.l + (self.u - self.l) * self.phi.cdf(
            self.xi + self.kappa * sinh(self.delta * (self.phi.ppf(x) + self.n * self.c))
        )

    def cdf(self, x):
        return self.phi.cdf(
            1.0 / self.delta * arcsinh(1.0 / self.kappa * (self.phi.ppf((x - self.l) / (self.u - self.l)) - self.xi))
            - self.n * self.c
        )


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

            - ``ppf``: quantile function (default)
            - ``dist``: fitted Gaussian function (scipy function)
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

            - ``ppf``: quantile function (default)
            - ``dist``: fitted Gamma function (scipy function)
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

            - ``ppf``: quantile function (default)
            - ``dist``: fitted negative binomial function (scipy function)
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
