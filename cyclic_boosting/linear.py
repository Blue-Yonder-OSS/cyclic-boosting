"""Cyclic boosting linear sum model.
"""
from __future__ import absolute_import, division, print_function

import logging
import warnings

import numpy as np

from cyclic_boosting.base import calc_factors_generic

_logger = logging.getLogger(__name__)


warnings.warn(
    "the cyclic_boosting.linear module is deprecated", DeprecationWarning, stacklevel=2
)


def calc_parameters_slope(
    lex_binnumbers, prediction, minlength, y, variance_y, weights, external_col
):
    """Calculates slopes and uncertainties for each bin of a feature group.

    Parameters
    ----------
    lex_binnumbers: :class:`numpy.ndarray` (float64, ndims=1)
        1-dimensional numpy array containing the bin numbers.
    prediction: :class:`numpy.ndarray` (float64, ndims=1)
        prediction of all *other* features.
    minlength: int
        number of bins for this feature including the `nan` bin
    y: :class:`numpy.ndarray` (float64, ndims=1)
        target, truth
    variance_y: :class:`numpy.ndarray` (float64, ndims=1)
       Variance estimate for each target value
    weights: :class:`numpy.ndarray` (float64, ndims=1)
       sample weights
    external_col: :class:`numpy.ndarray` (float64, ndims=1)
       external column array

    Returns
    -------
    tuple
        ``slopes`` and ``uncertainties``.
    """
    w = weights * external_col ** 2 / variance_y
    w_x = weights * external_col * (y - prediction) / variance_y
    w_x2 = weights * (y - prediction) ** 2 / variance_y
    x0 = 0.0
    w0 = 1e-2
    return calc_factors_generic(
        lex_binnumbers, w_x, w, w_x2, weights, minlength, x0, w0
    )


def precalc_variance_y(feature, y, weights, n_prior=1):
    """Calculate expected value of posterior of variance parameter for a gaussian
    with a gamma distributed prior Gamma(a_0, b_0) and known mean for each bin.

    Reference: Bishop: Pattern Recognition and Machine Learning, page 100

    The prior variance b_0 / a_0 is set to the global variance of y.

    Parameters
    ----------
    lex_binnumbers: :class:`numpy.ndarray` (float64, ndims=1)
        1-dimensional numpy array containing the bin numbers.
    y: :class:`numpy.ndarray` (float64, ndims=1)
        target, truth
    weights: :class:`numpy.ndarray` (float64, ndims=1)
       sample weights
    minlength: int
        number of bins for this feature including the `nan` bin
    n_prior: 'effective' prior observations,
        see bishop page 101 for a discussion

    Returns
    -------
    ndarray
        The estimated variance of y for each bin
    """
    lex_binnumbers = feature.lex_binned_data
    minlength = feature.n_bins

    weighted_mean_y = np.sum(y * weights) / np.sum(weights)
    variance_prior = np.sum(
        (y - weighted_mean_y) * (y - weighted_mean_y) * weights
    ) / np.sum(weights)
    if variance_prior <= 1e-9:  # No variation in y; happens only in tests
        variance_prior = 1.0

    sum_y = np.bincount(lex_binnumbers, weights=weights * y, minlength=minlength)
    sum_weights = np.bincount(lex_binnumbers, weights=weights, minlength=minlength)
    mean_y = sum_y[lex_binnumbers] / sum_weights[lex_binnumbers]
    weighted_squared_residual_sum = np.bincount(
        lex_binnumbers, weights=weights * (y - mean_y) ** 2, minlength=minlength
    )

    a_0 = 0.5 * n_prior
    b_0 = a_0 * variance_prior
    a = a_0 + 0.5 * sum_weights
    b = b_0 + 0.5 * weighted_squared_residual_sum
    return b / a


def calc_parameters_intercept(
    lex_binnumbers, prediction, minlength, y, variance_y, weights
):
    """Calculates intercepts and uncertainties for each bin of a feature group.

    Parameters
    ----------
    lex_binnumbers: :class:`numpy.ndarray` (float64, ndims=1)
        1-dimensional numpy array containing the bin numbers.
    prediction: :class:`numpy.ndarray` (float64, ndims=1)
        prediction of all *other* features.
    minlength: int
        number of bins for this feature including the `nan` bin
    y: :class:`numpy.ndarray` (float64, ndims=1)
        target, truth
    variance_y: :class:`numpy.ndarray` (float64, ndims=1)
       Variance estimate for each target value
    weights: :class:`numpy.ndarray` (float64, ndims=1)
       sample weights
    external_col: :class:`numpy.ndarray` (float64, ndims=1)
       external column array

    Returns
    -------
    tuple
        ``intercepts`` and ``uncertainties``.
    """
    w = weights / variance_y
    w_x = w * (y - prediction)
    w_x2 = w * (y - prediction) ** 2
    x0 = 0
    w0 = 1e-2
    return calc_factors_generic(
        lex_binnumbers, w_x, w, w_x2, weights, minlength, x0, w0
    )


__all__ = ["calc_parameters_slope", "calc_parameters_intercept"]
