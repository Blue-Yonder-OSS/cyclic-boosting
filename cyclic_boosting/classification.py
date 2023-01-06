"""
Cyclic Boosting Classifier
"""

from __future__ import absolute_import, division, print_function

import logging

import numpy as np
import scipy.stats
import sklearn.base

from cyclic_boosting import base as cyclic_boosting_base
from cyclic_boosting.base import CyclicBoostingBase
from cyclic_boosting.link import LogitLinkMixin


def get_beta_priors():
    r"""Prior values for beta distribution. The prior distribution was chosen
    to be Beta(1.001, 1.001), which is almost a uniform distribution but has a
    probability density function that goes to 0 for math:`x=0` and :math:`x=1`.

    Returns
    -------
    float
        :math:`alpha=1.001` and :math:`beta=1.001`
    """
    alpha_prior = 1.001
    beta_prior = 1.001
    return alpha_prior, beta_prior


_logger = logging.getLogger(__name__)


def boost_weights(y, prediction):
    r"""Returns weights for bincount operations on the CBClassifier.

    The weights are assigned so that they are suitable for boosting, i.e.
    weights for well-estimated samples are low, weights for bad estimations are
    high.

    .. math::

       w_i = \begin{cases} (1 - \left\langle y
                      \right\rangle_i) & y_{\text{truth}} = 1 \\
            \left\langle y \right\rangle_i &
            \text{otherwise} \end{cases}

    """
    epsilon = 1e-12
    prediction = np.where(prediction == 0.0, epsilon, prediction)
    prediction = np.where(prediction == 1.0, 1 - epsilon, prediction)
    return np.where(y, 1 - prediction, prediction)


class CBClassifier(
    sklearn.base.ClassifierMixin, CyclicBoostingBase, LogitLinkMixin
):
    """This regressor is the cyclic boosting core algorithm for classifications

    Its interface, methods and arguments are described in
    :class:`~CyclicBoostingBase`.
    """
    def _check_y(self, y):
        """Check that y has only values 0 or 1"""
        if not ((y == 0.0) | (y == 1.0)).all():
            raise ValueError(
                "The target y must be either 0 or 1 "
                "and not NAN. y[(y != 0) & (y != 1)] = {0}".format(
                    y[(y != 0) & (y != 1)]
                )
            )

    def precalc_parameters(self, feature, y, pred):
        return None

    def calc_parameters(self, feature, y, pred, prefit_data):
        lex_binnumbers = feature.lex_binned_data
        minlength = feature.n_bins
        prediction = self.unlink_func(pred.predict_link())
        event_weights = self.weights
        boosting_weights = boost_weights(y, prediction)
        weights = event_weights * boosting_weights
        alpha_prior, beta_prior = get_beta_priors()

        wsum = np.bincount(lex_binnumbers, weights=weights, minlength=minlength)

        w2sum = np.bincount(
            lex_binnumbers, weights=weights * boosting_weights, minlength=minlength
        )

        alpha = np.bincount(lex_binnumbers, weights=weights * y, minlength=minlength)

        beta = np.bincount(
            lex_binnumbers, weights=weights * (1 - y), minlength=minlength
        )

        weight_factor = np.where(wsum == 0, 1, wsum / w2sum)
        alpha *= weight_factor
        alpha = np.where(alpha < 0, 0, alpha)
        beta *= weight_factor
        beta = np.where(beta < 0, 0, beta)
        # Beta(1,1) is the uniform distribution, Beta(1.001, 1.001) has pdf
        # zero at 0 and 1. It is thus choosen as the prior.
        alpha_posterior = alpha + alpha_prior
        beta_posterior = beta + beta_prior
        posterior = scipy.stats.beta(alpha_posterior, beta_posterior)

        # old Gaussian matching
        # beta expectancy and variance
        # beta_mu = posterior.mean()
        # uncertainties_l = posterior.std() / (beta_mu * (1 - beta_mu))
        # factors_link = self.link_func(posterior.median())

        # new Gaussian matching
        # * Choose perc1 and perc2 for gaussian_matching_by_quantiles such that
        #   for an asymmetric beta distribution, the quantiles are rather far
        #   from the unsafe boundaries 0 and 1.
        shift = 0.4 * (alpha_posterior / (alpha_posterior + beta_posterior) - 0.5)
        perc1 = 0.75 - shift
        perc2 = 0.25 - shift

        # * the actual Gaussian matching
        (
            factors_link,
            uncertainties_l,
        ) = cyclic_boosting_base.gaussian_matching_by_quantiles(
            posterior, self.link_func, perc1, perc2
        )

        return factors_link, uncertainties_l

    def predict_proba(self, X, y=None, fit_mode=0):
        probability_signal = super(CBClassifier, self).predict(
            X, y=y, fit_mode=fit_mode, actions=None
        )
        return np.c_[1 - probability_signal, probability_signal]

    def predict(self, X, y=None, fit_mode=0, actions=None):
        probability_signal = super(CBClassifier, self).predict(
            X, y=y, fit_mode=fit_mode, actions=None
        )
        return np.asarray(probability_signal > 0.5, dtype=np.float64)


__all__ = ["CBClassifier", "boost_weights", "get_beta_priors"]
