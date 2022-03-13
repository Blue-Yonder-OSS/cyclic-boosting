r"""Cyclic-boosting regression core algorithm

Introductions by example to the Cyclic Boosting core regressor can be
found on our howto pages

 - A general introduction to Cyclic Boosting : :ref:`factor_model_howto`
 - An introduction using Cyclic Boosting with residuum emovs
   :ref:`howto_residuum_emovs`

For more information on the family of Cyclic Boosting algorithms see
:ref:`cyclic_boosting_family` The Cyclic Boosting regressor can be found
directly in the Cyclic Boosting package, see
:class:`~cyclic_boosting.CBFixedVarianceRegressor`

Assumptions of the CBFixedVarianceRegressor
---------------------------------------------------------------------------

Cyclic Boosting assumes an underlying `gamma poisson distribution
<http://en.wikipedia.org/wiki/Negative_binomial_distribution>`_ as conditional
distribution of the target.  The gamma poisson distribution is a discrete
version of the gamma distribution.  The prior values for the gamma distribution
:math:`\alpha = 2.0` and :math:`\beta = 1.67834` are chosen such that the
median of the gamma distribution is one
:math:`\Gamma_{\text{median}}(\alpha, \beta) =1`, which is the
neutral element of multiplication (Cyclic Boosting is a
multiplicative model).  The estimate for the factors is the median of the gamma
distribution with the measured values of :math:`\alpha` and :math:`\beta`.  To
determine the uncertainties of the factors the variance is estimated from a log
normal distribution that is approximated using the first two moments of the
gamma distribution.

**Original author of these algorithms**: Michael Feindt
"""
from __future__ import absolute_import, division, print_function

import abc
import logging

import numexpr
import numpy as np
import six
import sklearn.base

from cyclic_boosting.base import CyclicBoostingBase
from cyclic_boosting.link import LogLinkMixin

from .cxx_regression import _calc_factors_from_posterior

_logger = logging.getLogger(__name__)


def _calc_factors_and_uncertainties(alpha, beta, link_func):
    # pytest: disable=no-member

    # The prior is chosen such that the median of the
    # prior gamma distribution equals 1
    alpha_prior, beta_prior = get_gamma_priors()
    alpha_posterior = alpha + alpha_prior
    beta_posterior = beta + beta_prior

    factors = _calc_factors_from_posterior(alpha_posterior, beta_posterior)
    # factor_uncertainties:
    # The variance V used here was calculated by matching the
    # first two moments of the
    # gamma posterior with a lognormal distribution.
    uncertainties = np.sqrt(link_func(1 + alpha_posterior) - link_func(alpha_posterior))

    return factors, uncertainties


def get_gamma_priors():
    "prior values for gamma distribution with median 1"
    alpha_prior = 2
    beta_prior = 1.67834
    return alpha_prior, beta_prior


@six.add_metaclass(abc.ABCMeta)
class CBBaseRegressor(CyclicBoostingBase, sklearn.base.RegressorMixin, LogLinkMixin):
    r"""This is the base regressor for all cyclic boosting regression problems.
    It implements :class:`cyclic_boosting.link.LogLinkMixin` and is usable
    for regression problems with a target range of: :math:`0 \leq y < \infty`.

    Its interface, methods and arguments are described in
    :class:`~CyclicBoostingBase`.
    """

    def _check_y(self, y):
        """Check that y has no negative values."""
        if not (y >= 0.0).all():
            raise ValueError(
                "The target y must be positive semi-definite "
                "and not NAN. y[~(y>=0)] = {0}".format(y[~(y >= 0)])
            )

    @abc.abstractmethod
    def calc_parameters(self, feature, y, pred, prefit_data):
        raise NotImplementedError("implement in subclass")

    @abc.abstractmethod
    def precalc_parameters(self, feature, y, pred):
        return None


class CBFixedVarianceRegressor(CBBaseRegressor):
    r"""This regressor minimizes the mean squared error.

    The algorithm is usable for regressions of target-values
    :math:`0 \leq y < \infty`.  Its interface, methods and
    arguments are described in :class:`~CyclicBoostingBase`.

    """

    def __init__(
        self,
        feature_groups=None,
        feature_properties=None,
        weight_column=None,
        prior_prediction_column=None,
        minimal_loss_change=1e-3,
        minimal_factor_change=1e-3,
        maximal_iterations=10,
        observers=None,
        smoother_choice=None,
        output_column=None,
        learn_rate=None,
        a=1.0,
        c=0.0,
        aggregate=True,
    ):

        CyclicBoostingBase.__init__(
            self,
            feature_groups=feature_groups,
            feature_properties=feature_properties,
            weight_column=weight_column,
            prior_prediction_column=prior_prediction_column,
            minimal_loss_change=minimal_loss_change,
            minimal_factor_change=minimal_factor_change,
            maximal_iterations=maximal_iterations,
            observers=observers,
            smoother_choice=smoother_choice,
            output_column=output_column,
            learn_rate=learn_rate,
            aggregate=aggregate,
        )
        self.a = a
        self.c = c

    def precalc_parameters(self, feature, y, pred):
        pass

    def calc_parameters(self, feature, y, pred, prefit_data):
        prediction_link = pred.predict_link()  # noqa
        weights = self.weights  # noqa
        lex_binnumbers = feature.lex_binned_data
        minlength = feature.n_bins
        prediction = self.unlink_func(prediction_link)  # noqa
        a = self.a  # noqa
        c = self.c  # noqa

        w = numexpr.evaluate("weights * y / (a + c * prediction)")
        alpha = np.bincount(lex_binnumbers, weights=w, minlength=minlength)

        w = numexpr.evaluate("weights * prediction / (a + c * prediction)")
        beta = np.bincount(lex_binnumbers, weights=w, minlength=minlength)

        factors, old_unc = _calc_factors_and_uncertainties(alpha, beta, self.link_func)
        return factors, old_unc


class CBPoissonRegressor(CBBaseRegressor):
    r"""This regressor is an poisson regressor.

    It assumes *purely* poisson distributed target-values
    :math:`0 \leq y < \infty`.

    Its interface, methods and arguments are described in
    :class:`~CyclicBoostingBase`.
    """

    # pylint: disable=no-member, invalid-name, missing-docstring

    def precalc_parameters(self, feature, y, pred):
        return np.bincount(
            feature.lex_binned_data, weights=y * self.weights, minlength=feature.n_bins
        )

    def calc_parameters(self, feature, y, pred, prefit_data):
        prediction = self.unlink_func(pred.predict_link())

        prediction_sum_of_bins = np.bincount(
            feature.lex_binned_data,
            weights=self.weights * prediction,
            minlength=feature.n_bins,
        )

        return _calc_factors_and_uncertainties(
            alpha=prefit_data, beta=prediction_sum_of_bins, link_func=self.link_func
        )


__all__ = ["get_gamma_priors", "CBPoissonRegressor", "CBFixedVarianceRegressor"]
