from __future__ import absolute_import, division, print_function

import abc
import logging
import warnings

import numexpr
import numpy as np
import six
import sklearn.base
import scipy.special
from scipy.optimize import minimize

from cyclic_boosting.base import CyclicBoostingBase
from cyclic_boosting.features import Feature
from cyclic_boosting.link import LogLinkMixin
from cyclic_boosting.utils import continuous_quantile_from_discrete, get_X_column

from typing import Tuple

_logger = logging.getLogger(__name__)


def _calc_factors_from_posterior(alpha_posterior: np.ndarray, beta_posterior: np.ndarray) -> np.ndarray:
    # The posterior distribution of f_j (factor in bin j)
    # follows a Gamma distribution. We want to use the median as estimate
    # because it is more stable against the log transformation. But calculating
    # the median is much more expensive for large values of alpha_posterior and
    # beta_posterior. Therefore we omit calculating the median and take the
    # mean instead (since mean -> median for large values of alpha_posterior
    # and beta_posterior).

    noncritical_posterior = (alpha_posterior <= 1e12) & (beta_posterior <= 1e12)
    # Median of the gamma distribution
    posterior_gamma = (
        scipy.special.gammaincinv(alpha_posterior[noncritical_posterior], 0.5) / beta_posterior[noncritical_posterior]
    )

    factors = alpha_posterior / beta_posterior
    factors[noncritical_posterior] = posterior_gamma
    return np.log(factors)


def _calc_factors_and_uncertainties(alpha: np.ndarray, beta: np.ndarray, link_func: np.ndarray) -> Tuple[np.ndarray]:
    alpha_prior, beta_prior = get_gamma_priors()
    alpha_posterior = alpha + alpha_prior
    beta_posterior = beta + beta_prior

    factors = _calc_factors_from_posterior(alpha_posterior, beta_posterior)
    # factor_uncertainties:
    # The variance used here was calculated by matching the first two moments
    # of the Gamma posterior with a log-normal distribution.
    uncertainties = np.sqrt(link_func(1 + alpha_posterior) - link_func(alpha_posterior))

    return factors, uncertainties


def get_gamma_priors() -> Tuple[int, float]:
    """prior values for Gamma distribution with median 1"""
    alpha_prior = 2
    beta_prior = 1.67834
    return alpha_prior, beta_prior


@six.add_metaclass(abc.ABCMeta)
class CBBaseRegressor(CyclicBoostingBase, sklearn.base.RegressorMixin, LogLinkMixin):
    r"""This is the base regressor for all Cyclic Boosting regression problems.
    It implements :class:`cyclic_boosting.link.LogLinkMixin` and is usable
    for regression problems with a target range of: :math:`0 \leq y < \infty`.
    """

    def _check_y(self, y: np.ndarray) -> None:
        """Check that y has no negative values."""
        if not (y >= 0.0).all():
            raise ValueError(
                "The target y must be positive semi-definite " "and not NAN. y[~(y>=0)] = {0}".format(y[~(y >= 0)])
            )

    @abc.abstractmethod
    def calc_parameters(self, feature: Feature, y: np.ndarray, pred, prefit_data):
        raise NotImplementedError("implement in subclass")

    @abc.abstractmethod
    def precalc_parameters(self, feature: Feature, y: np.ndarray, pred) -> None:
        return None


class CBNBinomRegressor(CBBaseRegressor):
    r"""This regressor minimizes the mean squared error. It is usable for
    regressions of target-values :math:`0 \leq y < \infty`.

    This Cyclic Boosting mode assumes an underlying negative binomial (or
    Gamma-Poisson) distribution as conditional distribution of the target. The
    prior values for the Gamma distribution :math:`\alpha = 2.0` and
    :math:`\beta = 1.67834` are chosen such that its median is
    :math:`\Gamma_{\text{median}}(\alpha, \beta) =1`, which is the neutral
    element of multiplication (Cyclic Boosting in this mode is a multiplicative
    model). The estimate for each factor is the median of the Gamma
    distribution with the measured values of :math:`\alpha` and :math:`\beta`.
    To determine the uncertainties of the factors, the variance is estimated
    from a log-normal distribution that is approximated using the first two
    moments of the Gamma distribution.

    In the default case of parameter settings :math:`a = 1.0` and
    :math:`c = 0.0`, this regressor corresponds to the special case of a
    Poisson regressor, as implemented in :class:`~.CBPoissonRegressor`.
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

    def precalc_parameters(self, feature: Feature, y: np.ndarray, pred):
        pass

    def calc_parameters(self, feature: Feature, y: np.ndarray, pred, prefit_data) -> Tuple[np.ndarray]:
        prediction_link = pred.predict_link()
        weights = self.weights  # noqa: F841
        lex_binnumbers = feature.lex_binned_data
        minlength = feature.n_bins
        prediction = self.unlink_func(prediction_link)  # noqa: F841
        a = self.a  # noqa: F841
        c = self.c  # noqa: F841

        w = numexpr.evaluate("weights * y / (a + c * prediction)")
        alpha = np.bincount(lex_binnumbers, weights=w, minlength=minlength)

        w = numexpr.evaluate("weights * prediction / (a + c * prediction)")
        beta = np.bincount(lex_binnumbers, weights=w, minlength=minlength)
        link_func = self.link_func

        return _calc_factors_and_uncertainties(alpha, beta, link_func)


class CBPoissonRegressor(CBBaseRegressor):
    r"""This regressor minimizes the mean squared error. It is usable for
    regressions of target-values :math:`0 \leq y < \infty`.

    As Poisson regressor, it is a special case of the more general negative
    binomial regressor :class:`~.CBNBinomRegressor`, assuming *purely*
    Poisson-distributed target values.
    """

    def precalc_parameters(self, feature: Feature, y: np.ndarray, pred):
        return np.bincount(feature.lex_binned_data, weights=y * self.weights, minlength=feature.n_bins)

    def calc_parameters(self, feature: Feature, y: np.ndarray, pred, prefit_data):
        prediction = self.unlink_func(pred.predict_link())

        prediction_sum_of_bins = np.bincount(
            feature.lex_binned_data,
            weights=self.weights * prediction,
            minlength=feature.n_bins,
        )

        return _calc_factors_and_uncertainties(alpha=prefit_data, beta=prediction_sum_of_bins, link_func=self.link_func)


class CBMultiplicativeQuantileRegressor(CBBaseRegressor):
    """
    Cyclic Boosting multiplicative quantile-regression mode. While its general
    structure allows arbitrary/empirical target ranges/distributions, the
    multiplicative model of this mode requires non-negative target values.

    A quantile loss, according to the desired quantile to be predicted, is
    minimized in each bin of each feature. While binning, feature cycles,
    smoothing, and iterations work in the same way as usual in Cyclic Boosting,
    the minimization itself is performed via ``scipy.optimize.minimize``
    (instead of an analytical solution like, e.g., in ``CBPoissonRegressor``,
    ``CBNBinomRegressor``, or ``CBLocationRegressor``).

    Parameters
    ----------
    quantile : float
        quantile to be estimated
    See :class:`cyclic_boosting.base` for all other parameters.
    """

    def __init__(
        self,
        feature_groups=None,
        feature_properties=None,
        weight_column=None,
        prior_prediction_column=None,
        minimal_loss_change=1e-10,
        minimal_factor_change=1e-10,
        maximal_iterations=10,
        observers=None,
        smoother_choice=None,
        output_column=None,
        learn_rate=None,
        quantile=0.5,
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

        self.quantile = quantile

    def precalc_parameters(self, feature, y, pred):
        pass

    def loss(self, prediction, y, weights):
        """
        Calculation of the in-sample quantile loss, or to be exact costs,
        (potentially including sample weights) after full feature cycles, i.e.,
        iterations, to be used as stopping criteria.

        Parameters
        ----------
        prediction : np.ndarray
            (in-sample) predictions for desired quantile, containing data with `float` type
        y : np.ndarray
            target variable, containing data with `float` type (potentially discrete)
        weights : np.ndarray
            optional (otherwise set to 1) sample weights, containing data with `float` type

        Returns
        -------
        float
            calcualted quantile costs
        """
        if not len(y) > 0:
            raise ValueError("Loss cannot be computed on empty data")
        else:
            sum_weighted_error = np.nansum(
                (
                    (y < prediction) * (1 - self.quantile) * (prediction - y)
                    + (y >= prediction) * self.quantile * (y - prediction)
                )
                * weights
            )
            return sum_weighted_error / np.nansum(weights)

    def _init_global_scale(self, X, y):
        """
        Calculation of the global scale for quantile regression, corresponding
        to the (continuous approximation of the) respective quantile of the
        target values used in the training.

        The exact value of the global scale is not critical for the model
        accuracy (as the model has enough parameters to compensate), but a
        value not representating a good overall average leads to factors with
        averages unequal to 1 for each feature (making interpretation more
        difficult).
        """
        if self.weights is None:
            raise RuntimeError("The weights have to be initialized.")

        self.global_scale_link_ = self.link_func(continuous_quantile_from_discrete(y, self.quantile))

        if self.prior_prediction_column is not None:
            prior_pred = get_X_column(X, self.prior_prediction_column)
            finite = np.isfinite(prior_pred)
            if not np.all(finite):
                _logger.warning(
                    "Found a total number of {} non-finite values in the prior prediction column".format(
                        np.sum(~finite)
                    )
                )

            prior_pred_mean = np.sum(prior_pred[finite] * self.weights[finite]) / np.sum(self.weights[finite])

            prior_pred_link_mean = self.link_func(prior_pred_mean)

            if np.isfinite(prior_pred_link_mean):
                self.prior_pred_link_offset_ = self.global_scale_link_ - prior_pred_link_mean
            else:
                warnings.warn(
                    "The mean prior prediction in link-space is not finite. "
                    "Therefore no indiviualization is done "
                    "and no prior mean substraction is necessary."
                )
                self.prior_pred_link_offset_ = float(self.global_scale_link_)

    def quantile_costs(self, param, yhat_others, y, weights):
        """
        Calculation of the in-sample quantile costs (potentially including
        sample weights) for individual feature bins according to a quantile
        loss function, to be minimized subsequently.

        Parameters
        ----------
        param : float
            Factor to be estimated for the feature bin at hand.
        yhat_others : np.ndarray
            (in-sample) predictions of all other features (excluding the one at
            hand) for the bin at hand, containing data with `float` type
        y : np.ndarray
            target variable, containing data with `float` type (potentially discrete)
        weights : np.ndarray
            optional (otherwise set to 1) sample weights, containing data with `float` type

        Returns
        -------
        float
            calcualted quantile costs
        """
        quantile_loss = (y < (param * yhat_others)) * (1 - self.quantile) * (param * yhat_others - y) + (
            y >= (param * yhat_others)
        ) * self.quantile * (y - param * yhat_others)
        sum_weighted_error = np.nansum(quantile_loss * weights)
        return sum_weighted_error / np.nansum(weights)

    def optimization(self, y, yhat_others, weights):
        """
        Minimization of the quantile costs (potentially including sample
        weights) for individual feature bins. The initial value for the factors
        is set to 1 (neutral value for multiplicative model).

        Parameters
        ----------
        param : float
            Factor to be estimated for the feature bin at hand.
        yhat_others : np.ndarray
            (in-sample) predictions from all other features (excluding the one
            at hand) for the bin at hand, containing data with `float` type
        y : np.ndarray
            target variable, containing data with `float` type (potentially discrete).
        weights : np.ndarray
            optional (otherwise set to 1) sample weights, containing data with `float` type

        Returns
        -------
        float, float
            estimated parameter (factor) and its uncertainty
        """
        res = minimize(self.quantile_costs, 1, args=(yhat_others, y, weights))
        # use moment-matching of a Gamma posterior with a log-normal
        # distribution as approximation
        uncertainty = np.sqrt(np.log(1 + 2 + np.sum(y)) - np.log(2 + np.sum(y)))
        return res.x, uncertainty

    def calc_parameters(self, feature, y, pred, prefit_data):
        """
        Calling of the optimization (quantile loss minimization) for the
        different bins of the feature at hand. In contrast to the analytical
        solution in most other Cyclic Boosting modes (e.g.,
        ``CBPoissonRegressor``), working simply via bin statistics
        (`bincount`), the optimization here requires a dedicated loss funtion
        to be called for each observation.

        Parameters
        ----------
        feature : :class:`Feature`
            feature for which the parameters of each bin are estimated
        y : np.ndarray
            target variable, containing data with `float` type (potentially
            discrete)
        pred : np.ndarray
            (in-sample) predictions from all other features (excluding the one
            at hand), containing data with `float` type
        prefit_data
            data returned by :meth:`~.precalc_parameters` during fit, not used
            here

        Returns
        -------
        float, float
            estimated parameter (factor) and its uncertainty
        """
        sorting = feature.lex_binned_data.argsort()
        sorted_bins = feature.lex_binned_data[sorting]
        splits_indices = np.unique(sorted_bins, return_index=True)[1][1:]

        y_pred = np.hstack((y[..., np.newaxis], self.unlink_func(pred.predict_link())[..., np.newaxis]))
        y_pred = np.hstack((y_pred, self.weights[..., np.newaxis]))
        y_pred_bins = np.split(y_pred[sorting], splits_indices)

        n_bins = len(y_pred_bins)
        factors = np.zeros(n_bins)
        uncertainties = np.zeros(n_bins)

        for bin in range(n_bins):
            factors[bin], uncertainties[bin] = self.optimization(
                y_pred_bins[bin][:, 0], y_pred_bins[bin][:, 1], y_pred_bins[bin][:, 2]
            )

        if n_bins + 1 == feature.n_bins:
            factors = np.append(factors, 1)
            uncertainties = np.append(uncertainties, 0)

        epsilon = 1e-5
        factors = np.where(factors < epsilon, epsilon, factors)
        return np.log(factors), uncertainties


__all__ = ["get_gamma_priors", "CBPoissonRegressor", "CBNBinomRegressor", "CBMultiplicativeQuantileRegressor"]
