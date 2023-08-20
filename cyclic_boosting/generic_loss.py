from __future__ import absolute_import, division, print_function

import abc
import logging
import warnings

import numpy as np
import pandas as pd
import six
import sklearn.base
from scipy.optimize import minimize
from scipy.stats import beta

from cyclic_boosting.base import CyclicBoostingBase, gaussian_matching_by_quantiles, Feature, CBLinkPredictionsFactors
from cyclic_boosting.link import LogLinkMixin, IdentityLinkMixin, LogitLinkMixin
from cyclic_boosting.utils import continuous_quantile_from_discrete, get_X_column
from cyclic_boosting.classification import get_beta_priors

from typing import Tuple, Union

_logger = logging.getLogger(__name__)


@six.add_metaclass(abc.ABCMeta)
class CBGenericLoss(CyclicBoostingBase):
    """
    A generic loss, to be defined in the respective subclass, is minimized in
    each bin of each feature. While binning, feature cycles, smoothing, and
    iterations work in the same way as usual in Cyclic Boosting, the
    minimization itself is performed via ``scipy.optimize.minimize`` (instead
    of an analytical solution like, e.g., in ``CBPoissonRegressor``,
    ``CBNBinomRegressor``, or ``CBLocationRegressor``).
    """

    def precalc_parameters(self, feature: Feature, y: np.ndarray, pred: CBLinkPredictionsFactors) -> None:
        pass

    def calc_parameters(
        self, feature: Feature, y: np.ndarray, pred: CBLinkPredictionsFactors, prefit_data
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calling of the optimization (loss minimization) for the different bins
        of the feature at hand. In contrast to the analytical solution in most
        other Cyclic Boosting modes (e.g., ``CBPoissonRegressor``), working
        simply via bin statistics (`bincount`), the generic, numerical
        optimization here requires a dedicated loss funtion to be called for
        each observation.

        Parameters
        ----------
        feature : :class:`Feature`
            feature for which the parameters of each bin are estimated
        y : np.ndarray
            target variable, containing data with `float` type (potentially
            discrete)
        pred
            (in-sample) predictions from all other features (excluding the one
            at hand)
        prefit_data
            data returned by :meth:`~.precalc_parameters` during fit, not used
            here

        Returns
        -------
        float, float
            estimated parameters and its uncertainties
        """
        sorting = feature.lex_binned_data.argsort()
        sorted_bins = feature.lex_binned_data[sorting]
        splits_indices = np.unique(sorted_bins, return_index=True)[1][1:]

        y_pred = np.hstack((y[..., np.newaxis], self.unlink_func(pred.predict_link())[..., np.newaxis]))
        y_pred = np.hstack((y_pred, self.weights[..., np.newaxis]))
        y_pred_bins = np.split(y_pred[sorting], splits_indices)

        n_bins = len(y_pred_bins)
        parameters = np.zeros(n_bins)
        uncertainties = np.zeros(n_bins)

        for bin in range(n_bins):
            parameters[bin], uncertainties[bin] = self.optimization(
                y_pred_bins[bin][:, 0], y_pred_bins[bin][:, 1], y_pred_bins[bin][:, 2]
            )

        neutral_factor = self.unlink_func(np.array(self.neutral_factor_link))
        if n_bins + 1 == feature.n_bins:
            parameters = np.append(parameters, neutral_factor)
            uncertainties = np.append(uncertainties, 0)

        if neutral_factor != 0:
            epsilon = 1e-5
            parameters = np.where(np.abs(parameters) < epsilon, epsilon, parameters)
            parameters = np.log(parameters)
        return parameters, uncertainties

    def optimization(self, y: np.ndarray, yhat_others: np.ndarray, weights: np.ndarray) -> Tuple[float, float]:
        """
        Minimization of the costs (potentially including sample weights) for
        individual feature bins. The initial value for the parameters is set to
        the neutral value for the respective mode.

        Parameters
        ----------
        param : float
            Parameter to be estimated for the feature bin at hand.
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
            estimated parameter and its uncertainty
        """
        neutral_factor = self.unlink_func(np.array(self.neutral_factor_link))
        res = minimize(self.objective_function, neutral_factor, args=(yhat_others, y, weights))
        return res.x, self.uncertainty(y, weights)

    def objective_function(self, param: float, yhat_others: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
        """
        Calculation of the in-sample costs (potentially including sample
        weights) for individual feature bins according to a given loss
        function, to be minimized subsequently.

        Parameters
        ----------
        param : float
            Parameter to be estimated for the feature bin at hand.
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
            calcualted costs
        """
        model = self.model(param, yhat_others)
        return self.costs(model, y, weights)

    @abc.abstractmethod
    def costs(self, prediction: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
        raise NotImplementedError("implement in subclass")

    @abc.abstractmethod
    def model(self, param: float, yhat_others: np.ndarray) -> np.ndarray:
        raise NotImplementedError("implement in subclass")

    @abc.abstractmethod
    def uncertainty(self, y: np.ndarray, weights: np.ndarray) -> float:
        """
        Estimation of parameter uncertainty for a given feature bin.

        Parameters
        ----------
        y : np.ndarray
            target variable, containing data with `float` type (potentially discrete)
        weights : np.ndarray
            optional (otherwise set to 1) sample weights, containing data with `float` type

        Returns
        -------
        float
            estimated parameter uncertainty
        """
        raise NotImplementedError("implement in subclass")


class CBMultiplicativeQuantileRegressor(CBGenericLoss, sklearn.base.RegressorMixin, LogLinkMixin):
    """
    Cyclic Boosting multiplicative quantile-regression mode. A quantile loss,
    according to the desired quantile to be predicted, is minimized in each bin
    of each feature. While its general structure allows arbitrary/empirical
    target ranges/distributions, the multiplicative model of this mode requires
    non-negative target values.

    This should be used for non-negative target ranges, i.e., 0 to infinity.

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
        quantile=None,
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

    def _check_y(self, y: np.ndarray) -> None:
        check_y_multiplicative(y)

    def loss(self, prediction: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
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
        return quantile_costs(prediction, y, weights, self.quantile)

    def _init_global_scale(self, X: Union[pd.DataFrame, np.ndarray], y: np.ndarray) -> None:
        self.global_scale_link_, self.prior_pred_link_offset_ = quantile_global_scale(
            X, y, self.quantile, self.weights, self.prior_prediction_column, self.link_func
        )

    def costs(self, prediction: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
        return quantile_costs(prediction, y, weights, self.quantile)

    def model(self, param: float, yhat_others: np.ndarray) -> np.ndarray:
        return model_multiplicative(param, yhat_others)

    def uncertainty(self, y: np.ndarray, weights: np.ndarray) -> float:
        return uncertainty_gamma(y, weights)


class CBAdditiveQuantileRegressor(CBGenericLoss, sklearn.base.RegressorMixin, IdentityLinkMixin):
    """
    Cyclic Boosting additive quantile-regression mode. A quantile loss,
    according to the desired quantile to be predicted, is minimized in each bin
    of each feature.

    This should be used for unconstrained target ranges, i.e., -infinity to
    infinity.

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
        quantile=None,
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

    def _check_y(self, y: np.ndarray) -> None:
        check_y_additive(y)

    def loss(self, prediction: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
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
        return quantile_costs(prediction, y, weights, self.quantile)

    def _init_global_scale(self, X: Union[pd.DataFrame, np.ndarray], y: np.ndarray) -> None:
        self.global_scale_link_, self.prior_pred_link_offset_ = quantile_global_scale(
            X, y, self.quantile, self.weights, self.prior_prediction_column, self.link_func
        )

    def costs(self, prediction: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
        return quantile_costs(prediction, y, weights, self.quantile)

    def model(self, param: float, yhat_others: np.ndarray) -> np.ndarray:
        return model_additive(param, yhat_others)

    def uncertainty(self, y: np.ndarray, weights: np.ndarray) -> float:
        return uncertainty_gaussian(y, weights)


def quantile_costs(prediction: np.ndarray, y: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    """
    Calculation of the in-sample quantile costs (potentially including sample
    weights).

    Parameters
    ----------
    prediction : np.ndarray
        (in-sample) predictions for desired quantile, containing data with `float` type
    y : np.ndarray
        target variable, containing data with `float` type (potentially discrete)
    weights : np.ndarray
        optional (otherwise set to 1) sample weights, containing data with `float` type
    quantile : float
        quantile to be estimated

    Returns
    -------
    float
        calcualted quantile costs
    """
    if not len(y) > 0:
        raise ValueError("Loss cannot be computed on empty data")
    else:
        sum_weighted_error = np.nansum(
            ((y < prediction) * (1 - quantile) * (prediction - y) + (y >= prediction) * quantile * (y - prediction))
            * weights
        )
        return sum_weighted_error / np.nansum(weights)


def quantile_global_scale(
    X: Union[pd.DataFrame, np.ndarray],
    y: np.ndarray,
    quantile: float,
    weights: np.ndarray,
    prior_prediction_column: Union[str, int, None],
    link_func,
) -> None:
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
    if weights is None:
        raise RuntimeError("The weights have to be initialized.")

    global_scale_link_ = link_func(continuous_quantile_from_discrete(y, quantile))

    prior_pred_link_offset_ = None
    if prior_prediction_column is not None:
        prior_pred = get_X_column(X, prior_prediction_column)
        finite = np.isfinite(prior_pred)
        if not np.all(finite):
            _logger.warning(
                "Found a total number of {} non-finite values in the prior prediction column".format(np.sum(~finite))
            )

        prior_pred_mean = np.sum(prior_pred[finite] * weights[finite]) / np.sum(weights[finite])

        prior_pred_link_mean = link_func(prior_pred_mean)

        if np.isfinite(prior_pred_link_mean):
            prior_pred_link_offset_ = global_scale_link_ - prior_pred_link_mean
        else:
            warnings.warn(
                "The mean prior prediction in link-space is not finite. "
                "Therefore no indiviualization is done "
                "and no prior mean substraction is necessary."
            )
            prior_pred_link_offset_ = float(global_scale_link_)

    return global_scale_link_, prior_pred_link_offset_


def model_multiplicative(param: float, yhat_others: np.ndarray) -> np.ndarray:
    return param * yhat_others


def model_additive(param: float, yhat_others: np.ndarray) -> np.ndarray:
    return param + yhat_others


def uncertainty_gamma(y: np.ndarray, weights: np.ndarray) -> float:
    # use moment-matching of a Gamma posterior with a log-normal
    # distribution as approximation
    alpha_prior = 2
    alpha_posterior = np.sum(y) + alpha_prior
    sigma = np.sqrt(np.log(1 + alpha_posterior) - np.log(alpha_posterior))
    return sigma


def uncertainty_gaussian(y: np.ndarray, weights: np.ndarray) -> float:
    return np.sqrt(np.mean(y) / len(y))


def uncertainty_beta(y: np.ndarray, weights: np.ndarray, link_func) -> float:
    # use moment-matching of a Beta posterior with a log-normal
    # distribution as approximation
    alpha_prior, beta_prior = get_beta_priors()
    alpha_posterior = np.sum(y) + alpha_prior
    beta_posterior = np.sum(1 - y) + beta_prior
    shift = 0.4 * (alpha_posterior / (alpha_posterior + beta_posterior) - 0.5)
    perc1 = 0.75 - shift
    perc2 = 0.25 - shift
    posterior = beta(alpha_posterior, beta_posterior)
    _, sigma = gaussian_matching_by_quantiles(posterior, link_func, perc1, perc2)
    return sigma


def check_y_multiplicative(y: np.ndarray) -> None:
    """Check that y has no negative values."""
    if not (y >= 0.0).all():
        raise ValueError(
            "The target y must be positive semi-definite " "and not NAN. y[~(y>=0)] = {0}".format(y[~(y >= 0)])
        )


def check_y_additive(y: np.ndarray) -> None:
    if not np.isfinite(y).all():
        raise ValueError("The target y must be real value and not NAN.")


def check_y_classification(y: np.ndarray) -> None:
    """Check that y has only values 0. or 1."""
    if not ((y == 0.0) | (y == 1.0)).all():
        raise ValueError(
            "The target y must be either 0 or 1 "
            "and not NAN. y[(y != 0) & (y != 1)] = {0}".format(y[(y != 0) & (y != 1)])
        )


class CBMultiplicativeGenericCRegressor(CBGenericLoss, sklearn.base.RegressorMixin, LogLinkMixin):
    """
    Multiplicative regression mode allowing an arbitrary loss function to be
    minimized in each feature bin.This should be used for non-negative target
    ranges, i.e., 0 to infinity.

    Parameters
    ----------
    costs : function
        loss (to be exact, cost) function to be minimized
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
        aggregate=True,
        costs=None,
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

        self.costs = costs

    def loss(self, prediction: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
        return self.costs(prediction, y, weights)

    def _check_y(self, y: np.ndarray) -> None:
        check_y_multiplicative(y)

    def costs(self, prediction: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
        return self.costs(prediction, y, weights)

    def model(self, param: float, yhat_others: np.ndarray) -> np.ndarray:
        return model_multiplicative(param, yhat_others)

    def uncertainty(self, y: np.ndarray, weights: np.ndarray) -> float:
        return uncertainty_gamma(y, weights)


class CBAdditiveGenericCRegressor(CBGenericLoss, sklearn.base.RegressorMixin, IdentityLinkMixin):
    """
    Additive regression mode allowing an arbitrary loss function to be
    minimized in each feature bin. This should be used for unconstrained target
    ranges, i.e., -infinity to infinity.

    Parameters
    ----------
    costs : function
        loss (to be exact, cost) function to be minimized
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
        aggregate=True,
        costs=None,
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

        self.costs = costs

    def loss(self, prediction: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
        return self.costs(prediction, y, weights)

    def _check_y(self, y: np.ndarray) -> None:
        check_y_additive(y)

    def costs(self, prediction: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
        return self.costs(prediction, y, weights)

    def model(self, param: float, yhat_others: np.ndarray) -> np.ndarray:
        return model_additive(param, yhat_others)

    def uncertainty(self, y: np.ndarray, weights: np.ndarray) -> float:
        return uncertainty_gaussian(y, weights)


class CBGenericClassifier(CBGenericLoss, sklearn.base.ClassifierMixin, LogitLinkMixin):
    """
    Multiplicative (binary, i.e., target values 0 and 1) classification mode
    allowing an arbitrary loss function to be minimized in each feature bin.

    Parameters
    ----------
    costs : function
        loss (to be exact, cost) function to be minimized
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
        aggregate=True,
        costs=None,
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

        self.costs = costs

    def loss(self, prediction: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
        return self.costs(prediction, y, weights)

    def _check_y(self, y: np.ndarray) -> None:
        check_y_classification(y)

    def costs(self, prediction: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
        return self.costs(prediction, y, weights)

    def model(self, param: float, yhat_others: np.ndarray) -> np.ndarray:
        return model_multiplicative(param, yhat_others)

    def uncertainty(self, y: np.ndarray, weights: np.ndarray) -> float:
        return uncertainty_beta(y, weights, self.link_func)


__all__ = [
    "CBMultiplicativeQuantileRegressor",
    "CBAdditiveQuantileRegressor",
    "CBMultiplicativeGenericCRegressor",
    "CBAdditiveGenericCRegressor",
    "CBGenericClassifier",
]
