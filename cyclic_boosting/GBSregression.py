"""
Cyclic Boosting Regression for Generalized Background Subtraction regression.
"""

from __future__ import absolute_import, division, print_function

import logging

import numpy as np
from sklearn.base import RegressorMixin

from cyclic_boosting.base import CyclicBoostingBase
from cyclic_boosting.link import IdentityLinkMixin

_logger = logging.getLogger(__name__)


class CBGBSRegressor(RegressorMixin, CyclicBoostingBase, IdentityLinkMixin):
    r"""
    Variant form of Cyclic Boosting's location regressor, that corresponds to
    the regression of the outcome of a previous statistical subtraction of two
    classes of observations from each other (e.g. groups A and B: A - B).

    For this, the target y has to be set to positive values for group A and
    negative values for group B.

    Additional Parameter
    --------------------
    regalpha: float
        A hyperparameter to steer the strength of regularization, i.e. a
        shrinkage of the regression result for A _B to 0. A value of 0
        corresponds to no regularization.
    """

    def __init__(
        self,
        feature_groups=None,
        feature_properties=None,
        weight_column=None,
        minimal_loss_change=1e-10,
        minimal_factor_change=1e-10,
        maximal_iterations=10,
        observers=None,
        smoother_choice=None,
        output_column=None,
        learn_rate=None,
        regalpha=0.0,
        aggregate=True,
    ):

        CyclicBoostingBase.__init__(
            self,
            feature_groups=feature_groups,
            feature_properties=feature_properties,
            weight_column=weight_column,
            minimal_loss_change=minimal_loss_change,
            minimal_factor_change=minimal_factor_change,
            maximal_iterations=maximal_iterations,
            observers=observers,
            smoother_choice=smoother_choice,
            output_column=output_column,
            learn_rate=learn_rate,
            aggregate=aggregate,
        )

        self.regalpha = regalpha

    def calc_parameters(self, feature, y, pred, prefit_data):
        lex_binnumbers = feature.lex_binned_data
        minlength = feature.n_bins
        prediction = pred.predict_link()

        n = (y - prediction) * self.weights
        d = self.weights * (1 + self.regalpha)

        sum_n = np.bincount(lex_binnumbers, weights=n, minlength=minlength)

        sum_d = np.bincount(lex_binnumbers, weights=d, minlength=minlength)

        sum_nd = np.bincount(lex_binnumbers, weights=n * d, minlength=minlength)

        sum_n2 = np.bincount(lex_binnumbers, weights=n * n, minlength=minlength)

        sum_d2 = np.bincount(lex_binnumbers, weights=d * d, minlength=minlength)

        sum_d += 1
        sum_d2 += 1 ** 2

        summand = sum_n / sum_d
        variance_summand = (
            sum_d ** 2 * sum_n2 - 2.0 * sum_n * sum_d * sum_nd + sum_n ** 2 * sum_d2
        ) / sum_d ** 4

        return summand, np.sqrt(variance_summand)

    def _check_y(self, y):
        pass

    def _init_global_scale(self, X, y):
        if self.weights is None:
            raise RuntimeError("The weights have to be initialized.")
        self.global_scale_link_ = (y * self.weights).sum() / self.weights.sum()

    def loss(self, prediction, y, weights):
        wvisitsum = ((y != 0).astype(int) * weights).sum()
        return (weights * (prediction - y) ** 2).sum() / wvisitsum

    def precalc_parameters(self, feature, y, pred):
        return None


__all__ = ["CBGBSRegressor"]
