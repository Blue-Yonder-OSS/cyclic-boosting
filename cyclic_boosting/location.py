"""
Cyclic Boosting Regression for a location parameter target.
"""

from __future__ import absolute_import, division, print_function

import logging

import numexpr
import numpy as np
import sklearn.base

from cyclic_boosting.base import CyclicBoostingBase
from cyclic_boosting.linear import calc_parameters_intercept, precalc_variance_y
from cyclic_boosting.link import IdentityLinkMixin
from cyclic_boosting.utils import weighted_stddev

_logger = logging.getLogger(__name__)


class CBCoreLinearPoissonRegressor(
    CyclicBoostingBase, sklearn.base.RegressorMixin, IdentityLinkMixin
):

    # pylint: disable=no-member, invalid-name, missing-docstring
    def precalc_parameters(self, feature, y, pred):
        lex_binnumbers = feature.lex_binned_data
        minlength = feature.n_bins
        weights = self.weights

        global_std = weighted_stddev(y, weights)
        y_sum = np.bincount(lex_binnumbers, weights=y * weights, minlength=minlength)
        bincount = np.bincount(lex_binnumbers, weights=weights, minlength=minlength)
        return y_sum, bincount, global_std

    def _check_y(self, y):
        """Check that y has no negative values."""
        if not np.isfinite(y).all():
            raise ValueError("The target y must be real value and not NAN.")

    def _regularize_summands(self, bincount, summands, uncertainties, global_std):
        r"""Regularize the summands to the prior mean. Depenending on the
        measured statistics of the bin.

        :math:`\mu_n = \sigma_{n}^{2} \left(\frac{\mu_0}{sigma_{0}^{2}} +
            \frac{n \bar{x}}{\sigma_{x}^{2}}`
        """
        prior_mean = 0.0
        reg_mean = (
            1.0
            / (bincount / uncertainties ** 2 + 1.0 / global_std ** 2)
            * (bincount * summands / uncertainties ** 2 + prior_mean / global_std ** 2)
        )
        return reg_mean

    def calc_parameters(self, feature, y, pred, prefit_data=None):
        lex_binnumbers = feature.lex_binned_data
        minlength = feature.n_bins
        prediction = self.unlink_func(pred.predict_link())
        prediction = np.where(prediction > 0, prediction, 0)
        y_sum, bincount, global_std = prefit_data
        weights = self.weights

        variance = numexpr.evaluate("where(prediction <= 0., 1, prediction)")

        factor_numerator = np.bincount(
            lex_binnumbers,
            weights=weights * (y - prediction) / variance,
            minlength=minlength,
        )
        denominator = np.bincount(
            lex_binnumbers, weights=weights / variance, minlength=minlength
        )
        uncertainty_numerator = np.bincount(
            lex_binnumbers, weights=weights ** 2 / variance, minlength=minlength
        )

        denominator = np.where(denominator > 0, denominator, 1)
        summands = factor_numerator / denominator
        uncertainties = uncertainty_numerator / denominator
        uncertainties = np.where(uncertainties > 0, uncertainties, global_std)

        summands = self._regularize_summands(
            bincount, summands, uncertainties, global_std
        )
        return summands, uncertainties

    def predict(self, X, y=None, fit_mode=0, actions=None):
        result = super(CBCoreLinearPoissonRegressor, self).predict(
            X, y=y, fit_mode=fit_mode, actions=actions
        )
        return np.where(result > 0, result, 0)


class CBCoreLocationRegressor(
    sklearn.base.RegressorMixin, CyclicBoostingBase, IdentityLinkMixin
):
    def _check_y(self, y):
        """Check that y has no negative values."""
        if not np.isfinite(y).all():
            raise ValueError("The target y must be real value and not NAN.")

    def precalc_parameters(self, feature, y, pred):
        """Calculations that are not  dependent on intermediate predictions. If
        these are not needed, return :obj:`None` in the subclass.

        Results returned by this method will be served to
        :meth:`factors_and_uncertainties` as the ``prefit_data`` argument.

        Parameters
        ----------
        feature: :class:`~.Feature`
            class containing all features
        y: np.ndarray
            target, truth
        prediction_link: np.ndarray
            prediction in link space.
        """
        return precalc_variance_y(feature, y, self.weights)

    def calc_parameters(self, feature, y, pred, prefit_data):
        """Calculates factors and uncertainties of the bins of a feature group
        in the original space (not the link space) and transforms them to the
        link space afterwards

        The factors and uncertainties cannot be determined in link space, not
        least because target values like 0 diverge in link spaces like `log`
        or `logit`.

        Parameters
        ----------
        feature: :class:`~.Feature`
            class containing all features
        y: np.ndarray
            target, truth
        prediction_link: np.ndarray
            prediction in link space of all *other* features.
        prefit_data
            data returned by :meth:`~.precalc_parameters` during fit

        Returns
        -------
        tuple
            This method must return a tuple of ``factors`` and
            ``uncertainties`` in the **link space**.
        """
        lex_binnumbers = feature.lex_binned_data
        minlength = feature.n_bins
        variance_y = prefit_data[lex_binnumbers]
        return calc_parameters_intercept(
            lex_binnumbers, pred.predict_link(), minlength, y, variance_y, self.weights
        )

    def calibrate_to_weighted_mean(self, feature):
        if feature.missing_not_learned:
            calibrated_factors_link = (
                feature.factors_link[:-1]
                - (feature.factors_link[:-1] * feature.bin_weightsums[:-1]).sum()
                / feature.bin_weightsums[:-1].sum()
            )
            calibrated_factors_link = np.append(
                calibrated_factors_link, self.neutral_factor_link
            )
        else:
            calibrated_factors_link = (
                feature.factors_link
                - (feature.factors_link * feature.bin_weightsums).sum()
                / feature.bin_weightsums.sum()
            )
        return calibrated_factors_link


__all__ = ["CBCoreLocationRegressor", "CBCoreLinearPoissonRegressor"]
