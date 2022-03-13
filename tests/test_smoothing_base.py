from __future__ import absolute_import, division, print_function

import unittest

import numpy as np

from cyclic_boosting import smoothing, testing_utils


class TestEstimators(unittest.TestCase):
    def test_cloning(self):
        for smoother in [
            smoothing.onedim.BinValuesSmoother(),
            smoothing.onedim.RegularizeToPriorExpectationSmoother(2, threshold=4),
            smoothing.onedim.RegularizeToOneSmoother(threshold=4),
            smoothing.onedim.UnivariateSplineSmoother(k=4, s=5),
            smoothing.onedim.OrthogonalPolynomialSmoother(),
            smoothing.onedim.SeasonalSmoother(),
            smoothing.onedim.PolynomialSmoother(k=3),
            smoothing.onedim.LSQUnivariateSpline([-1, 0, 1]),
            smoothing.onedim.IsotonicRegressor(),
            smoothing.multidim.BinValuesSmoother(),
            smoothing.multidim.RegularizeToPriorExpectationSmoother(2, threshold=4),
            smoothing.multidim.RegularizeToOneSmoother(threshold=4),
        ]:
            testing_utils.check_sklearn_style_cloning(smoother)

    def test_exceptions(self):
        smoother = smoothing.multidim.BinValuesSmoother()
        self.assertRaisesRegexp(
            ValueError,
            'Please call the method "fit" before "predict" and "set_n_bins"',
            smoother.predict,
            None,
        )
        smoother = smoothing.onedim.PolynomialSmoother(k=2)
        self.assertRaisesRegexp(
            ValueError,
            "The PolynomialSmoother has not been fitted!",
            smoother.predict,
            None,
        )


def compare_smoother(est1, est2, X, y, dim=1):
    est1.fit(X.copy(), y.copy())
    pred1 = est1.predict(X[:, :dim].copy())
    pred2 = None
    if est2 is not None:
        est2.fit(X.copy(), y.copy())
        pred2 = est2.predict(X[:, :dim].copy())
    return pred1, pred2


def get_data(onedim=True):
    if onedim:
        y = np.array([0.91, 0.92, 0.93, 0.94, 1.75, 1.80, 0.40, 0.92])
        n = len(y)
        X = np.c_[
            np.arange(n), np.ones(n), [0.05, 0.05, 0.05, 0.05, 0.05, 0.15, 0.15, 0.05]
        ]
    else:
        X = np.c_[
            np.repeat(np.arange(5), 5) * 1.0,
            np.tile(np.arange(5), 5) * 1.0,
            np.ones(25),
            np.ones(25),
        ]
        y = (X[:, 0] + 1) * (X[:, 1] + 1)
    return X, y


class TestMetaSmoother_ThresholdZero(unittest.TestCase):
    def test_meta_smoother_onedim_weighted_mean_smoother(self):
        est1 = smoothing.onedim.WeightedMeanSmoother()
        est2 = smoothing.onedim.PriorExpectationMetaSmoother(
            smoothing.onedim.WeightedMeanSmoother(), prior_expectation=1, threshold=0.0
        )
        X, y = get_data()
        pred1, pred2 = compare_smoother(est1, est2, X, y)
        np.testing.assert_allclose(pred1, pred2)

    def test_meta_smoother_onedim_orthogonal_smoother(self):
        est1 = smoothing.onedim.OrthogonalPolynomialSmoother()
        est2 = smoothing.onedim.PriorExpectationMetaSmoother(
            smoothing.onedim.OrthogonalPolynomialSmoother(),
            prior_expectation=1,
            threshold=0.0,
        )
        X, y = get_data()
        pred1, pred2 = compare_smoother(est1, est2, X, y)
        np.testing.assert_allclose(pred1, pred2)

    def test_meta_smoother_multdim_weighted_mean_smoother(self):
        X, y = get_data(onedim=False)
        est1 = smoothing.multidim.WeightedMeanSmoother()
        est2 = smoothing.multidim.PriorExpectationMetaSmoother(
            smoothing.multidim.WeightedMeanSmoother(), np.mean(y), threshold=0.0
        )
        pred1, pred2 = compare_smoother(est1, est2, X, y, dim=2)
        np.testing.assert_allclose(pred1, pred2)


class TestMetaSmoother_ThresholdHigh(unittest.TestCase):
    def test_meta_smoother_onedim_weighted_mean_smoother(self):
        est = smoothing.onedim.PriorExpectationMetaSmoother(
            smoothing.onedim.WeightedMeanSmoother(), prior_expectation=1, threshold=20.0
        )
        X, y = get_data()
        pred1, _2 = compare_smoother(est, None, X, y)
        np.testing.assert_allclose(pred1, 1)

    def test_meta_smoother_onedim_orthogonal_smoother(self):
        est = smoothing.onedim.PriorExpectationMetaSmoother(
            smoothing.onedim.OrthogonalPolynomialSmoother(),
            prior_expectation=1,
            threshold=20.0,
        )
        X, y = get_data()
        pred1, _2 = compare_smoother(est, None, X, y)
        np.testing.assert_allclose(pred1, 1)

    def test_meta_smoother_multdim_weighted_mean_smoother(self):
        X, y = get_data(onedim=False)
        est = smoothing.multidim.PriorExpectationMetaSmoother(
            smoothing.multidim.WeightedMeanSmoother(), np.mean(y), threshold=20.0
        )
        pred1, _2 = compare_smoother(est, None, X, y, dim=2)
        np.testing.assert_allclose(pred1, np.mean(y))


class TestMetaSmoother_Exceptions(unittest.TestCase):
    def test_minimum_number_of_columns(self):
        X, y = get_data(onedim=True)
        est = smoothing.multidim.PriorExpectationMetaSmoother(
            smoothing.multidim.WeightedMeanSmoother(), np.mean(y), threshold=20.0
        )
        with self.assertRaises(ValueError):
            est.fit(X, y)
