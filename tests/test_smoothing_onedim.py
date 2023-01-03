import warnings

import numpy as np
import pandas as pd
import pytest

from cyclic_boosting import smoothing


def regularize1d(factors_dim, errors_dim):
    v = factors_dim
    varv = errors_dim ** 2.0
    wv = 1.0 / varv
    #  regularize each component towards inclusive mean
    #  (include inclusive distribution as a prior measurement)
    muv = np.sum(wv * v) / np.sum(wv)
    wvincl = 1.0 / (np.sum(wv * (v - muv) ** 2) / np.sum(wv))
    v_smoothed = (wv * v + wvincl * muv) / (wv + wvincl)
    return v_smoothed


def _create_seasonal_test_data(
    c_const=0.0,
    c_sin_1x=0.5,
    c_cos_1x=0.5,
    c_sin_2x=0.0,
    c_cos_2x=0.0,
    c_sin_3x=0.0,
    c_cos_3x=0.0,
    c_sin_4x=0.0,
    c_cos_4x=0.0,
    n=100,
):
    np.random.seed(123)
    xraw = np.arange(n)
    x = 2 * np.pi * xraw * (1.0 / n)
    y_truth = (
        c_const
        + c_sin_1x * np.sin(x)
        + c_cos_1x * np.cos(x)
        + c_sin_2x * np.sin(2 * x)
        + c_cos_2x * np.cos(2 * x)
        + c_sin_3x * np.sin(3 * x)
        + c_cos_3x * np.cos(3 * x)
        + c_sin_4x * np.sin(4 * x)
        + c_cos_4x * np.cos(4 * x)
    )
    y = y_truth * np.exp(np.random.rand(n) * 0.05)
    err = np.ones(n) * np.mean(np.abs(y_truth)) * 0.05

    X = np.c_[xraw, np.ones(n), err]
    return X, y, y_truth


# SeasonalSmoother

def test_mc_first_order():
    X, y, ytruth = _create_seasonal_test_data()
    smoother = smoothing.onedim.SeasonalSmoother(order=1, offset_tozero=False)
    smoother.fit(X.copy(), y)
    ypred = smoother.predict(X)
    assert np.mean(np.abs(ypred - y)) < 0.1
    assert np.mean(np.abs(ytruth - ypred)) < 0.1


def test_mc_second_order():
    X, y, ytruth = _create_seasonal_test_data(c_sin_2x=3.0)
    smoother = smoothing.onedim.SeasonalSmoother(order=2, offset_tozero=False)
    smoother.fit(X.copy(), y)
    ypred = smoother.predict(X)
    assert np.mean(np.abs(ypred - y)) < 0.1
    assert np.mean(np.abs(ytruth - ypred)) < 0.1


def test_mc_third_order():
    X, y, ytruth = _create_seasonal_test_data(c_cos_2x=0.05, c_sin_3x=0.1)
    smoother = smoothing.onedim.SeasonalSmoother(order=3, offset_tozero=False)
    smoother.fit(X.copy(), y)
    ypred = smoother.predict(X)
    assert np.mean(np.abs(ypred - y)) < 0.1
    assert np.mean(np.abs(ytruth - ypred)) < 0.1


def test_force_offset_to_zero():
    X, y, ytruth = _create_seasonal_test_data(c_const=5.0)
    smoother = smoothing.onedim.SeasonalSmoother(order=1, offset_tozero=True)
    smoother.fit(X.copy(), y)
    ypred = smoother.predict(X)
    assert np.abs(np.mean(np.abs(ypred - ytruth)) - 5.0) < 1e-5


def test_exceptions():
    with np.testing.assert_raises(ValueError):
        smoothing.onedim.SeasonalSmoother(order=5)

    with np.testing.assert_raises(ValueError):
        est = smoothing.onedim.SeasonalSmoother(order=1)
        est.predict(np.c_[np.arange(10)])


def test_warning_custom_function_and_order():
    my_cff = lambda x: x

    with warnings.catch_warnings(record=True) as w:
        smoothing.onedim.SeasonalSmoother(order=3, custom_fit_function=my_cff)
        assert len(w) == 1
        expected_message = "You specified a `custom_fit_function` thus the `order` parameter will be ignored!"
        assert expected_message in str(w[0])


def test_custom_choice_function():
    def cust_func(x, a, b, c, d, e):
        return (
            a
            + b * np.sin(x)
            + c * np.cos(x)
            + d * np.sin(4 * x)
            + e * np.cos(4 * x)
        )

    X, y, ytruth = _create_seasonal_test_data(
        c_const=1.0, c_sin_1x=1.0, c_cos_1x=0.0, c_sin_4x=0.2, c_cos_4x=-0.1
    )

    smoother = smoothing.onedim.SeasonalSmoother(
        offset_tozero=False, custom_fit_function=cust_func
    )
    smoother.fit(X.copy(), y)
    ypred = smoother.predict(X)
    assert np.mean(np.abs(ypred - y)) < 0.1
    assert np.mean(np.abs(ytruth - ypred)) < 0.1


def test_small_sample_size():
    X, y, ytruth = _create_seasonal_test_data()
    fallback_value = 1
    smoother = smoothing.onedim.SeasonalSmoother(fallback_value=fallback_value)
    with pytest.warns(UserWarning):
        smoother.fit(X[:2].copy(), y[:2])
    ypred = smoother.predict(X)
    assert np.all(ypred == fallback_value)


# SeasonalLassoSmoother

def test_fit_predict():
    X, y, ytruth = _create_seasonal_test_data(n=365)
    smoother = smoothing.onedim.SeasonalLassoSmoother(min_days=300)
    smoother.fit(X, y)
    ypred = smoother.predict(X)
    assert np.all(np.abs(ypred - y) < 0.1)
    assert np.all(np.abs(ytruth - ypred) < 0.1)


def test_fallback_predict():
    X, y, ytruth = _create_seasonal_test_data(n=365)
    X[0:200, 1] = np.zeros(200)
    fallback_value = 0
    smoother = smoothing.onedim.SeasonalLassoSmoother(min_days=300)
    smoother.fit(X, y)
    ypred = smoother.predict(X)
    assert np.all(ypred == fallback_value)


def test_raises_on_missing_fit():
    X, y, ytruth = _create_seasonal_test_data(n=20)
    smoother = smoothing.onedim.SeasonalLassoSmoother(min_days=300)
    with pytest.raises(ValueError):
        smoother.predict(X)


def check_orthogonal_poly_close(feature_values, y_truth):
    # The prediction should be close to the ``truth_values``.
    np.random.seed(10)

    sigma = 1.0 / 10.0
    y_noise = np.random.standard_normal(len(y_truth)) * sigma
    y_observed = y_truth + y_noise

    y_errors = np.ones_like(y_observed) * sigma

    smooth = smoothing.onedim.OrthogonalPolynomialSmoother()

    X = np.c_[feature_values, np.ones_like(y_observed), y_errors]

    smooth.fit(X, y_observed)
    np.testing.assert_allclose(smooth.predict(X), y_truth, atol=0.025)


@pytest.mark.parametrize(
    "feature_values", [np.arange(200), np.arange(0.8, 200), np.arange(0.2, 200)]
)
def test_orthogonal_polynomial_smoother_smooths_sine_curve(feature_values):
    check_orthogonal_poly_close(
        feature_values, np.sin(2 * np.pi * feature_values / len(feature_values))
    )


@pytest.mark.parametrize(
    "feature_values", [np.arange(200), np.arange(0.8, 200), np.arange(0.2, 200)]
)
def test_orthogonal_poly_close_simple_multiple_of_feature_values(feature_values):
    check_orthogonal_poly_close(feature_values, 100 * feature_values)


def test__orthogonal_polynomial_smoother_raise_not_fitted_exception():
    n = 10
    X_dummy = np.c_[np.arange(n), np.ones(n), np.zeros(n)]
    smoother = smoothing.onedim.OrthogonalPolynomialSmoother()
    with np.testing.assert_raises(ValueError):
        smoother.predict(X_dummy)


def generate_data_from_sinus(n_bins, n_modes):
    np.random.seed(0)
    x = np.random.randn(10000)
    fractional = x / (x.max() - x.min())
    y = (
        np.sin(fractional * 2 * np.pi * n_modes)
        + 3 * np.random.randn(len(x)) * fractional
    )
    df = pd.DataFrame({"x": x, "y": y})
    df["bins"] = pd.qcut(x, n_bins).codes

    groups = df.groupby("bins")["y"].agg(["mean", "std"])
    groups["x"] = groups.index

    X_for_smoother = np.c_[
        groups["x"].values, np.zeros_like(groups["x"].values), groups["std"].values
    ]
    y_grouped = groups["mean"].values
    return X_for_smoother, y_grouped


@pytest.mark.parametrize("n_bins", [1, 25, 100, 200])
@pytest.mark.parametrize("n_modes", [1, 3, 9])
def test_orthogonal_polynomial_smoother(n_bins, n_modes):
    X_for_smoother, y_grouped = generate_data_from_sinus(n_bins, n_modes)
    x = X_for_smoother[:, 0]

    est1 = smoothing.onedim.OrthogonalPolynomialSmoother().fit(
        X_for_smoother, y_grouped
    )
    est1.predict(np.c_[x])


def test_polynomial_smoother():
    np.random.seed(10)
    n = 1000
    x = np.linspace(-5, 5, n)
    coeffs = np.array([2, 15, 3])
    y = coeffs[2] + coeffs[1] * x + coeffs[0] * x ** 2
    truth = y.copy()
    y += np.random.standard_normal(size=len(y)) / 10.0
    X = np.c_[x, np.ones(n), np.ones(n)]
    smoother = smoothing.onedim.PolynomialSmoother(k=2)
    smoother.fit(X, y)
    np.testing.assert_allclose(smoother.coefficients_, coeffs, atol=0.01, rtol=0.01)
    pred = smoother.predict(X)
    np.testing.assert_allclose(truth, pred, atol=0.1, rtol=0.1)


def test_linear_smoother_fit():
    np.random.seed(10)
    n = 100
    x = np.linspace(-10, 10, n)
    y = -3 * x + 2
    y += np.random.randn(n) / 10.0

    X = np.c_[x, np.ones(n), np.ones(n)]
    smoother = smoothing.onedim.LinearSmoother(fallback_when_negative_slope=False)
    smoother.fit(X, y)
    np.testing.assert_allclose(smoother.coefficients_, [-3, 2], atol=0.1, rtol=0.1)

    # fallback when negative slope
    smoother = smoothing.onedim.LinearSmoother(fallback_when_negative_slope=True)
    smoother.fit(X, y)
    assert len(smoother.coefficients_) == 1
    np.testing.assert_allclose(smoother.coefficients_[0], np.mean(y))


def test_crosstest_weighted_mean_smoother():
    y_for_reg = np.array([0.91, 0.92, 0.93, 0.94, 1.75, 1.80, 0.40, 0.92])
    n = len(y_for_reg)
    X_for_reg = np.c_[
        np.arange(n), np.ones(n), [0.05, 0.05, 0.05, 0.05, 0.05, 0.15, 0.15, 0.05]
    ]
    smoother = smoothing.onedim.WeightedMeanSmoother()
    smoother.fit(X_for_reg, y_for_reg)
    erg = smoother.smoothed_y_
    ref = regularize1d(y_for_reg, X_for_reg[:, 2])
    np.testing.assert_allclose(erg, ref)


def test_weighted_mean_smoother_raise_not_fitted_exception():
    n = 10
    X_dummy = np.c_[np.arange(n), np.ones(n), np.zeros(n)]
    smoother = smoothing.onedim.WeightedMeanSmoother()
    with np.testing.assert_raises(ValueError):
        smoother.predict(X_dummy)


def test_lsq_univariate_spline_close():
    np.random.seed(10)
    n = 200
    t = [-1, 0, 1]
    X = np.c_[np.linspace(-3, 3, n), np.ones(n)]
    y = np.exp(-X[:, 0] ** 2)
    truth = y.copy()
    y += 0.2 * np.random.randn(n)
    lsq = smoothing.onedim.LSQUnivariateSpline(t)
    lsq.fit(X, y)
    pred = lsq.predict(X)
    np.testing.assert_allclose(truth, pred, atol=0.1, rtol=0.1)


def test_lsq_univariate_spline_raise_not_fitted_exception():
    n = 10
    X_dummy = np.c_[np.arange(n), np.ones(n), np.zeros(n)]
    degree = 3
    smoother = smoothing.onedim.LSQUnivariateSpline([-1, 0, 1], degree=degree)
    with np.testing.assert_raises_regex(
        ValueError, "LSQUnivariateSpline has not been fitted!"
    ):
        smoother.predict(X_dummy)


def test_isotonic_regressor():
    np.random.seed(10)
    n = 200
    sigma = 1.0 / 10
    weights = np.ones(n) * sigma
    X = np.c_[np.arange(n), np.ones(n), weights]
    y_errors = np.random.randint(-50, 50, n)
    y_truth = 100.0 * np.log(1 + np.arange(n))
    y = y_errors + y_truth
    iso_r = smoothing.onedim.IsotonicRegressor()
    iso_r.fit(X, y)
    pred = iso_r.predict(X)
    np.testing.assert_allclose(y_truth, pred, atol=50, rtol=0.0)


def test_isotonic_regressor_empty_bins():
    n = 200
    sigma = 1.0 / 10
    weights = np.ones(n)
    # set some weights to zero
    weights[5] = 0
    weights[76] = 0
    weights[143] = 0

    sigma = 0.1 * weights
    x = np.arange(n)
    X = np.c_[x, weights, sigma]
    y = np.exp(0.1 * x) + 3 * x

    iso_r = smoothing.onedim.IsotonicRegressor()
    iso_r.fit(X, y)
    pred = iso_r.predict(X)
    np.testing.assert_allclose(y, pred, atol=50, rtol=0.0)


def test_isotonic_regressor_raise_not_fitted_exception():
    n = 10
    X_dummy = np.c_[np.arange(n), np.ones(n), np.zeros(n)]
    smoother = smoothing.onedim.IsotonicRegressor()
    with np.testing.assert_raises_regex(
        ValueError, "IsotonicRegressor has not been fitted!"
    ):
        smoother.predict(X_dummy)


def test_orthogonal_polynomial_constant_extrapolation():
    n = 100
    left, right = -3, 5
    x = np.linspace(left, right, n)
    slope = 2
    y = slope * x.copy()
    smoother = smoothing.onedim.OrthogonalPolynomialSmoother()

    X = np.c_[x, np.ones_like(y), np.ones_like(y)]

    smoother.fit(X, y)

    def check(value, expected):
        X = np.c_[[value], [1], [1]]
        np.testing.assert_allclose(smoother.predict(X), np.r_[expected])

    check(left, slope * left)
    check(left - 2, slope * left)

    check(right, slope * right)
    check(right + 3, slope * right)
