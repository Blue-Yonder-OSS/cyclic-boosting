import numpy as np
from sklearn import linear_model as lm

from cyclic_boosting import smoothing, utils
from cyclic_boosting.smoothing import RegressionType


# Neutralize2DMetaSmoother

def test_minimum_number_of_columns():
    y = np.array([0.91, 0.92, 0.93, 0.94])
    X = np.c_[np.arange(4), np.ones(4), [0.05, 0.05, 0.15, 0.15]]
    est = smoothing.multidim.Neutralize2DMetaSmoother(
        smoothing.multidim.WeightedMeanSmoother()
    )
    with np.testing.assert_raises(ValueError):
        est.fit(X, y)

def test_no_effect_on_neutralized_values():
    X = np.c_[
        np.repeat(np.arange(4), 4) * 1.0,
        np.tile(np.arange(4), 4) * 1.0,
        np.ones(16),
        np.ones(16),
    ]
    y = np.array([-1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1]).astype(
        float
    )
    est = smoothing.multidim.WeightedMeanSmoother()
    est2 = smoothing.multidim.Neutralize2DMetaSmoother(utils.clone(est))

    est.fit(X.copy(), y.copy())
    pred1 = est.predict(X[:, :2].copy())
    est2.fit(X.copy(), y.copy())
    pred2 = est2.predict(X[:, :2].copy())

    np.testing.assert_allclose(pred1, pred2)


def test_same_result_as_with_neutralized_values():
    X = np.c_[
        np.repeat(np.arange(4), 4) * 1.0,
        np.tile(np.arange(4), 4) * 1.0,
        np.ones(16),
        np.ones(16),
    ]
    y_neutralized = np.array(
        [-1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1]
    ).astype(float)
    est = smoothing.multidim.WeightedMeanSmoother()

    est.fit(X.copy(), y_neutralized.copy())
    pred1 = est.predict(X[:, :2].copy())

    est2 = smoothing.multidim.Neutralize2DMetaSmoother(utils.clone(est))

    est2.fit(X.copy(), y_neutralized.copy())
    pred2 = est2.predict(X[:, :2].copy())
    np.testing.assert_allclose(pred1, pred2)


# WeightedMeanSmoother

def test_weighted_mean_multidim():
    X = np.c_[
        np.repeat(np.arange(4), 4) * 1.0,
        np.tile(np.arange(4), 4) * 1.0,
        np.ones(16),
        np.ones(16),
    ]
    y = np.arange(16) * 0.5 + 2

    est = smoothing.multidim.WeightedMeanSmoother()
    est.fit(X, y)
    pred = est.predict(X[:, :2])

    est = smoothing.meta_smoother.NormalizationSmoother(
        smoothing.multidim.WeightedMeanSmoother()
    )
    est.fit(X, y)
    pred2 = est.predict(X[:, :2])
    assert est.norm_ == np.mean(y)
    np.testing.assert_allclose(pred, pred2)


def test_weighted_mean_apply_cut():
    w = np.tile([1, 0, 1], 3)
    X = np.c_[
        np.repeat(np.arange(3), 3) * 1.0,
        np.tile(np.arange(3), 3) * 1.0,
        w,
        np.ones(9),
    ]
    y = np.arange(9) * 0.5 + 2
    est = smoothing.meta_smoother.RegressionTypeSmoother(
        smoothing.multidim.WeightedMeanSmoother(),
        reg_type=RegressionType.discontinuous,
    )
    est.fit(X, y)
    pred = est.predict(X[:, :2])
    np.testing.assert_allclose(pred[w == 0], np.nan)


# GroupBySmoother

def test_gb_onedim():
    X = np.c_[
        [0, 0, 1, 1],
        [10, 10, 1, 1],
    ]
    Xt = X.copy()
    y = np.array([20, 10, 0.5, 2])
    n_dim = 2
    est = smoothing.multidim.GroupBySmoother(lm.LinearRegression(), n_dim)
    est.fit(X, y)
    p = est.predict(Xt)
    assert np.all(p == np.array([15.0, 15.0, 1.25, 1.25]))


def test_gb_multi():
    X = np.c_[
        [0, 0, 1, 1],
        [0, 1, 0, 1],
        [10, 10, 1, 1],
    ]
    Xt = X.copy()
    y = np.array([20, 10, 0.5, 2])
    n_dim = 3
    est = smoothing.multidim.GroupBySmoother(lm.LinearRegression(), n_dim)
    est.fit(X, y)
    p = est.predict(Xt)
    assert np.all(p == y)


def test_gb_multi_index_key_error():
    # Test to ensure that Pandas does not throw a "Too many indexer" error instead of the expected KeyError.
    X = np.c_[
        [0, 0, 1, 1],
        [0, 1, 0, 1],
        [10, 10, 1, 1],
    ]

    y = np.array([20, 10, 0.5, 2])
    n_dim = 3
    est = smoothing.multidim.GroupBySmoother(lm.LinearRegression(), n_dim)
    est.fit(X, y)

    Xt = X.copy()
    Xt[0, 1] = 2

    p = est.predict(Xt)
    assert np.isnan(p[0])
    assert np.all(p[1:] == y[1:])


def test_gb_other_keys_in_fit_than_predict():
    # Test to ensure that Pandas does not throw a "Too many indexer" error instead of the expected KeyError.
    xone = np.ones(8)

    X = np.c_[[0, 0, 1, 1, 2, 2, 4, 4], [0, 1, 0, 1, 0, 1, 0, 1], xone, xone]

    y = np.array([1, 2, 3, 5, 7, 9, 11, 12])
    n_dim = 2
    est = smoothing.multidim.GroupBySmootherCB(
        smoothing.onedim.WeightedMeanSmoother(), n_dim
    )
    est.fit(X, y)

    Xt = np.c_[[3, 3, 0, 0, 1, 1, 4, 4], [0, 1, 0, 1, 0, 1, 0, 1]]

    p = est.predict(Xt)
    assert np.isnan(p[0]) and np.isnan(p[1])
    expected = np.array([np.nan, np.nan, 1, 2, 3, 5, 11, 12])
    np.testing.assert_allclose(p, expected, atol=0.5)


def test_groupby_smoother_cb_weights():
    xone = np.ones(4)
    weights = [0, 0, 0, 1]
    X = np.c_[[0, 0, 1, 1], [0, 0, 0, 0], [0, 1, 0, 1], weights, xone]

    y = np.array([1, 2, 3, 5])
    n_dim = 3
    est = smoothing.multidim.GroupBySmootherCB(
        smoothing.onedim.WeightedMeanSmoother(), n_dim
    )
    est.fit(X, y)
    p = est.predict(X)
    assert np.all(np.isnan(p[:2]))
    assert not np.isnan(p[2])


def test_groupby_smoother_cb_weights_all_zero():
    xone = np.ones(4)
    weights = [0, 0, 0, 0]
    X = np.c_[[0, 0, 1, 1], [0, 1, 0, 1], weights, xone]

    y = np.array([1, 2, 3, 5])
    n_dim = 2
    est = smoothing.multidim.GroupBySmootherCB(
        smoothing.onedim.WeightedMeanSmoother(), n_dim
    )
    est.fit(X, y)
    p = est.predict(X)
    assert np.all(np.isnan(p))
