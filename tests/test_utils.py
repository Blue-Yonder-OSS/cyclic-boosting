import numpy as np
import pandas as pd

from cyclic_boosting import utils


def test_lin_reg_weight_one():
    alpha = 3.0
    beta = 0.5
    n = 100
    x = np.linspace(0, 10.0, n)
    y = alpha + beta * x
    w = np.ones(n)
    a, b = utils.linear_regression(x, y, w)
    np.testing.assert_almost_equal(alpha, a)
    np.testing.assert_almost_equal(beta, b)


def test_lin_reg_weighted_events():
    np.random.seed(456)
    alpha = 3.0
    beta = 0.5
    n = 100
    x = np.linspace(0, 10.0, n)
    y = alpha + beta * x
    err = np.array([np.mean(np.random.randn(n)) for i in range(n)])
    w = 1.0 / err**2
    a, b = utils.linear_regression(x, y + err, w)
    assert abs(alpha - a) < 0.01
    assert abs(beta - b) < 0.01


def test_get_feature_column_names():
    X = pd.DataFrame({"a": [0, 1], "b": [3, 6], "c": [2, 7]})
    features = utils.get_feature_column_names(X)
    np.testing.assert_equal(features, ["a", "b", "c"])
    features = utils.get_feature_column_names(X, exclude_columns=["a"])
    np.testing.assert_equal(features, ["b", "c"])


def test_continuous_quantile_from_discrete():
    y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    quantile_value = utils.continuous_quantile_from_discrete(y, 0.8)
    assert quantile_value == 8.0
    quantile_value = utils.continuous_quantile_from_discrete(y, 0.35)
    assert quantile_value == 3.0
    y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2)
    quantile_value = utils.continuous_quantile_from_discrete(y, 0.35)
    assert quantile_value == 3.5


def test_convergence_parameters():
    cp = utils.ConvergenceParameters()
    assert cp.loss_change == 1e20
    assert cp.delta == 100.0
