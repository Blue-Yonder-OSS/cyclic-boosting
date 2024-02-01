import numpy as np

from cyclic_boosting.tornado.trainer import metrics


def test_mean_deviation() -> None:
    yhat = np.arange(1, 10)
    y = yhat - (1/2 * yhat)
    desired = -2.5
    actual = metrics.mean_deviation(y, yhat)
    np.testing.assert_almost_equal(actual, desired, decimal=3)


def test_mean_absolute_deviation() -> None:
    yhat = np.arange(1, 10)
    y = yhat - (1/2 * yhat)
    desired = 2.944
    actual = metrics.mean_absolute_deviation(y, yhat)
    np.testing.assert_almost_equal(actual, desired, decimal=3)


def test_mean_square_error() -> None:
    yhat = np.arange(1, 10)
    y = yhat - (1/2 * yhat)
    desired = 7.916
    actual = metrics.mean_square_error(y, yhat)
    np.testing.assert_almost_equal(actual, desired, decimal=3)


def test_mean_absolute_error() -> None:
    yhat = np.arange(1, 10)
    y = yhat - (1/2 * yhat)
    desired = 2.5
    actual = metrics.mean_absolute_error(y, yhat)
    np.testing.assert_almost_equal(actual, desired, decimal=3)


def test_median_absolute_error() -> None:
    yhat = np.arange(1, 10)
    y = yhat - (1/2 * yhat)
    desired = 2.5
    actual = metrics.median_absolute_error(y, yhat)
    np.testing.assert_almost_equal(actual, desired, decimal=3)


def test_weighted_absolute_percentage_error() -> None:
    yhat = np.arange(1, 10)
    y = yhat - (1/2 * yhat)
    desired = 3.166
    actual = metrics.weighted_absolute_percentage_error(y, yhat)
    np.testing.assert_almost_equal(actual, desired, decimal=3)


def test_symmetric_mean_absolute_percentage_error() -> None:
    yhat = np.arange(1, 10)
    y = yhat - (1/2 * yhat)
    desired = 66.6666
    actual = metrics.symmetric_mean_absolute_percentage_error(y, yhat)
    np.testing.assert_almost_equal(actual, desired, decimal=3)


def test_mean_y() -> None:
    yhat = np.arange(1, 10)
    y = yhat - (1/2 * yhat)
    desired = 2.5
    actual = metrics.mean_y(y)
    np.testing.assert_almost_equal(actual, desired, decimal=3)


def test_coefficient_of_determination() -> None:
    yhat = np.arange(1, 10)
    y = yhat - (1/2 * yhat)
    desired = -4.428
    actual = metrics.coefficient_of_determination(y, yhat, k=1)
    np.testing.assert_almost_equal(actual, desired, decimal=3)


def test_F() -> None:
    yhat = np.arange(1, 10)
    y = yhat - (1/2 * yhat)
    desired = 11.421
    actual = metrics.F(y, yhat, k=1)
    np.testing.assert_almost_equal(actual, desired, decimal=3)


def test_mean_pinball_loss() -> None:
    yhat = np.arange(1, 10)
    y = yhat - (1/2 * yhat)
    desired = 2.0
    actual = metrics.mean_pinball_loss(y, yhat, alpha=0.2)
    np.testing.assert_almost_equal(actual, desired, decimal=3)
