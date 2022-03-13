import unittest

import numpy as np
import pandas as pd

from cyclic_boosting.smoothing.onedim import OrthogonalPolynomialSmoother


class TestOrthogonalPolynomialSmootherByBlobelExample(unittest.TestCase):
    """Test case from the Blobel book"""

    def setUp(self):
        x = np.arange(20) + 1.0
        self.y = np.asarray(
            [
                5.0935,
                2.1777,
                0.2089,
                -2.3949,
                -2.4457,
                -3.0430,
                -2.2731,
                -2.0706,
                -1.6231,
                -2.5605,
                -0.7703,
                -0.3055,
                1.16817,
                1.8728,
                3.6586,
                3.2353,
                4.2520,
                5.2550,
                3.8766,
                4.2890,
            ]
        )

        assert len(x) == len(self.y)

        stddev = np.ones_like(x) * 0.6
        w = stddev ** -2

        self.X_for_smoother = np.c_[x, w, w]

    def test_cython_implementation(self):
        est = OrthogonalPolynomialSmoother().fit(self.X_for_smoother, self.y)
        y = est.predict(self.X_for_smoother)
        mad = np.mean(np.abs(y - self.y))
        np.testing.assert_allclose(mad, 1.315830289473684)


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


def test_orthogonal_polynomial_smoother_does_not_learn_target_by_heart():
    N = 3
    x = np.arange(N)
    y = np.poly1d([1, 3, 4])(x)
    X_for_smoother = np.c_[x, np.ones_like(x), np.ones_like(x)]
    est = OrthogonalPolynomialSmoother()
    est.fit(X_for_smoother, y)

    assert est.n_degrees_ == 2


def test_orthogonal_polynomial_smoother_degree_is_reduced_to_one_fith_of_the_supporting_points():
    N = 21
    x = np.arange(N)
    y = np.poly1d(list(range(8)))(x)
    variance = 1e-15  # choosing a low variance, so the fit is heavily constrained
    X_for_smoother = np.c_[x, np.ones_like(x), variance * np.ones_like(x)]
    est = OrthogonalPolynomialSmoother()
    est.fit(X_for_smoother, y)

    assert est.n_degrees_ == 4
