"""Unit tests for the plots module.
"""
from __future__ import absolute_import, division, print_function

import os
import unittest
from collections import OrderedDict

import numpy as np
import pandas as pd
from six.moves import range

from cyclic_boosting import (
    CBCoreClassifier,
    CBCoreLocationRegressor,
    CBExponential,
    CBFixedVarianceRegressor,
    CBPoissonRegressor,
    flags,
    observers,
    plots,
    testing_utils,
)
from cyclic_boosting.features import Feature, FeatureID
from cyclic_boosting.link import IdentityLinkMixin, Log2LinkMixin, LogitLinkMixin
from cyclic_boosting.plots import _guess_suitable_number_of_histogram_bins
from cyclic_boosting.plots._1dplots import (
    _ensure_tuple,
    _format_tick,
    _get_optimal_number_of_ticks,
    _get_x_axis,
    _get_y_axis,
    _plot_axes,
    _plot_factors,
    _plot_missing_factor,
    _plot_smoothed_factors,
)
from cyclic_boosting.plots._2dplots import plot_factor_2d
from cyclic_boosting.testing_utils import generate_binned_data, long_test


def get_inputs():
    n = 10000
    d = 10
    X, y = generate_binned_data(n, d)
    feature_prop = dict(
        [(str(i), flags.IS_CONTINUOUS) for i in range(5)]
        + [(str(i), flags.IS_UNORDERED) for i in range(5, 8)]
        + [("8", flags.IS_CONTINUOUS | flags.IS_LINEAR)]
        + [("9", flags.IS_CONTINUOUS | flags.IS_SEASONAL)]
    )
    feature_groups = [str(i) for i in range(d)] + [("1", "7"), ("2", "4", "8")]
    return X, y, feature_prop, feature_groups


class TestPlotsFactor(unittest.TestCase):
    @long_test
    def test_analysis_core_regressor(self):
        X, y, feature_prop, feature_groups = get_inputs()
        plobs = observers.PlottingObserver()
        est = CBFixedVarianceRegressor(
            feature_groups=feature_groups,
            feature_properties=feature_prop,
            observers=[plobs],
        )
        est.fit(X, y)
        with testing_utils.temp_dirname_created_and_removed() as dirname:
            name = "core_regressor"
            filepath = os.path.join(dirname, name)
            plots.plot_analysis(plobs, filepath)

    @long_test
    def test_analysis_poisson_regressor(self):
        X, y, feature_prop, feature_groups = get_inputs()
        plobs = observers.PlottingObserver()
        est = CBPoissonRegressor(
            feature_groups=feature_groups,
            feature_properties=feature_prop,
            observers=[plobs],
        )
        est.fit(X, y)
        with testing_utils.temp_dirname_created_and_removed() as dirname:
            name = "poisson_regressor"
            filepath = os.path.join(dirname, name)
            plots.plot_analysis(plobs, filepath)

    @long_test
    def test_analysis_exponential_regressor(self):
        X, y, feature_prop, _fg = get_inputs()
        X["10"] = np.random.uniform(low=0.6, high=1.25, size=len(X))

        feature_groups = [str(i) for i in range(9)] + [("1", "7"), ("2", "4", "8")]
        external_groups = [str(i) for i in range(3, 6)]

        plobs = observers.PlottingObserver()
        est = CBExponential(
            external_colname="10",
            standard_feature_groups=feature_groups,
            external_feature_groups=external_groups,
            feature_properties=feature_prop,
            observers=[plobs],
        )
        est.fit(X, y)
        with testing_utils.temp_dirname_created_and_removed() as dirname:
            name = "exponential_regressor"
            filepath = os.path.join(dirname, name)
            plots.plot_analysis(plobs, filepath)

    @long_test
    def test_analysis_regressor_with_file_handle(self):
        X, y, feature_prop, feature_groups = get_inputs()
        plobs = observers.PlottingObserver()
        est = CBFixedVarianceRegressor(
            feature_groups=feature_groups,
            feature_properties=feature_prop,
            observers=[plobs],
        )
        est.fit(X, y)
        with testing_utils.temp_dirname_created_and_removed() as dirname:
            name = "regressor-handle.pdf"
            filepath = os.path.join(dirname, name)
            assert not os.path.exists(filepath)
            with open(filepath, "wb") as f:
                plots.plot_analysis(plobs, f)
            assert os.path.exists(filepath)

    @long_test
    def test_analysis_location(self):
        X, y, feature_prop, feature_groups = get_inputs()
        y = np.sin(y) * 5.3
        plobs = observers.PlottingObserver()
        est = CBCoreLocationRegressor(
            feature_groups=feature_groups,
            feature_properties=feature_prop,
            observers=[plobs],
        )
        est.fit(X, y)
        with testing_utils.temp_dirname_created_and_removed() as dirname:
            name = "location"
            filepath = os.path.join(dirname, name)
            plots.plot_analysis(plobs, filepath)

    @long_test
    def test_analysis_classification(self):
        X, y, feature_prop, feature_groups = get_inputs()
        y = y > 5
        plobs = observers.PlottingObserver()
        est = CBCoreClassifier(
            feature_groups=feature_groups,
            feature_properties=feature_prop,
            observers=[plobs],
        )
        est.fit(X, y)
        with testing_utils.temp_dirname_created_and_removed() as dirname:
            name = "classification"
            filepath = os.path.join(dirname, name)
            plots.plot_analysis(plobs, filepath)


class TestSuitableNumberOfHistogramBins(unittest.TestCase):
    """Tests here should not assert too tightly, as we might adjust
    _guess_suitable_number_of_histogram_bins for corner cases. The assertions
    in the tests should just make sure that we don't have "weird" behaviour,
    like decreasing number of bins with increasing samle size, etc."""

    def test_number_of_bins_is_finite(self):
        """In the limit of sample size -> HUGE, we would like to still have a
        finite number of bins."""
        n1 = 1e10
        b1 = _guess_suitable_number_of_histogram_bins(n1)
        b2 = _guess_suitable_number_of_histogram_bins(10 * n1)
        assert b1 == b2

    def test_minimum_number_of_bins(self):
        """For very low sample sizes, we would still like to have a few bins"""
        b1 = _guess_suitable_number_of_histogram_bins(1)
        b2 = _guess_suitable_number_of_histogram_bins(2)
        assert b2 >= b1
        assert b1 > 0

    def test_reasonable_range_for_10000(self):
        b1 = _guess_suitable_number_of_histogram_bins(10000)
        assert 10 < b1 < 20


class TestFormatTick(unittest.TestCase):
    def test_integers(self):
        label = _format_tick(0.005, precision=1e-2)
        assert label == "0"
        label = _format_tick(23.992, precision=1e-2)
        assert label == "24"
        label = _format_tick(-30.001, precision=1e-2)
        assert label == "-30"

    def test_decimals(self):
        label = _format_tick(0.012, precision=1e-2)
        assert label == "0.01"
        label = _format_tick(23.98, precision=1e-2)
        assert label == "23.98"
        label = _format_tick(-23.5, precision=1e-2)
        assert label == "-23.50"
        label = _format_tick(0.505, precision=1e-2)
        assert label == "0.51"
        label = _format_tick(0.33, precision=1e-2)
        assert label == "0.33"
        label = _format_tick(-0.33, precision=1e-2)
        assert label == "-0.33"
        label = _format_tick(0.1, precision=1e-2)
        assert label == "0.10"


class TestGetOptimalNumberOfTicks(unittest.TestCase):
    def test_distance(self):
        assert _get_optimal_number_of_ticks(1) == 21
        assert _get_optimal_number_of_ticks(2) == 21
        assert _get_optimal_number_of_ticks(3) == 16
        assert _get_optimal_number_of_ticks(4) == 21
        assert _get_optimal_number_of_ticks(5) == 21
        assert _get_optimal_number_of_ticks(6) == 13
        assert _get_optimal_number_of_ticks(7) == 15
        assert _get_optimal_number_of_ticks(8) == 17
        assert _get_optimal_number_of_ticks(9) == 19
        assert _get_optimal_number_of_ticks(10) == 21
        for i in range(11, 21):
            assert _get_optimal_number_of_ticks(i) == i + 1

    def test_long_distance(self):
        assert _get_optimal_number_of_ticks(1000) == 21
        assert _get_optimal_number_of_ticks(2000) == 21
        assert _get_optimal_number_of_ticks(3000) == 16
        assert _get_optimal_number_of_ticks(4000) == 21
        assert _get_optimal_number_of_ticks(5000) == 21
        assert _get_optimal_number_of_ticks(6000) == 13
        assert _get_optimal_number_of_ticks(7000) == 15
        assert _get_optimal_number_of_ticks(8000) == 17
        assert _get_optimal_number_of_ticks(9000) == 19
        assert _get_optimal_number_of_ticks(1000) == 21
        for i in range(11, 21):
            assert _get_optimal_number_of_ticks(i * 1000) == i + 1

    def test_unexpected(self):
        assert _get_optimal_number_of_ticks(-1) == 21
        assert _get_optimal_number_of_ticks(0) == 21
        assert _get_optimal_number_of_ticks(np.nan) == 21
        assert _get_optimal_number_of_ticks(np.inf) == 21
        assert _get_optimal_number_of_ticks(-np.inf) == 21


class TestGetXAxis(unittest.TestCase):
    def setUp(self):
        self.factors = np.array([0.1, 0.5, -0.6, 0.4])
        self.bin_bounds = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    def test_bin_bounds_is_None(self):
        x_axis_range, x_labels = _get_x_axis(
            self.factors, bin_bounds=None, is_continuous=True
        )
        assert np.allclose(x_axis_range, np.array([0.0, 1.0, 2.0, 3.0]))
        assert x_labels is None

    def test_is_continuous(self):
        x_axis_range, x_labels = _get_x_axis(
            self.factors, self.bin_bounds, is_continuous=True
        )
        assert np.allclose(x_axis_range, np.array([0.0, 1.0, 2.0, 3.0]))
        assert np.allclose(x_labels, np.array([0.5, 1.5, 2.5, 3.5]))

    def test_is_not_continuous(self):
        x_axis_range, x_labels = _get_x_axis(
            self.factors, self.bin_bounds, is_continuous=False
        )
        assert np.allclose(x_axis_range, np.array([0.0, 1.0, 2.0, 3.0]))
        assert np.allclose(x_labels, np.array([1.0, 2.0, 3.0, 4.0]))


class TestGetYAxis(unittest.TestCase):
    def setUp(self):
        self.factors = np.array([0.1, 0.5, -0.6, 0.4])
        self.uncertainties = [
            np.array([0.4, 0.4, 2.0, 1.0]),
            np.array([0.4, 0.4, 2.0, 1.0]),
        ]
        self.expected_y_axis_range = np.array(
            [
                -1.0,
                -0.9,
                -0.8,
                -0.7,
                -0.6,
                -0.5,
                -0.4,
                -0.3,
                -0.2,
                -0.1,
                0.0,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                1.0,
            ]
        )
        self.expected_y_axis_range_with_uncertainty = np.array(
            [
                -3.0,
                -2.75,
                -2.5,
                -2.25,
                -2.0,
                -1.75,
                -1.5,
                -1.25,
                -1.0,
                -0.75,
                -0.5,
                -0.25,
                0.0,
                0.25,
                0.5,
                0.75,
                1.0,
                1.25,
                1.5,
                1.75,
                2.0,
            ]
        )

    def test_identity_link_function(self):
        link_function = IdentityLinkMixin()
        y_axis_range, y_labels = _get_y_axis(self.factors, link_function)
        assert np.allclose(y_axis_range, self.expected_y_axis_range)
        assert y_labels[0] == "-1"
        assert y_labels[-1] == "1"

    def test_identity_link_function_with_uncertainty(self):
        link_function = IdentityLinkMixin()
        y_axis_range, y_labels = _get_y_axis(
            self.factors, link_function, self.uncertainties
        )

        assert np.allclose(y_axis_range, self.expected_y_axis_range_with_uncertainty)
        assert y_labels[0] == "-3"
        assert y_labels[-1] == "2"

    def test_log2_link_function(self):
        link_function = Log2LinkMixin()
        y_axis_range, y_labels = _get_y_axis(self.factors, link_function)
        assert np.allclose(y_axis_range, self.expected_y_axis_range)
        assert y_labels[0] == "0.50"
        assert y_labels[-1] == "2"

    def test_log2_link_function_with_uncertainty(self):
        link_function = Log2LinkMixin()
        y_axis_range, y_labels = _get_y_axis(
            self.factors, link_function, self.uncertainties
        )
        assert np.allclose(y_axis_range, self.expected_y_axis_range_with_uncertainty)
        assert y_labels[0] == "0.12"
        assert y_labels[-1] == "4"

    def test_logit_link_function(self):
        link_function = LogitLinkMixin()
        y_axis_range, y_labels = _get_y_axis(self.factors, link_function)
        assert np.allclose(y_axis_range, self.expected_y_axis_range)
        assert y_labels[0] == "0.27"
        assert y_labels[-1] == "0.73"

    def test_logit_link_function_with_uncertainty(self):
        link_function = LogitLinkMixin()
        y_axis_range, y_labels = _get_y_axis(
            self.factors, link_function, self.uncertainties
        )
        assert np.allclose(y_axis_range, self.expected_y_axis_range_with_uncertainty)
        assert y_labels[0] == "0.05"
        assert y_labels[-1] == "0.88"


class TestEnsureTuple(unittest.TestCase):
    def test_ensure_tuple(self):
        assert _ensure_tuple(1) == (1,)
        assert _ensure_tuple((1, 2)) == (1, 2)
        assert _ensure_tuple((1,)) == (1,)


class TestPlotFactors(unittest.TestCase):
    def test_code_runs_without_error(self):
        _plot_factors(
            np.array([0.3, 0.4, 0.1]),
            np.array([1.0, 2.0, 3.0]),
            "label",
            uncertainties=None,
        )

    def test_code_runs_without_error_with_uncertainties(self):
        _plot_factors(
            np.array([0.3, 0.4, 0.1]),
            np.array([1.0, 2.0, 3.0]),
            "label",
            uncertainties=np.array([1.0, 2.0, 0.5]),
        )


class TestPlotSmoothedFactors(unittest.TestCase):
    def test_code_runs_without_error_with_continuous_factors(self):
        _plot_smoothed_factors(
            np.array([0.3, 0.4, 0.1]), np.array([1.0, 2.0, 3.0]), True
        )

    def test_code_runs_without_error_with_discrete_factors(self):
        _plot_smoothed_factors(
            np.array([0.3, 0.4, 0.1]), np.array([1.0, 2.0, 3.0]), False
        )


class TestPlotMissingFactor(unittest.TestCase):
    def test_code_runs_without_error_with_continuous_factors(self):
        _plot_missing_factor(
            np.array([0.3, 0.4, 0.1]),
            np.array([1.0, 2.0, 3.0]),
            np.array([0.0, 0.4, 1.0]),
        )


class TestPlotAxes(unittest.TestCase):
    def setUp(self):
        self.x_range = np.array([1.0, 2.0, 3.0])
        self.x_labels = ["1", "2", "3"]
        self.y_range = np.array([1.0, 1.5, 2.0, 3.0, 10.3])
        self.y_labels = ["1", "1.5", "2", "3", "10.3"]

    def test_code_runs_without_error_with_continuous_factors(self):
        _plot_axes(self.x_range, self.x_labels, self.y_range, self.y_labels, True)

    def test_code_runs_without_error_with_discrete_factors(self):
        _plot_axes(self.x_range, self.x_labels, self.y_range, self.y_labels, False)


def test_from_docstring_that_failed():
    from cyclic_boosting import CBFixedVarianceRegressor
    from cyclic_boosting.flags import (
        IS_CONTINUOUS,
        IS_LINEAR,
        IS_SEASONAL,
        IS_UNORDERED,
    )
    from cyclic_boosting.observers import PlottingObserver
    from cyclic_boosting.plots import plot_analysis
    from cyclic_boosting.testing_utils import (
        generate_binned_data,
        temp_dirname_created_and_removed,
    )

    X, y = generate_binned_data(10000, 4)
    plobs = PlottingObserver()
    est = CBFixedVarianceRegressor(
        feature_groups=["0", "1", "2", "3", ("1", "0"), ("2", "3", "0")],
        feature_properties=OrderedDict(
            [
                ("0", IS_CONTINUOUS),
                ("1", IS_UNORDERED),
                ("2", IS_CONTINUOUS | IS_LINEAR),
                ("3", IS_CONTINUOUS | IS_SEASONAL),
            ]
        ),
        observers=[plobs],
    )
    est.fit(X, y)

    with temp_dirname_created_and_removed() as dirname:
        plot_analysis(
            plot_observer=plobs,
            file_obj=os.path.join(dirname, "cyclic_boosting_analysis"),
        )


def test_plotting_does_not_crash_when_only_nan_feature_is_used(tmpdir):
    with tmpdir.as_cwd():
        X = pd.DataFrame(
            {"feature": [0, 1, 2, 3, 4] * 20, "non-feature": np.nan * np.ones(100)}
        )
        y = np.array([2, 3, 3, 1, 7] * 20, dtype=np.float64)

        feature_properties = {
            "feature": flags.IS_CONTINUOUS,
            "non-feature": flags.IS_CONTINUOUS,
        }
        observer = observers.PlottingObserver()

        regressor = CBFixedVarianceRegressor(
            feature_groups=X.columns,
            feature_properties=feature_properties,
            observers=[observer],
        )

        regressor.fit(X, y)
        plots.plot_analysis(observer, "test_ana_plots.pdf")


def test_plot_factor_2d_empty():
    feature = Feature(
        FeatureID(feature_group=("a", "b"), feature_type=None),
        (flags.IS_CONTINUOUS, flags.IS_UNORDERED),
        None,
    )
    feature.unfitted_factors_link = np.array([np.log(4)], dtype=np.float64)
    feature.factors_link = np.array([np.log(8)], dtype=np.float64)
    feature.unfitted_uncertainties_link = np.array([1], dtype=np.float64)
    feature.bin_weightsums = np.array([1000, 1000.0])
    feature.y = np.array([np.log(4)], dtype=np.float64)
    feature.prediction = np.array([np.log(4)], dtype=np.float64)
    feature.mean_dev = np.array([np.log(4)], dtype=np.float64)
    plot_factor_2d((0, 0), feature)
    import matplotlib.pyplot as plt

    plt.show()
