import os
import contextlib
import shutil
import tempfile

import numpy as np
import pandas as pd
from six.moves import range

from cyclic_boosting import CBClassifier, CBLocationRegressor, \
    CBExponential, CBNBinomRegressor, CBPoissonRegressor, flags, \
    observers, plots
from cyclic_boosting.plots import _guess_suitable_number_of_histogram_bins
from cyclic_boosting.plots._1dplots import _ensure_tuple,_format_tick, \
    _get_optimal_number_of_ticks, _get_x_axis, _get_y_axis, _plot_axes, \
    _plot_factors, _plot_missing_factor, _plot_smoothed_factors


@contextlib.contextmanager
def temp_dirname_created_and_removed(suffix="", prefix="tmp", base_dir=None):
    """Context manager returning the name of new temporary directory.

    At the end of the scope, the directory is automatically removed.
    All parameters are passed to :func:`tempfile.mkdtemp`.

    :param suffix: If specified, the file name will end with that suffix,
        otherwise there will be no suffix.
    :type suffix: string

    :param prefix: If specified, the file name will begin with that prefix;
        otherwise, a default prefix is used.
    :type prefix: string

    :param base_dir: If base_dir is specified,
        the file will be created in that directory.
        For details see ``dir`` parameter in :func:`tempfile.mkdtemp`.
    :type base_dir: str

    :return: Name of the temporary directory. The path of the
        directory is chosen by :func:`tempfile.mkdtemp`.
    :rtype: :obj:`str`
    """
    dirname = None
    try:
        dirname = tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=base_dir)
        yield dirname
    finally:
        if dirname is not None:
            shutil.rmtree(dirname)


def generate_binned_data(n_samples, n_features=10, seed=123):
    """
    Generate uncorrelated binned data for a sales problem.

    :param n_samples: number of samples that ought to be created.
    :type n_samples: int

    :param seed: random seed used in data creation
    :type seed: int

    :returns: Randomly generated binned feature matrix and binned target array.
    :rtype: :obj:`tuple` of :class:`pandas.DataFrame` and
        :class:`numpy.ndarray`

    """
    np.random.seed(seed)
    X = pd.DataFrame()
    y = np.random.randint(0, 10, n_samples)
    for i in range(0, n_features):
        n_bins = np.random.randint(1, 100)
        X[str(i)] = np.random.randint(0, n_bins, n_samples)
    return X, y


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


def test_analysis_core_regressor():
    X, y, feature_prop, feature_groups = get_inputs()
    plobs = observers.PlottingObserver()
    est = CBNBinomRegressor(
        feature_groups=feature_groups,
        feature_properties=feature_prop,
        observers=[plobs],
    )
    est.fit(X, y)
    with temp_dirname_created_and_removed() as dirname:
        name = "core_regressor"
        filepath = os.path.join(dirname, name)
        plots.plot_analysis(plobs, filepath)


def test_analysis_poisson_regressor():
    X, y, feature_prop, feature_groups = get_inputs()
    plobs = observers.PlottingObserver()
    est = CBPoissonRegressor(
        feature_groups=feature_groups,
        feature_properties=feature_prop,
        observers=[plobs],
    )
    est.fit(X, y)
    with temp_dirname_created_and_removed() as dirname:
        name = "poisson_regressor"
        filepath = os.path.join(dirname, name)
        plots.plot_analysis(plobs, filepath)


def test_analysis_exponential_regressor():
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
    with temp_dirname_created_and_removed() as dirname:
        name = "exponential_regressor"
        filepath = os.path.join(dirname, name)
        plots.plot_analysis(plobs, filepath)


def test_analysis_regressor_with_file_handle():
    X, y, feature_prop, feature_groups = get_inputs()
    plobs = observers.PlottingObserver()
    est = CBNBinomRegressor(
        feature_groups=feature_groups,
        feature_properties=feature_prop,
        observers=[plobs],
    )
    est.fit(X, y)
    with temp_dirname_created_and_removed() as dirname:
        name = "regressor-handle.pdf"
        filepath = os.path.join(dirname, name)
        assert not os.path.exists(filepath)
        with open(filepath, "wb") as f:
            plots.plot_analysis(plobs, f)
        assert os.path.exists(filepath)


def test_analysis_location():
    X, y, feature_prop, feature_groups = get_inputs()
    y = np.sin(y) * 5.3
    plobs = observers.PlottingObserver()
    est = CBLocationRegressor(
        feature_groups=feature_groups,
        feature_properties=feature_prop,
        observers=[plobs],
    )
    est.fit(X, y)
    with temp_dirname_created_and_removed() as dirname:
        name = "location"
        filepath = os.path.join(dirname, name)
        plots.plot_analysis(plobs, filepath)


def test_analysis_classification():
    X, y, feature_prop, feature_groups = get_inputs()
    y = y > 5
    plobs = observers.PlottingObserver()
    est = CBClassifier(
        feature_groups=feature_groups,
        feature_properties=feature_prop,
        observers=[plobs],
    )
    est.fit(X, y)
    with temp_dirname_created_and_removed() as dirname:
        name = "classification"
        filepath = os.path.join(dirname, name)
        plots.plot_analysis(plobs, filepath)


def test_number_of_bins_is_finite():
    # In the limit of sample size -> HUGE, we would like to still have a finite number of bins.
    n1 = 1e10
    b1 = _guess_suitable_number_of_histogram_bins(n1)
    b2 = _guess_suitable_number_of_histogram_bins(10 * n1)
    assert b1 == b2


def test_minimum_number_of_bins():
    # For very low sample sizes, we would still like to have a few bins
    b1 = _guess_suitable_number_of_histogram_bins(1)
    b2 = _guess_suitable_number_of_histogram_bins(2)
    assert b2 >= b1
    assert b1 > 0


def test_reasonable_range_for_10000():
    b1 = _guess_suitable_number_of_histogram_bins(10000)
    assert 10 < b1 < 20


def test_integers():
    label = _format_tick(0.005, precision=1e-2)
    assert label == "0"
    label = _format_tick(23.992, precision=1e-2)
    assert label == "24"
    label = _format_tick(-30.001, precision=1e-2)
    assert label == "-30"


def test_decimals():
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


def test_distance():
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


def test_long_distance():
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


def test_unexpected():
    assert _get_optimal_number_of_ticks(-1) == 21
    assert _get_optimal_number_of_ticks(0) == 21
    assert _get_optimal_number_of_ticks(np.nan) == 21
    assert _get_optimal_number_of_ticks(np.inf) == 21
    assert _get_optimal_number_of_ticks(-np.inf) == 21


def test_get_x_axis():
    factors = np.array([0.1, 0.5, -0.6, 0.4])
    bin_bounds = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    # no bounds
    x_axis_range, x_labels = _get_x_axis(
        factors, bin_bounds=None, is_continuous=True
    )
    assert np.allclose(x_axis_range, np.array([0.0, 1.0, 2.0, 3.0]))
    assert x_labels is None

    # continuous
    x_axis_range, x_labels = _get_x_axis(
        factors, bin_bounds, is_continuous=True
    )
    assert np.allclose(x_axis_range, np.array([0.0, 1.0, 2.0, 3.0]))
    assert np.allclose(x_labels, np.array([0.5, 1.5, 2.5, 3.5]))

    # not continoush
    x_axis_range, x_labels = _get_x_axis(
        factors, bin_bounds, is_continuous=False
    )
    assert np.allclose(x_axis_range, np.array([0.0, 1.0, 2.0, 3.0]))
    assert np.allclose(x_labels, np.array([1.0, 2.0, 3.0, 4.0]))


def test_get_y_axis():
    factors = np.array([0.1, 0.5, -0.6, 0.4])
    uncertainties = [
        np.array([0.4, 0.4, 2.0, 1.0]),
        np.array([0.4, 0.4, 2.0, 1.0]),
    ]
    expected_y_axis_range = np.array(
        [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0,
         0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )
    expected_y_axis_range_with_uncertainty = np.array(
        [-3.0, -2.75, -2.5, -2.25, -2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0,
         0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0,]
    )

    y_axis_range, y_labels = _get_y_axis(factors)
    assert np.allclose(y_axis_range, expected_y_axis_range)
    assert y_labels[0] == "-1"
    assert y_labels[-1] == "1"

    y_axis_range, y_labels = _get_y_axis(
        factors, uncertainties
    )
    assert np.allclose(y_axis_range, expected_y_axis_range_with_uncertainty)
    assert y_labels[0] == "-3"
    assert y_labels[-1] == "2"


def test_ensure_tuple():
    assert _ensure_tuple(1) == (1,)
    assert _ensure_tuple((1, 2)) == (1, 2)
    assert _ensure_tuple((1,)) == (1,)


def test_plot_factors():
    _plot_factors(
        np.array([0.3, 0.4, 0.1]),
        np.array([1.0, 2.0, 3.0]),
        "label",
        uncertainties=None,
    )


def test_plot_factors_with_uncertainties():
    _plot_factors(
        np.array([0.3, 0.4, 0.1]),
        np.array([1.0, 2.0, 3.0]),
        "label",
        uncertainties=np.array([1.0, 2.0, 0.5]),
    )


def test_plot_smoothed_factors():
    # continuous
    _plot_smoothed_factors(
        np.array([0.3, 0.4, 0.1]), np.array([1.0, 2.0, 3.0]), True
    )
    # discrete
    _plot_smoothed_factors(
        np.array([0.3, 0.4, 0.1]), np.array([1.0, 2.0, 3.0]), False
    )


def test_plot_missing_factor():
    _plot_missing_factor(
        np.array([0.3, 0.4, 0.1]),
        np.array([1.0, 2.0, 3.0]),
        np.array([0.0, 0.4, 1.0]),
    )


def test_plot_axes():
    x_range = np.array([1.0, 2.0, 3.0])
    x_labels = ["1", "2", "3"]
    y_range = np.array([1.0, 1.5, 2.0, 3.0, 10.3])
    y_labels = ["1", "1.5", "2", "3", "10.3"]
    # continuous
    _plot_axes(x_range, x_labels, y_range, y_labels, True)
    # discrete
    _plot_axes(x_range, x_labels, y_range, y_labels, False)


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

        regressor = CBNBinomRegressor(
            feature_groups=X.columns,
            feature_properties=feature_properties,
            observers=[observer],
        )

        regressor.fit(X, y)
        plots.plot_analysis(observer, "test_ana_plots.pdf")
