"""
Plots for the :ref:`cyclic_boosting_family`

**Examples**

.. plot::
    :include-source:
    :context:

    import os

    from cyclic_boosting import CBFixedVarianceRegressor
    from cyclic_boosting.observers import PlottingObserver
    from cyclic_boosting.plots import plot_analysis
    from cyclic_boosting.testing_utils import generate_binned_data, temp_dirname_created_and_removed
    from cyclic_boosting.flags import IS_CONTINUOUS, IS_UNORDERED, IS_LINEAR, IS_SEASONAL

    import matplotlib.pyplot as plt

    X, y = generate_binned_data(10000, 4)
    plobs = PlottingObserver()
    est = CBFixedVarianceRegressor(feature_groups=["0", "1", "2", "3", ("1", "0"), ("2", "3", "0")],
                            feature_properties={"0": IS_CONTINUOUS,
                                                "1": IS_UNORDERED,
                                                "2": IS_CONTINUOUS | IS_LINEAR,
                                                "3": IS_CONTINUOUS | IS_SEASONAL},
                            observers=[plobs])
    est.fit(X, y)

    with temp_dirname_created_and_removed() as dirname:
        plot_analysis(plot_observer=plobs,
                      file_obj=os.path.join(dirname, 'cyclic_boosting_analysis'))
"""
from __future__ import absolute_import, division, print_function

import contextlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.backends.backend_pdf import PdfPages

from cyclic_boosting.features import create_feature_id
from cyclic_boosting.utils import get_bin_bounds

from ._1dplots import plot_factor_1d
from ._2dplots import plot_factor_2d
from .plot_utils import append_extension, nbpy_style


def _guess_suitable_number_of_histogram_bins(n):
    """Guesses a suitable number of histograms for a given number of samples.

    https://en.wikipedia.org/wiki/Histogram#Number_of_bins_and_width for more
    information."""
    # sturges' rule is probably a bit better than the square-root choice in
    # many cases.
    sturges = int(np.ceil(np.log(n) / np.log(2) + 1))
    return np.clip(sturges, 5, 30)


def _factor_plots_y_limits(
    factor_arr, uncert_arr=None, percentage=0.05, ymin_for_zero=1e-3
):
    """
    >>> y = np.array([0, 1e-5, 0.1, 0.5, 1, 2, 10, 100, 1e5])
    >>> np.allclose(_factor_plots_y_limits(y), (-1e-3, 1e5 * 1.05))
    True

    >>> y = np.array([1e-5, 10])
    >>> np.allclose(_factor_plots_y_limits(y), (1e-5 * 0.95, 10 * 1.05))
    True
    """
    if uncert_arr is not None:
        ymax = np.max(factor_arr + uncert_arr)
        ymin = np.min(factor_arr - uncert_arr)
    else:
        ymax = np.max(factor_arr)
        ymin = np.min(factor_arr)

    if ymin == 0:
        ymin = -ymin_for_zero
    else:
        ymin = ymin * (1 - percentage)

    return ymin, ymax * (1 + percentage)


@nbpy_style
def plot_iteration_info(plot_observer):
    """
    Convenience method calling :func:`plot_loss` and
    :func:`plot_factor_change`.

    Parameters
    ----------
    plot_observer: :class:`~cyclic_boosting.observers.PlottingObserver`
        A fitted plotting observer.

    Example
    -------

    .. plot::
        :include-source:

        from cyclic_boosting.plots.plot_utils import _nbpy_style_figure
        from nbpy.testing import plotting as plot_testing
        from cyclic_boosting import plots

        plobs = plot_testing.get_plotting_observer_from_CBRegressor()

        with _nbpy_style_figure(figsize=(13., 8.)):
            plots.plot_iteration_info(plobs)
    """
    plt.subplot(211)
    plot_loss(plot_observer)
    plt.subplot(212)
    plot_factor_change(plot_observer)


@nbpy_style
def plot_factor_change(plot_observer):
    """
    Plot the global factor changes for all iterations.

    Parameters
    ----------
    plot_observer: :class:`~cyclic_boosting.observers.PlottingObserver`
        A fitted plotting observer.

    Example
    -------

    .. plot::
        :include-source:

        from cyclic_boosting.plots.plot_utils import _nbpy_style_figure
        from nbpy.testing import plotting as plot_testing
        from cyclic_boosting import plots

        plobs = plot_testing.get_plotting_observer_from_CBRegressor()

        with _nbpy_style_figure(figsize=(13., 5.)):
            plots.plot_factor_change(plobs)
    """
    factor_changes = plot_observer.factor_change
    n = len(factor_changes)
    iterations = np.arange(1, n + 1)
    plt.xlabel("Iterations")
    plt.ylabel("Factor change of all features")
    ymin, ymax = _factor_plots_y_limits(factor_changes)
    plt.ylim(ymin, ymax)
    plt.xlim(0.9, n + 0.1)
    plt.plot(iterations, factor_changes, "ob-")
    plt.grid(True)


@nbpy_style
def plot_loss(plot_observer):
    """
    Plot the change of the loss function between the in-sample y
    and the predicted factors for all iterations.
    For the zeroth iteration the mean of y is used as prediction.

    Parameters
    ----------
    plot_observer: :class:`~cyclic_boosting.observers.PlottingObserver`
        A fitted plotting observer.

    Example
    -------

    .. plot::
        :include-source:

        from cyclic_boosting.plots.plot_utils import _nbpy_style_figure
        from nbpy.testing import plotting as plot_testing
        from cyclic_boosting import plots

        plobs = plot_testing.get_plotting_observer_from_CBRegressor()

        with _nbpy_style_figure(figsize=(13., 5.)):
            plots.plot_loss(plobs)
    """
    loss = plot_observer.loss
    n = len(loss)
    iterations = np.arange(n)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    ymin, ymax = _factor_plots_y_limits(loss)
    plt.ylim(ymin, ymax)
    plt.xlim(-0.1, n + 0.1)
    plt.plot(iterations, loss, "o-r")
    plt.grid(True)


@nbpy_style
def plot_in_sample_diagonal_plot(plot_observer):
    """
    Plot the in sample diagonal plot for prediction and truth.

    Parameters
    ----------
    plot_observer: :class:`~cyclic_boosting.observers.PlottingObserver`
        A fitted plotting observer.

    Example
    -------

    .. plot::
        :include-source:

        from cyclic_boosting.plots.plot_utils import _nbpy_style_figure
        from nbpy.testing import plotting as plot_testing
        from cyclic_boosting import plots

        plobs = plot_testing.get_plotting_observer_from_CBRegressor()

        with _nbpy_style_figure(figsize=(4., 4.)):
            plots.plot_in_sample_diagonal_plot(plobs)
    """
    plot_observer.check_fitted()
    means, bin_centers, errors, _ = plot_observer.histograms

    plt.plot(bin_centers, means, "o", color="b")
    ymin, ymax = plt.ylim()
    plt.errorbar(
        bin_centers,
        means,
        yerr=[-errors[0], errors[1]],
        fmt="none",
        ecolor="b",
        capsize=2.5,
    )
    xmin, xmax = plt.xlim()
    plt.ylim(min(xmin, ymin), max(xmax, ymax))
    plt.xlim(xmin, xmax)
    plt.plot([xmin, xmax], [xmin, xmax], "k", linewidth=0.4)
    plt.xlabel("Prediction")
    plt.ylabel("Truth")


def plot_analysis(
    plot_observer,
    file_obj,
    binners=None,
    figsize=(11.69, 8.27),
    use_tightlayout=True,
    plot_yp=True,
):
    """
    Plot factors as a multipage PDF, also include plots for Loss and
    factor change behaviour and an insample diagonal plot.

    Parameters
    ----------
    plot_observer: :class:`~cyclic_boosting.observers.PlottingObserver`
        A fitted plotting observer.
    file: string or file-like
        If a string indicates the name of the file, to which the plots are
        written. The ending '.pdf' is added if it is not present already.
        You may also pass a file-like object.
    binners: list
        A list of binners. If binners are given the labels of the x-axis of
        factor plots are better interpretable.
    figsize: tuple
        A tuple with length containing the width and height of the figures.
    use_tightlayout: bool
        If true the tightlayout option of matplotlib is used.
    """
    filepath_or_object = append_extension(file_obj, ".pdf")
    dpi = 200
    with contextlib.closing(PdfPages(filepath_or_object)) as pdf_pages:

        plt.figure(figsize=figsize)
        plot_in_sample_diagonal_plot(plot_observer)
        plt.savefig(pdf_pages, format="pdf", dpi=dpi)

        for feature in plot_observer.features:
            plt.figure(figsize=figsize)
            grid = gridspec.GridSpec(1, 1)
            _plot_one_feature_group(
                plot_observer,
                grid[0],
                feature,
                binners,
                use_tightlayout,
                plot_yp=plot_yp,
            )
            plt.savefig(pdf_pages, format="pdf", dpi=dpi)

        # plt.figure(figsize=figsize)
        # plot_factor_change(plot_observer)
        # plt.savefig(pdf_pages, format="pdf", dpi=dpi)
        plt.figure(figsize=figsize)
        plot_loss(plot_observer)
        plt.savefig(pdf_pages, format="pdf", dpi=dpi)
        # for feature in plot_observer.features:
        #     plt.figure(figsize=figsize)
        #     f_sum = feature.factor_sum
        #     # f_sum /= f_sum[-1]
        #     plt.plot(f_sum)
        #     plt.title(
        #         "absolute factor sum for {} over iterations".format(
        #             feature.feature_group
        #         )
        #     )
        #     plt.savefig(pdf_pages, format="pdf", dpi=dpi)
    plt.close("all")


@nbpy_style
def plot_factors(
    plot_observer,
    binners=None,
    feature_groups_or_ids=None,
    features_per_row=2,
    use_tightlayout=True,
    plot_yp=True,
):
    """
    Create a matplotlib figure containing several factor plots (of a
    possibly pre-defined subset of features) in a grid.

    Parameters
    ----------
    plot_observer: :class:`~cyclic_boosting.observers.PlottingObserver`
        A fitted plotting observer.
    binners: list
        A list of binners. If binners are given the labels of the x-axis of
        factor plots are better interpretable.
    feature_groups_or_ids: list
        A list of feature group names or
        :class:`cyclic_boosting.base.FeatureID`
        for which the factors will be plotted.
        Default is to plot the factors of all features.
    features_per_row: int
        The number of factor plots in one row.
    use_tightlayout: bool
        If true the tightlayout option of matplotlib is used.

    Example
    -------

    .. plot::
        :include-source:

        from cyclic_boosting.plots.plot_utils import _nbpy_style_figure
        from nbpy.testing import plotting as plot_testing
        from cyclic_boosting import plots

        plobs = plot_testing.get_plotting_observer_from_CBRegressor()

        with _nbpy_style_figure(figsize=(13., 10.)):
            plots.plot_factors(plobs)
    """
    plot_observer.check_fitted()
    if feature_groups_or_ids is None:
        features = plot_observer.features
    else:
        feature_ids = [
            create_feature_id(feature_group_or_id)
            for feature_group_or_id in feature_groups_or_ids
        ]
        features = [plot_observer.features[feature_id] for feature_id in feature_ids]

    n_plots = len(features)
    grid = gridspec.GridSpec(int(np.ceil(n_plots / features_per_row)), features_per_row)

    for i, feature in enumerate(features):
        _plot_one_feature_group(
            plot_observer, grid[i], feature, binners, use_tightlayout, plot_yp=plot_yp
        )


def _format_groupname_with_type(feature_group, feature_type):
    name = ", ".join(feature_group)
    if feature_type is None:
        return name
    else:
        return "{0} ({1})".format(name, feature_type)


def _plot_one_feature_group(
    plot_observer, grid_item, feature, binners=None, use_tightlayout=True, plot_yp=True
):
    """
    Create a factor subplot for one feature group. This function is probably
    bet not called by the user directly. The user will invoke a plotting
    method of the estimator directly that will indirectly invoke this
    method. :meth:`nbpy.factor_model.CBFixedVarianceRegressor.plot_factors`.
    """

    if len(feature.feature_group) == 1:
        # treatment of one-dimensional features
        plt.subplot(grid_item)
        plot_factor_1d(
            feature,
            bin_bounds=get_bin_bounds(binners, feature.feature_group[0]),
            link_function=plot_observer.link_function,
            plot_yp=plot_yp,
        )
        plt.grid(True, which="both")

    elif len(feature.feature_group) == 2:
        # treatment of two-dimensional features
        plot_factor_2d(
            n_bins_finite=plot_observer.n_feature_bins[feature.feature_group],
            feature=feature,
            grid_item=grid_item,
        )

    else:
        plt.subplot(grid_item)
        plot_factor_high_dim(feature)

    if use_tightlayout:
        plt.tight_layout()


def plot_factor_histogram(feature):
    """
    Plots a histogram of the given factors with a logarithmic y-axis.

    Parameters
    ----------
    feature: cyclic_boosting.base.Feature
        Feature object from as it can be obtained from a plotting
        observer

    Example
    -------
    .. plot::
        :include-source:

        from cyclic_boosting.plots.plot_utils import _nbpy_style_figure
        from nbpy.testing import plotting as plot_testing
        from cyclic_boosting import plots

        plobs = plot_testing.get_plotting_observer_from_CBRegressor()
        feature = plobs.features["continuous feature"]

        with _nbpy_style_figure(figsize=(6., 5.)):
            plots.plot_factor_histogram(feature)
    """
    factors = feature.unfitted_factors_link
    smoothed_factors = feature.factors_link
    feat_group = feature.feature_group
    feature_type = feature.feature_type
    feat_group = _format_groupname_with_type(feat_group, feature_type)
    n_bins = _guess_suitable_number_of_histogram_bins(len(factors))

    gs = gridspec.GridSpec(2, 1)

    plt.sca(plt.subplot(gs[0, 0]))
    plt.title("Unsmoothed Factor-Histogram for {0}".format(feat_group))
    plt.xlabel("Factor")
    plt.ylabel("Count")
    plt.hist(factors, bins=n_bins, log=True)

    plt.sca(plt.subplot(gs[1, 0]))
    dev = smoothed_factors - factors
    plt.hist(dev, bins=100, log=True)
    plt.title("smoothed_factors - factors")
    plt.xlabel("Factor")
    plt.ylabel("Count")
    plt.hist(smoothed_factors - factors, bins=100, log=True)


plot_factor_high_dim = plot_factor_histogram


__all__ = [
    "plot_iteration_info",
    "plot_factor_change",
    "plot_loss",
    "plot_in_sample_diagonal_plot",
    "plot_analysis",
    "plot_factors",
    "plot_factor_1d",
    "plot_factor_2d",
    "plot_factor_high_dim",
    "plot_factor_histogram",
]
