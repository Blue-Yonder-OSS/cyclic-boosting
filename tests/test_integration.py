import numpy as np
import pytest


from scipy.special import factorial

from cyclic_boosting import flags, common_smoothers, observers
from cyclic_boosting.smoothing.onedim import SeasonalSmoother, IsotonicRegressor
from cyclic_boosting.pipelines import (
    pipeline_CBPoissonRegressor,
    pipeline_CBClassifier,
    pipeline_CBLocationRegressor,
    pipeline_CBExponential,
    pipeline_CBNBinomRegressor,
    pipeline_CBNBinomC,
    pipeline_CBGBSRegressor,
    pipeline_CBMultiplicativeQuantileRegressor,
    pipeline_CBAdditiveQuantileRegressor,
    pipeline_CBMultiplicativeGenericCRegressor,
    pipeline_CBAdditiveGenericCRegressor,
    pipeline_CBGenericClassifier,
)
from tests.utils import plot_CB, costs_mad, costs_mse

np.random.seed(42)


@pytest.fixture(scope="function")
def cb_poisson_regressor_model(features, feature_properties):
    explicit_smoothers = {
        ("dayofyear",): SeasonalSmoother(order=3),
        ("price_ratio",): IsotonicRegressor(increasing=False),
    }

    plobs = [
        observers.PlottingObserver(iteration=1),
        observers.PlottingObserver(iteration=-1),
    ]

    CB_pipeline = pipeline_CBPoissonRegressor(
        feature_properties=feature_properties,
        feature_groups=features,
        observers=plobs,
        maximal_iterations=50,
        smoother_choice=common_smoothers.SmootherChoiceGroupBy(
            use_regression_type=True, use_normalization=False, explicit_smoothers=explicit_smoothers
        ),
    )

    return CB_pipeline


def test_poisson_regression(is_plot, prepare_data, cb_poisson_regressor_model):
    X, y = prepare_data

    CB_est = cb_poisson_regressor_model
    CB_est.fit(X.copy(), y)

    if is_plot:
        plot_CB("analysis_CB_iterfirst", [CB_est[-1].observers[0]], CB_est[-2])
        plot_CB("analysis_CB_iterlast", [CB_est[-1].observers[-1]], CB_est[-2])

    yhat = CB_est.predict(X.copy())

    mad = np.nanmean(np.abs(y - yhat))
    np.testing.assert_almost_equal(mad, 1.6997, 3)


def test_poisson_regression_default_features(prepare_data, default_features, feature_properties):
    X, y = prepare_data
    X = X[default_features]

    CB_est = pipeline_CBPoissonRegressor(feature_properties=feature_properties)
    CB_est.fit(X.copy(), y)

    yhat = CB_est.predict(X.copy())

    mad = np.nanmean(np.abs(y - yhat))
    np.testing.assert_almost_equal(mad, 1.7185, 3)


@pytest.mark.parametrize(("feature_groups", "expected"), [(None, 1.689), ([0, 1, 4, 5], 1.950)])
def test_poisson_regression_ndarray(prepare_data, default_features, feature_properties, feature_groups, expected):
    X, y = prepare_data
    X = X[default_features].to_numpy()

    CB_est = pipeline_CBPoissonRegressor(feature_groups=feature_groups, feature_properties=feature_properties)
    CB_est.fit(X.copy(), y)

    yhat = CB_est.predict(X.copy())

    mad = np.nanmean(np.abs(y - yhat))
    np.testing.assert_almost_equal(mad, expected, 3)


@pytest.mark.parametrize("regressor", ["BinomRegressor", "PoissonRegressor"])
def test_regression_ndarray_w_feature_properties(prepare_data, default_features, regressor):
    X, y = prepare_data
    X = X[default_features].to_numpy()

    fp = {
        0: flags.IS_UNORDERED,
        2: flags.IS_UNORDERED,
        3: flags.IS_ORDERED,
        5: flags.IS_CONTINUOUS | flags.HAS_MISSING | flags.MISSING_NOT_LEARNED,
        6: flags.IS_ORDERED,
    }

    if regressor == "BinomRegressor":
        CB_est = pipeline_CBNBinomRegressor(feature_properties=fp)
    else:
        CB_est = pipeline_CBPoissonRegressor(feature_properties=fp)

    CB_est.fit(X.copy(), y)
    yhat = CB_est.predict(X.copy())
    mad = np.nanmean(np.abs(y - yhat))
    np.testing.assert_almost_equal(mad, 1.695, 3)


def test_poisson_regression_default_features_and_properties(is_plot, prepare_data, default_features):
    X, y = prepare_data
    X = X[default_features]

    plobs = [
        observers.PlottingObserver(iteration=1),
        observers.PlottingObserver(iteration=-1),
    ]
    CB_est = pipeline_CBPoissonRegressor(
        observers=plobs,
    )
    CB_est.fit(X.copy(), y)

    if is_plot:
        plot_CB("analysis_CB_iterfirst", [CB_est[-1].observers[0]], CB_est[-2])
        plot_CB("analysis_CB_iterlast", [CB_est[-1].observers[-1]], CB_est[-2])

    yhat = CB_est.predict(X.copy())

    mad = np.nanmean(np.abs(y - yhat))
    np.testing.assert_almost_equal(mad, 1.6982, 3)


def test_poisson_regression_default_features_notaggregated(prepare_data, default_features, feature_properties):
    X, y = prepare_data
    X = X[default_features]

    CB_est = pipeline_CBPoissonRegressor(feature_properties=feature_properties, aggregate=False)
    CB_est.fit(X.copy(), y)

    yhat = CB_est.predict(X.copy())

    mad = np.nanmean(np.abs(y - yhat))
    np.testing.assert_almost_equal(mad, 1.7144, 3)


def test_nbinom_regression_default_features(prepare_data, default_features, feature_properties):
    X, y = prepare_data
    X = X[default_features]

    CB_est = pipeline_CBNBinomRegressor(
        feature_properties=feature_properties,
        a=1.2,
        c=0.1,
    )
    CB_est.fit(X.copy(), y)

    yhat = CB_est.predict(X.copy())

    mad = np.nanmean(np.abs(y - yhat))
    np.testing.assert_almost_equal(mad, 1.7198, 3)


@pytest.mark.parametrize(("feature_groups", "expected"), [(None, 1.689), ([0, 1, 4, 5], 1.950)])
def test_nbinom_regression_ndarray(prepare_data, default_features, feature_properties, feature_groups, expected):
    X, y = prepare_data
    X = X[default_features].to_numpy()

    fp = feature_properties
    CB_est = pipeline_CBNBinomRegressor(
        feature_groups=feature_groups,
        feature_properties=fp,
    )
    CB_est.fit(X.copy(), y)

    yhat = CB_est.predict(X.copy())

    mad = np.nanmean(np.abs(y - yhat))
    np.testing.assert_almost_equal(mad, expected, 3)


@pytest.fixture(scope="function")
def cb_exponential_regressor_model(features, feature_properties):
    features = features
    features.remove("price_ratio")
    price_features = [
        "L_ID",
        "PG_ID_1",
        "PG_ID_2",
        "PG_ID_3",
        "P_ID",
        "dayofweek",
    ]

    feature_properties = feature_properties
    explicit_smoothers = {
        ("dayofyear",): SeasonalSmoother(order=3),
        ("price_ratio",): IsotonicRegressor(increasing=False),
    }

    plobs = [
        observers.PlottingObserver(iteration=1),
        observers.PlottingObserver(iteration=-1),
    ]

    CB_pipeline = pipeline_CBExponential(
        feature_properties=feature_properties,
        standard_feature_groups=features,
        external_feature_groups=price_features,
        external_colname="price_ratio",
        observers=plobs,
        maximal_iterations=50,
        smoother_choice=common_smoothers.SmootherChoiceGroupBy(
            use_regression_type=True, use_normalization=False, explicit_smoothers=explicit_smoothers
        ),
    )

    return CB_pipeline


def test_exponential_regression(is_plot, prepare_data, cb_exponential_regressor_model):
    X, y = prepare_data
    X.loc[X["price_ratio"] == np.nan, "price_ratio"] = 1.0

    CB_est = cb_exponential_regressor_model
    CB_est.fit(X.copy(), y)

    if is_plot:
        plot_CB("analysis_CB_iterfirst", [CB_est[-1].observers[0]], CB_est[-2])
        plot_CB("analysis_CB_iterlast", [CB_est[-1].observers[-1]], CB_est[-2])

    yhat = CB_est.predict(X.copy())

    mad = np.nanmean(np.abs(y - yhat))
    np.testing.assert_almost_equal(mad, 1.7203, 3)


@pytest.fixture(scope="function")
def cb_classifier_model(features, feature_properties):
    explicit_smoothers = {
        ("dayofyear",): SeasonalSmoother(order=3),
        ("price_ratio",): IsotonicRegressor(increasing=False),
    }

    plobs = [observers.PlottingObserver(iteration=-1)]

    CB_pipeline = pipeline_CBClassifier(
        feature_properties=feature_properties,
        feature_groups=features,
        observers=plobs,
        maximal_iterations=50,
        smoother_choice=common_smoothers.SmootherChoiceGroupBy(
            use_regression_type=True, use_normalization=False, explicit_smoothers=explicit_smoothers
        ),
    )

    return CB_pipeline


def test_classification(is_plot, prepare_data, cb_classifier_model):
    X, y = prepare_data
    y = y >= 3

    CB_est = cb_classifier_model
    CB_est.fit(X.copy(), y)

    if is_plot:
        plot_CB("analysis_CB_iterlast", [CB_est[-1].observers[-1]], CB_est[-2])

    yhat = CB_est.predict(X.copy())

    mad = np.nanmean(np.abs(y - yhat))
    np.testing.assert_almost_equal(mad, 0.3075, 3)


def test_location_regression_default_features(is_plot, feature_properties, default_features, prepare_data):
    X, y = prepare_data
    X = X[default_features]

    fp = feature_properties

    CB_est = pipeline_CBLocationRegressor(feature_properties=fp)
    CB_est.fit(X.copy(), y)

    if is_plot:
        plot_CB("analysis_CB_iterlast", [CB_est[-1].observers[-1]], CB_est[-2])

    yhat = CB_est.predict(X.copy())

    mad = np.nanmean(np.abs(y - yhat))
    np.testing.assert_almost_equal(mad, 1.7511, 3)


@pytest.fixture(scope="function")
def cb_width_model(feature_properties):
    features = ["dayofweek", "L_ID", "PG_ID_3", "PROMOTION_TYPE"]

    explicit_smoothers = {}

    plobs = [observers.PlottingObserver(iteration=-1)]

    CB_pipeline = pipeline_CBNBinomC(
        mean_prediction_column="yhat_mean",
        feature_properties=feature_properties,
        feature_groups=features,
        observers=plobs,
        maximal_iterations=50,
        smoother_choice=common_smoothers.SmootherChoiceGroupBy(
            use_regression_type=True, use_normalization=False, explicit_smoothers=explicit_smoothers
        ),
    )

    return CB_pipeline


def test_width_regression_default_features(feature_properties, default_features, prepare_data, cb_width_model):
    X, y = prepare_data
    X = X[default_features]

    fp = feature_properties
    CB_est = pipeline_CBPoissonRegressor(feature_properties=fp)
    CB_est.fit(X.copy(), y)
    yhat = CB_est.predict(X.copy())
    X["yhat_mean"] = yhat

    CB_est_width = cb_width_model
    CB_est_width.fit(X.copy(), y)
    c = CB_est_width.predict(X.copy())
    np.testing.assert_almost_equal(c.mean(), 0.365, 3)


def test_GBS_regression_default_features(is_plot, feature_properties, default_features, prepare_data):
    X, y = prepare_data
    X = X[default_features]

    y[1000:10000] = -y[1000:10000]

    CB_est = pipeline_CBGBSRegressor(feature_properties=feature_properties)
    CB_est.fit(X.copy(), y)

    if is_plot:
        plot_CB("analysis_CB_iterlast", [CB_est[-1].observers[-1]], CB_est[-2])

    yhat = CB_est.predict(X.copy())

    mad = np.nanmean(np.abs(y - yhat))
    np.testing.assert_almost_equal(mad, 2.5755, 3)


def evaluate_quantile(y, yhat):
    quantile_acc = (y <= yhat).mean()
    return quantile_acc


def cb_multiplicative_quantile_regressor_model(quantile, features, feature_properties):
    explicit_smoothers = {
        ("dayofyear",): SeasonalSmoother(order=3),
        ("price_ratio",): IsotonicRegressor(increasing=False),
    }

    plobs = [
        observers.PlottingObserver(iteration=1),
        observers.PlottingObserver(iteration=-1),
    ]

    CB_pipeline = pipeline_CBMultiplicativeQuantileRegressor(
        quantile=quantile,
        feature_properties=feature_properties,
        feature_groups=features,
        observers=plobs,
        maximal_iterations=50,
        smoother_choice=common_smoothers.SmootherChoiceGroupBy(
            use_regression_type=True, use_normalization=False, explicit_smoothers=explicit_smoothers
        ),
    )

    return CB_pipeline


def test_multiplicative_quantile_regression_median(is_plot, prepare_data, features, feature_properties):
    X, y = prepare_data
    y = abs(y)

    quantile = 0.5
    CB_est = cb_multiplicative_quantile_regressor_model(
        quantile=quantile, features=features, feature_properties=feature_properties
    )
    CB_est.fit(X.copy(), y)

    if is_plot:
        plot_CB("analysis_CB_iterfirst", [CB_est[-1].observers[0]], CB_est[-2])
        plot_CB("analysis_CB_iterlast", [CB_est[-1].observers[-1]], CB_est[-2])

    yhat = CB_est.predict(X.copy())

    quantile_acc = evaluate_quantile(y, yhat)
    np.testing.assert_almost_equal(quantile_acc, 0.5043, 3)

    mad = np.nanmean(np.abs(y - yhat))
    np.testing.assert_almost_equal(mad, 1.6559, 3)


def test_multiplicative_quantile_regression_90(is_plot, prepare_data, features, feature_properties):
    X, y = prepare_data
    y = abs(y)

    quantile = 0.9
    CB_est = cb_multiplicative_quantile_regressor_model(
        quantile=quantile, features=features, feature_properties=feature_properties
    )
    CB_est.fit(X.copy(), y)

    if is_plot:
        plot_CB("analysis_CB_iterfirst", [CB_est[-1].observers[0]], CB_est[-2])
        plot_CB("analysis_CB_iterlast", [CB_est[-1].observers[-1]], CB_est[-2])

    yhat = CB_est.predict(X.copy())

    quantile_acc = evaluate_quantile(y, yhat)
    np.testing.assert_almost_equal(quantile_acc, 0.9015, 3)


def test_additive_quantile_regression_median(is_plot, prepare_data, default_features, feature_properties):
    X, y = prepare_data
    X = X[default_features]

    CB_est = pipeline_CBAdditiveQuantileRegressor(
        feature_properties=feature_properties,
        quantile=0.5,
    )
    CB_est.fit(X.copy(), y)

    if is_plot:
        plot_CB("analysis_CB_iterlast", [CB_est[-1].observers[-1]], CB_est[-2])

    yhat = CB_est.predict(X.copy())

    quantile_acc = evaluate_quantile(y, yhat)
    np.testing.assert_almost_equal(quantile_acc, 0.4973, 3)

    mad = np.nanmean(np.abs(y - yhat))
    np.testing.assert_almost_equal(mad, 1.6990, 3)


def test_additive_quantile_regression_90(is_plot, prepare_data, default_features, feature_properties):
    X, y = prepare_data
    X = X[default_features]

    CB_est = pipeline_CBAdditiveQuantileRegressor(
        feature_properties=feature_properties,
        quantile=0.9,
    )
    CB_est.fit(X.copy(), y)

    if is_plot:
        plot_CB("analysis_CB_iterlast", [CB_est[-1].observers[-1]], CB_est[-2])

    yhat = CB_est.predict(X.copy())

    quantile_acc = evaluate_quantile(y, yhat)
    np.testing.assert_almost_equal(quantile_acc, 0.8969, 3)


def test_additive_regression_mad(is_plot, prepare_data, default_features, feature_properties):
    X, y = prepare_data
    X = X[default_features]

    CB_est = pipeline_CBAdditiveGenericCRegressor(
        feature_properties=feature_properties,
        costs=costs_mad,
    )
    CB_est.fit(X.copy(), y)

    if is_plot:
        plot_CB("analysis_CB_iterlast", [CB_est[-1].observers[-1]], CB_est[-2])

    yhat = CB_est.predict(X.copy())

    mad = np.nanmean(np.abs(y - yhat))
    np.testing.assert_almost_equal(mad, 1.6990, 3)


def test_additive_regression_mse(is_plot, prepare_data, default_features, feature_properties):
    X, y = prepare_data
    X = X[default_features]

    CB_est = pipeline_CBAdditiveGenericCRegressor(
        feature_properties=feature_properties,
        costs=costs_mse,
    )
    CB_est.fit(X.copy(), y)

    if is_plot:
        plot_CB("analysis_CB_iterlast", [CB_est[-1].observers[-1]], CB_est[-2])

    yhat = CB_est.predict(X.copy())

    mad = np.nanmean(np.abs(y - yhat))
    np.testing.assert_almost_equal(mad, 1.7480, 3)


def test_multiplicative_regression_mad(is_plot, prepare_data, default_features, feature_properties):
    X, y = prepare_data
    y = abs(y)

    X = X[default_features]

    CB_est = pipeline_CBMultiplicativeGenericCRegressor(
        feature_properties=feature_properties,
        costs=costs_mad,
    )
    CB_est.fit(X.copy(), y)

    if is_plot:
        plot_CB("analysis_CB_iterlast", [CB_est[-1].observers[-1]], CB_est[-2])

    yhat = CB_est.predict(X.copy())

    mad = np.nanmean(np.abs(y - yhat))
    np.testing.assert_almost_equal(mad, 1.6705, 3)


def test_multiplicative_regression_mse(is_plot, prepare_data, default_features, feature_properties):
    X, y = prepare_data
    y = abs(y)

    X = X[default_features]

    CB_est = pipeline_CBMultiplicativeGenericCRegressor(
        feature_properties=feature_properties,
        costs=costs_mse,
    )
    CB_est.fit(X.copy(), y)

    if is_plot:
        plot_CB("analysis_CB_iterlast", [CB_est[-1].observers[-1]], CB_est[-2])

    yhat = CB_est.predict(X.copy())

    mad = np.nanmean(np.abs(y - yhat))
    np.testing.assert_almost_equal(mad, 1.7171, 3)


def poisson_likelihood(prediction, y, weights):
    negative_log_likelihood = np.nanmean(prediction + np.log(factorial(y)) - np.log(prediction) * y)
    return negative_log_likelihood


@pytest.mark.skip(reason="Long running time")
def test_multiplicative_regression_likelihood(is_plot, prepare_data, default_features, feature_properties):
    X, y = prepare_data
    X = X[default_features]

    CB_est = pipeline_CBMultiplicativeGenericCRegressor(
        feature_properties=feature_properties,
        costs=poisson_likelihood,
    )
    CB_est.fit(X.copy(), y)

    if is_plot:
        plot_CB("analysis_CB_iterlast", [CB_est[-1].observers[-1]], CB_est[-2])

    yhat = CB_est.predict(X.copy())

    mad = np.nanmean(np.abs(y - yhat))
    np.testing.assert_almost_equal(mad, 1.9310, 3)


def costs_logloss(prediction, y, weights):
    prediction = np.where(prediction < 0.001, 0.001, prediction)
    prediction = np.where(prediction > 0.999, 0.999, prediction)
    return -np.nanmean(y * np.log(prediction) + (1 - y) * np.log(1 - prediction))


@pytest.fixture(scope="function")
def cb_classifier_logloss_model(features, feature_properties):
    explicit_smoothers = {
        ("dayofyear",): SeasonalSmoother(order=3),
        ("price_ratio",): IsotonicRegressor(increasing=False),
    }

    plobs = [observers.PlottingObserver(iteration=-1)]

    CB_pipeline = pipeline_CBGenericClassifier(
        feature_properties=feature_properties,
        feature_groups=features,
        observers=plobs,
        maximal_iterations=50,
        smoother_choice=common_smoothers.SmootherChoiceGroupBy(
            use_regression_type=True, use_normalization=False, explicit_smoothers=explicit_smoothers
        ),
        costs=costs_logloss,
    )

    return CB_pipeline


def test_classification_logloss(is_plot, prepare_data, cb_classifier_logloss_model):
    X, y = prepare_data
    y = y >= 3

    CB_est = cb_classifier_logloss_model
    CB_est.fit(X.copy(), y)

    if is_plot:
        plot_CB("analysis_CB_iterlast", [CB_est[-1].observers[-1]], CB_est[-2])

    yhat = CB_est.predict(X.copy())

    mad = np.nanmean(np.abs(y - yhat))
    np.testing.assert_almost_equal(mad, 0.4044, 3)
