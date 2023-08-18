import pandas as pd
import numpy as np
import pytest

from sklearn.preprocessing import OrdinalEncoder
from scipy.special import factorial

from cyclic_boosting import flags, common_smoothers, observers
from cyclic_boosting.smoothing.onedim import SeasonalSmoother, IsotonicRegressor
from cyclic_boosting.plots import plot_analysis
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


def plot_CB(filename, plobs, binner):
    for i, p in enumerate(plobs):
        plot_analysis(plot_observer=p, file_obj=filename + "_{}".format(i), use_tightlayout=False, binners=[binner])


def prepare_data(df):
    df["DATE"] = pd.to_datetime(df["DATE"])
    df["dayofweek"] = df["DATE"].dt.dayofweek
    df["dayofyear"] = df["DATE"].dt.dayofyear

    df["price_ratio"] = df["SALES_PRICE"] / df["NORMAL_PRICE"]
    df["price_ratio"].fillna(1, inplace=True)
    df["price_ratio"].clip(0, 1, inplace=True)
    df.loc[df["price_ratio"] == 1.0, "price_ratio"] = np.nan

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
    df[["L_ID", "P_ID", "PG_ID_3"]] = enc.fit_transform(df[["L_ID", "P_ID", "PG_ID_3"]])

    y = np.asarray(df["SALES"])
    X = df.drop(columns="SALES")
    return X, y


def feature_properties():
    fp = {}
    fp["P_ID"] = flags.IS_UNORDERED
    fp["PG_ID_3"] = flags.IS_UNORDERED
    fp["L_ID"] = flags.IS_UNORDERED
    fp["dayofweek"] = flags.IS_ORDERED
    fp["dayofyear"] = flags.IS_CONTINUOUS | flags.IS_LINEAR
    fp["price_ratio"] = flags.IS_CONTINUOUS | flags.HAS_MISSING | flags.MISSING_NOT_LEARNED
    fp["PROMOTION_TYPE"] = flags.IS_ORDERED
    return fp


def get_features():
    features = [
        "dayofweek",
        "L_ID",
        "PG_ID_3",
        "P_ID",
        "PROMOTION_TYPE",
        "price_ratio",
        "dayofyear",
        ("P_ID", "L_ID"),
    ]
    return features


def cb_poisson_regressor_model():
    features = get_features()

    fp = feature_properties()
    explicit_smoothers = {
        ("dayofyear",): SeasonalSmoother(order=3),
        ("price_ratio",): IsotonicRegressor(increasing=False),
    }

    plobs = [
        observers.PlottingObserver(iteration=1),
        observers.PlottingObserver(iteration=-1),
    ]

    CB_pipeline = pipeline_CBPoissonRegressor(
        feature_properties=fp,
        feature_groups=features,
        observers=plobs,
        maximal_iterations=50,
        smoother_choice=common_smoothers.SmootherChoiceGroupBy(
            use_regression_type=True, use_normalization=False, explicit_smoothers=explicit_smoothers
        ),
    )

    return CB_pipeline


def test_poisson_regression():
    np.random.seed(42)

    df = pd.read_csv("./tests/integration_test_data.csv")

    X, y = prepare_data(df)

    CB_est = cb_poisson_regressor_model()
    CB_est.fit(X.copy(), y)
    # plot_CB('analysis_CB_iterfirst',
    #         [CB_est[-1].observers[0]], CB_est[-2])
    # plot_CB('analysis_CB_iterlast',
    #         [CB_est[-1].observers[-1]], CB_est[-2])

    yhat = CB_est.predict(X.copy())

    mad = np.nanmean(np.abs(y - yhat))
    np.testing.assert_almost_equal(mad, 1.6997, 3)


def test_poisson_regression_default_features():
    np.random.seed(42)

    df = pd.read_csv("./tests/integration_test_data.csv")

    X, y = prepare_data(df)
    X = X[["dayofweek", "L_ID", "PG_ID_3", "P_ID", "PROMOTION_TYPE", "price_ratio", "dayofyear"]]

    fp = feature_properties()
    CB_est = pipeline_CBPoissonRegressor(feature_properties=fp)
    CB_est.fit(X.copy(), y)

    yhat = CB_est.predict(X.copy())

    mad = np.nanmean(np.abs(y - yhat))
    np.testing.assert_almost_equal(mad, 1.7185, 3)


@pytest.mark.parametrize(("feature_groups", "expected"), [(None, 1.689), ([0, 1, 4, 5], 1.950)])
def test_poisson_regression_ndarray(feature_groups, expected):
    np.random.seed(42)

    df = pd.read_csv("./tests/integration_test_data.csv")

    X, y = prepare_data(df)
    X = X[["dayofweek", "L_ID", "PG_ID_3", "P_ID", "PROMOTION_TYPE", "price_ratio", "dayofyear"]].to_numpy()

    fp = feature_properties()
    CB_est = pipeline_CBPoissonRegressor(feature_groups=feature_groups, feature_properties=fp)
    CB_est.fit(X.copy(), y)

    yhat = CB_est.predict(X.copy())

    mad = np.nanmean(np.abs(y - yhat))
    np.testing.assert_almost_equal(mad, expected, 3)


@pytest.mark.parametrize("regressor", ["BinomRegressor", "PoissonRegressor"])
def test_regression_ndarray_w_feature_properties(regressor):
    np.random.seed(42)

    df = pd.read_csv("./tests/integration_test_data.csv")

    X, y = prepare_data(df)
    X = X[["dayofweek", "L_ID", "PG_ID_3", "P_ID", "PROMOTION_TYPE", "price_ratio", "dayofyear"]].to_numpy()

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


def test_poisson_regression_default_features_and_properties():
    np.random.seed(42)

    df = pd.read_csv("./tests/integration_test_data.csv")

    X, y = prepare_data(df)
    X = X[["dayofweek", "L_ID", "PG_ID_3", "P_ID", "PROMOTION_TYPE", "price_ratio", "dayofyear"]]

    plobs = [
        observers.PlottingObserver(iteration=1),
        observers.PlottingObserver(iteration=-1),
    ]
    CB_est = pipeline_CBPoissonRegressor(
        observers=plobs,
    )
    CB_est.fit(X.copy(), y)
    # plot_CB('analysis_CB_iterfirst',
    #         [CB_est[-1].observers[0]], CB_est[-2])
    # plot_CB('analysis_CB_iterlast',
    #         [CB_est[-1].observers[-1]], CB_est[-2])

    yhat = CB_est.predict(X.copy())

    mad = np.nanmean(np.abs(y - yhat))
    np.testing.assert_almost_equal(mad, 1.6982, 3)


def test_poisson_regression_default_features_notaggregated():
    np.random.seed(42)

    df = pd.read_csv("./tests/integration_test_data.csv")

    X, y = prepare_data(df)
    X = X[["dayofweek", "L_ID", "PG_ID_3", "P_ID", "PROMOTION_TYPE", "price_ratio", "dayofyear"]]

    fp = feature_properties()
    CB_est = pipeline_CBPoissonRegressor(feature_properties=fp, aggregate=False)
    CB_est.fit(X.copy(), y)

    yhat = CB_est.predict(X.copy())

    mad = np.nanmean(np.abs(y - yhat))
    np.testing.assert_almost_equal(mad, 1.7144, 3)


def test_nbinom_regression_default_features():
    np.random.seed(42)

    df = pd.read_csv("./tests/integration_test_data.csv")

    X, y = prepare_data(df)
    X = X[["dayofweek", "L_ID", "PG_ID_3", "P_ID", "PROMOTION_TYPE", "price_ratio", "dayofyear"]]

    fp = feature_properties()
    CB_est = pipeline_CBNBinomRegressor(
        feature_properties=fp,
        a=1.2,
        c=0.1,
    )
    CB_est.fit(X.copy(), y)

    yhat = CB_est.predict(X.copy())

    mad = np.nanmean(np.abs(y - yhat))
    np.testing.assert_almost_equal(mad, 1.7198, 3)


@pytest.mark.parametrize(("feature_groups", "expected"), [(None, 1.689), ([0, 1, 4, 5], 1.950)])
def test_nbinom_regression_ndarray(feature_groups, expected):
    np.random.seed(42)

    df = pd.read_csv("./tests/integration_test_data.csv")

    X, y = prepare_data(df)
    X = X[["dayofweek", "L_ID", "PG_ID_3", "P_ID", "PROMOTION_TYPE", "price_ratio", "dayofyear"]].to_numpy()

    fp = feature_properties()
    CB_est = pipeline_CBNBinomRegressor(
        feature_groups=feature_groups,
        feature_properties=fp,
    )
    CB_est.fit(X.copy(), y)

    yhat = CB_est.predict(X.copy())

    mad = np.nanmean(np.abs(y - yhat))
    np.testing.assert_almost_equal(mad, expected, 3)


def cb_exponential_regressor_model():
    features = get_features()
    features.remove("price_ratio")
    price_features = [
        "L_ID",
        "PG_ID_1",
        "PG_ID_2",
        "PG_ID_3",
        "P_ID",
        "dayofweek",
    ]

    fp = feature_properties()
    explicit_smoothers = {
        ("dayofyear",): SeasonalSmoother(order=3),
        ("price_ratio",): IsotonicRegressor(increasing=False),
    }

    plobs = [
        observers.PlottingObserver(iteration=1),
        observers.PlottingObserver(iteration=-1),
    ]

    CB_pipeline = pipeline_CBExponential(
        feature_properties=fp,
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


def test_exponential_regression():
    np.random.seed(42)

    df = pd.read_csv("./tests/integration_test_data.csv")

    X, y = prepare_data(df)
    X.loc[df["price_ratio"] == np.nan, "price_ratio"] = 1.0

    CB_est = cb_exponential_regressor_model()
    CB_est.fit(X.copy(), y)
    # plot_CB('analysis_CB_iterfirst',
    #         [CB_est[-1].observers[0]], CB_est[-2])
    # plot_CB('analysis_CB_iterlast',
    #         [CB_est[-1].observers[-1]], CB_est[-2])

    yhat = CB_est.predict(X.copy())

    mad = np.nanmean(np.abs(y - yhat))
    np.testing.assert_almost_equal(mad, 1.7203, 3)


def cb_classifier_model():
    features = get_features()

    fp = feature_properties()
    explicit_smoothers = {
        ("dayofyear",): SeasonalSmoother(order=3),
        ("price_ratio",): IsotonicRegressor(increasing=False),
    }

    plobs = [observers.PlottingObserver(iteration=-1)]

    CB_pipeline = pipeline_CBClassifier(
        feature_properties=fp,
        feature_groups=features,
        observers=plobs,
        maximal_iterations=50,
        smoother_choice=common_smoothers.SmootherChoiceGroupBy(
            use_regression_type=True, use_normalization=False, explicit_smoothers=explicit_smoothers
        ),
    )

    return CB_pipeline


def test_classification():
    np.random.seed(42)

    df = pd.read_csv("./tests/integration_test_data.csv")

    X, y = prepare_data(df)
    y = y >= 3

    CB_est = cb_classifier_model()
    CB_est.fit(X.copy(), y)
    # plot_CB('analysis_CB_iterlast',
    #         [CB_est[-1].observers[-1]], CB_est[-2])

    yhat = CB_est.predict(X.copy())

    mad = np.nanmean(np.abs(y - yhat))
    np.testing.assert_almost_equal(mad, 0.3075, 3)


def test_location_regression_default_features():
    np.random.seed(42)

    df = pd.read_csv("./tests/integration_test_data.csv")

    X, y = prepare_data(df)
    X = X[["dayofweek", "L_ID", "PG_ID_3", "P_ID", "PROMOTION_TYPE", "price_ratio", "dayofyear"]]

    fp = feature_properties()
    # plobs = [
    #     observers.PlottingObserver(iteration=-1)
    # ]
    CB_est = pipeline_CBLocationRegressor(
        # observers=plobs,
        feature_properties=fp
    )
    CB_est.fit(X.copy(), y)
    # plot_CB('analysis_CB_iterlast',
    #         [CB_est[-1].observers[-1]], CB_est[-2])

    yhat = CB_est.predict(X.copy())

    mad = np.nanmean(np.abs(y - yhat))
    np.testing.assert_almost_equal(mad, 1.7511, 3)


def cb_width_model():
    features = ["dayofweek", "L_ID", "PG_ID_3", "PROMOTION_TYPE"]

    fp = feature_properties()
    explicit_smoothers = {}

    plobs = [observers.PlottingObserver(iteration=-1)]

    CB_pipeline = pipeline_CBNBinomC(
        mean_prediction_column="yhat_mean",
        feature_properties=fp,
        feature_groups=features,
        observers=plobs,
        maximal_iterations=50,
        smoother_choice=common_smoothers.SmootherChoiceGroupBy(
            use_regression_type=True, use_normalization=False, explicit_smoothers=explicit_smoothers
        ),
    )

    return CB_pipeline


def test_width_regression_default_features():
    np.random.seed(42)

    df = pd.read_csv("./tests/integration_test_data.csv")

    X, y = prepare_data(df)
    X = X[["dayofweek", "L_ID", "PG_ID_3", "P_ID", "PROMOTION_TYPE", "price_ratio", "dayofyear"]]

    fp = feature_properties()
    CB_est = pipeline_CBPoissonRegressor(feature_properties=fp)
    CB_est.fit(X.copy(), y)
    yhat = CB_est.predict(X.copy())
    X["yhat_mean"] = yhat

    CB_est_width = cb_width_model()
    CB_est_width.fit(X.copy(), y)
    c = CB_est_width.predict(X.copy())
    np.testing.assert_almost_equal(c.mean(), 0.365, 3)


def test_GBS_regression_default_features():
    np.random.seed(42)

    df = pd.read_csv("./tests/integration_test_data.csv")

    X, y = prepare_data(df)
    X = X[["dayofweek", "L_ID", "PG_ID_3", "P_ID", "PROMOTION_TYPE", "price_ratio", "dayofyear"]]

    y[1000:10000] = -y[1000:10000]

    fp = feature_properties()
    # plobs = [
    #     observers.PlottingObserver(iteration=-1)
    # ]
    CB_est = pipeline_CBGBSRegressor(
        # observers=plobs,
        feature_properties=fp
    )
    CB_est.fit(X.copy(), y)
    # plot_CB('analysis_CB_iterlast',
    #         [CB_est[-1].observers[-1]], CB_est[-2])

    yhat = CB_est.predict(X.copy())

    mad = np.nanmean(np.abs(y - yhat))
    np.testing.assert_almost_equal(mad, 2.5755, 3)


def evaluate_quantile(y, yhat):
    quantile_acc = (y <= yhat).mean()
    return quantile_acc


def cb_multiplicative_quantile_regressor_model(quantile):
    features = get_features()

    fp = feature_properties()
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
        feature_properties=fp,
        feature_groups=features,
        observers=plobs,
        maximal_iterations=50,
        smoother_choice=common_smoothers.SmootherChoiceGroupBy(
            use_regression_type=True, use_normalization=False, explicit_smoothers=explicit_smoothers
        ),
    )

    return CB_pipeline


def test_multiplicative_quantile_regression_median():
    np.random.seed(42)

    df = pd.read_csv("./tests/integration_test_data.csv")

    X, y = prepare_data(df)

    quantile = 0.5
    CB_est = cb_multiplicative_quantile_regressor_model(quantile)
    CB_est.fit(X.copy(), y)
    # plot_CB('analysis_CB_iterfirst',
    #         [CB_est[-1].observers[0]], CB_est[-2])
    # plot_CB('analysis_CB_iterlast',
    #         [CB_est[-1].observers[-1]], CB_est[-2])

    yhat = CB_est.predict(X.copy())

    quantile_acc = evaluate_quantile(y, yhat)
    np.testing.assert_almost_equal(quantile_acc, 0.5043, 3)

    mad = np.nanmean(np.abs(y - yhat))
    np.testing.assert_almost_equal(mad, 1.6559, 3)


def test_multiplicative_quantile_regression_90():
    np.random.seed(42)

    df = pd.read_csv("./tests/integration_test_data.csv")

    X, y = prepare_data(df)

    quantile = 0.9
    CB_est = cb_multiplicative_quantile_regressor_model(quantile)
    CB_est.fit(X.copy(), y)
    # plot_CB('analysis_CB_iterfirst',
    #         [CB_est[-1].observers[0]], CB_est[-2])
    # plot_CB('analysis_CB_iterlast',
    #         [CB_est[-1].observers[-1]], CB_est[-2])

    yhat = CB_est.predict(X.copy())

    quantile_acc = evaluate_quantile(y, yhat)
    np.testing.assert_almost_equal(quantile_acc, 0.9015, 3)


def test_additive_quantile_regression_median():
    np.random.seed(42)

    df = pd.read_csv("./tests/integration_test_data.csv")

    X, y = prepare_data(df)
    X = X[["dayofweek", "L_ID", "PG_ID_3", "P_ID", "PROMOTION_TYPE", "price_ratio", "dayofyear"]]

    fp = feature_properties()
    # plobs = [
    #     observers.PlottingObserver(iteration=-1)
    # ]
    CB_est = pipeline_CBAdditiveQuantileRegressor(
        # observers=plobs,
        feature_properties=fp,
        quantile=0.5,
    )
    CB_est.fit(X.copy(), y)
    # plot_CB('analysis_CB_iterlast',
    #         [CB_est[-1].observers[-1]], CB_est[-2])

    yhat = CB_est.predict(X.copy())

    quantile_acc = evaluate_quantile(y, yhat)
    np.testing.assert_almost_equal(quantile_acc, 0.4973, 3)

    mad = np.nanmean(np.abs(y - yhat))
    np.testing.assert_almost_equal(mad, 1.6990, 3)


def test_additive_quantile_regression_90():
    np.random.seed(42)

    df = pd.read_csv("./tests/integration_test_data.csv")

    X, y = prepare_data(df)
    X = X[["dayofweek", "L_ID", "PG_ID_3", "P_ID", "PROMOTION_TYPE", "price_ratio", "dayofyear"]]

    fp = feature_properties()
    # plobs = [
    #     observers.PlottingObserver(iteration=-1)
    # ]
    CB_est = pipeline_CBAdditiveQuantileRegressor(
        # observers=plobs,
        feature_properties=fp,
        quantile=0.9,
    )
    CB_est.fit(X.copy(), y)
    # plot_CB('analysis_CB_iterlast',
    #         [CB_est[-1].observers[-1]], CB_est[-2])

    yhat = CB_est.predict(X.copy())

    quantile_acc = evaluate_quantile(y, yhat)
    np.testing.assert_almost_equal(quantile_acc, 0.8969, 3)


def costs_mad(prediction, y, weights):
    return np.nanmean(np.abs(y - prediction))


def costs_mse(prediction, y, weights):
    return np.nanmean(np.square(y - prediction))


def test_additive_regression_mad():
    np.random.seed(42)

    df = pd.read_csv("./tests/integration_test_data.csv")

    X, y = prepare_data(df)
    X = X[["dayofweek", "L_ID", "PG_ID_3", "P_ID", "PROMOTION_TYPE", "price_ratio", "dayofyear"]]

    fp = feature_properties()
    # plobs = [
    #     observers.PlottingObserver(iteration=-1)
    # ]
    CB_est = pipeline_CBAdditiveGenericCRegressor(
        # observers=plobs,
        feature_properties=fp,
        costs=costs_mad,
    )
    CB_est.fit(X.copy(), y)
    # plot_CB('analysis_CB_iterlast',
    #         [CB_est[-1].observers[-1]], CB_est[-2])

    yhat = CB_est.predict(X.copy())

    mad = np.nanmean(np.abs(y - yhat))
    np.testing.assert_almost_equal(mad, 1.6990, 3)


def test_additive_regression_mse():
    np.random.seed(42)

    df = pd.read_csv("./tests/integration_test_data.csv")

    X, y = prepare_data(df)
    X = X[["dayofweek", "L_ID", "PG_ID_3", "P_ID", "PROMOTION_TYPE", "price_ratio", "dayofyear"]]

    fp = feature_properties()
    # plobs = [
    #     observers.PlottingObserver(iteration=-1)
    # ]
    CB_est = pipeline_CBAdditiveGenericCRegressor(
        # observers=plobs,
        feature_properties=fp,
        costs=costs_mse,
    )
    CB_est.fit(X.copy(), y)
    # plot_CB('analysis_CB_iterlast',
    #         [CB_est[-1].observers[-1]], CB_est[-2])

    yhat = CB_est.predict(X.copy())

    mad = np.nanmean(np.abs(y - yhat))
    np.testing.assert_almost_equal(mad, 1.7480, 3)


def test_multiplicative_regression_mad():
    np.random.seed(42)

    df = pd.read_csv("./tests/integration_test_data.csv")

    X, y = prepare_data(df)
    X = X[["dayofweek", "L_ID", "PG_ID_3", "P_ID", "PROMOTION_TYPE", "price_ratio", "dayofyear"]]

    fp = feature_properties()
    # plobs = [
    #     observers.PlottingObserver(iteration=-1)
    # ]
    CB_est = pipeline_CBMultiplicativeGenericCRegressor(
        # observers=plobs,
        feature_properties=fp,
        costs=costs_mad,
    )
    CB_est.fit(X.copy(), y)
    # plot_CB('analysis_CB_iterlast',
    #         [CB_est[-1].observers[-1]], CB_est[-2])

    yhat = CB_est.predict(X.copy())

    mad = np.nanmean(np.abs(y - yhat))
    np.testing.assert_almost_equal(mad, 1.6705, 3)


def test_multiplicative_regression_mse():
    np.random.seed(42)

    df = pd.read_csv("./tests/integration_test_data.csv")

    X, y = prepare_data(df)
    X = X[["dayofweek", "L_ID", "PG_ID_3", "P_ID", "PROMOTION_TYPE", "price_ratio", "dayofyear"]]

    fp = feature_properties()
    # plobs = [
    #     observers.PlottingObserver(iteration=-1)
    # ]
    CB_est = pipeline_CBMultiplicativeGenericCRegressor(
        # observers=plobs,
        feature_properties=fp,
        costs=costs_mse,
    )
    CB_est.fit(X.copy(), y)
    # plot_CB('analysis_CB_iterlast',
    #         [CB_est[-1].observers[-1]], CB_est[-2])

    yhat = CB_est.predict(X.copy())

    mad = np.nanmean(np.abs(y - yhat))
    np.testing.assert_almost_equal(mad, 1.7171, 3)


def poisson_likelihood(prediction, y, weights):
    negative_log_likelihood = np.nanmean(prediction + np.log(factorial(y)) - np.log(prediction) * y)
    return negative_log_likelihood


# commented out due to rather long runtime
# def test_multiplicative_regression_likelihood():
#     np.random.seed(42)

#     df = pd.read_csv("./tests/integration_test_data.csv")

#     X, y = prepare_data(df)
#     X = X[["dayofweek", "L_ID", "PG_ID_3", "P_ID", "PROMOTION_TYPE", "price_ratio", "dayofyear"]]

#     fp = feature_properties()
#     # plobs = [
#     #     observers.PlottingObserver(iteration=-1)
#     # ]
#     CB_est = pipeline_CBMultiplicativeGenericCRegressor(
#         # observers=plobs,
#         feature_properties=fp,
#         costs=poisson_likelihood,
#     )
#     CB_est.fit(X.copy(), y)
#     # plot_CB('analysis_CB_iterlast',
#     #         [CB_est[-1].observers[-1]], CB_est[-2])

#     yhat = CB_est.predict(X.copy())

#     mad = np.nanmean(np.abs(y - yhat))
#     np.testing.assert_almost_equal(mad, 1.9310, 3)


def costs_logloss(prediction, y, weights):
    prediction = np.where(prediction < 0.001, 0.001, prediction)
    prediction = np.where(prediction > 0.999, 0.999, prediction)
    return -np.nanmean(y * np.log(prediction) + (1 - y) * np.log(1 - prediction))


def cb_classifier_logloss_model():
    features = get_features()

    fp = feature_properties()
    explicit_smoothers = {
        ("dayofyear",): SeasonalSmoother(order=3),
        ("price_ratio",): IsotonicRegressor(increasing=False),
    }

    plobs = [observers.PlottingObserver(iteration=-1)]

    CB_pipeline = pipeline_CBGenericClassifier(
        feature_properties=fp,
        feature_groups=features,
        observers=plobs,
        maximal_iterations=50,
        smoother_choice=common_smoothers.SmootherChoiceGroupBy(
            use_regression_type=True, use_normalization=False, explicit_smoothers=explicit_smoothers
        ),
        costs=costs_logloss,
    )

    return CB_pipeline


def test_classification_logloss():
    np.random.seed(42)

    df = pd.read_csv("./tests/integration_test_data.csv")

    X, y = prepare_data(df)
    y = y >= 3

    CB_est = cb_classifier_logloss_model()
    CB_est.fit(X.copy(), y)
    # plot_CB('analysis_CB_iterlast',
    #         [CB_est[-1].observers[-1]], CB_est[-2])

    yhat = CB_est.predict(X.copy())

    mad = np.nanmean(np.abs(y - yhat))
    np.testing.assert_almost_equal(mad, 0.4044, 3)
