import pandas as pd
import numpy as np

from sklearn.preprocessing import OrdinalEncoder

from cyclic_boosting import flags, common_smoothers, observers
from cyclic_boosting.smoothing.onedim import SeasonalSmoother,\
    IsotonicRegressor
from cyclic_boosting.plots import plot_analysis
from cyclic_boosting.pipelines import pipeline_CBPoissonRegressor, \
    pipeline_CBClassifier, pipeline_CBLocationRegressor, pipeline_CBExponential, \
    pipeline_CBNBinomRegressor, pipeline_CBNBinomC, pipeline_CBGBSRegressor


def plot_CB(filename, plobs, binner):
    for i, p in enumerate(plobs):
        plot_analysis(
            plot_observer=p,
            file_obj=filename + "_{}".format(i),
            use_tightlayout=False,
            binners=[binner]
        )


def prepare_data(df):
    df['DATE'] = pd.to_datetime(df['DATE'])
    df['dayofweek'] = df['DATE'].dt.dayofweek
    df['dayofyear'] = df['DATE'].dt.dayofyear

    df['price_ratio'] = df['SALES_PRICE'] / df['NORMAL_PRICE']
    df['price_ratio'].fillna(1, inplace=True)
    df['price_ratio'].clip(0, 1, inplace=True)
    df.loc[df['price_ratio'] == 1., 'price_ratio'] = np.nan

    enc = OrdinalEncoder(
        handle_unknown='use_encoded_value', unknown_value=np.nan)
    df[['L_ID', 'P_ID', 'PG_ID_3']] = enc.fit_transform(
        df[['L_ID', 'P_ID', 'PG_ID_3']])

    y = np.asarray(df['SALES'])
    X = df.drop(columns='SALES')
    return X, y


def feature_properties():
    fp = {}
    fp['P_ID'] = flags.IS_UNORDERED
    fp['PG_ID_3'] = flags.IS_UNORDERED
    fp['L_ID'] = flags.IS_UNORDERED
    fp['dayofweek'] = flags.IS_ORDERED
    fp['dayofyear'] = flags.IS_CONTINUOUS | flags.IS_LINEAR
    fp['price_ratio'] = \
        flags.IS_CONTINUOUS | flags.HAS_MISSING | flags.MISSING_NOT_LEARNED
    fp['PROMOTION_TYPE'] = flags.IS_ORDERED
    return fp


def get_features():
    features = [
        'dayofweek',
        'L_ID',
        'PG_ID_3',
        'P_ID',
        'PROMOTION_TYPE',
        'price_ratio',
        'dayofyear',
        ('P_ID', 'L_ID'),
    ]
    return features


def cb_poisson_regressor_model():
    features = get_features()

    fp = feature_properties()
    explicit_smoothers = {('dayofyear',): SeasonalSmoother(order=3),
                          ('price_ratio',): IsotonicRegressor(increasing=False),
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
            use_regression_type=True,
            use_normalization=False,
            explicit_smoothers=explicit_smoothers),
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
    X = X[[
        'dayofweek',
        'L_ID',
        'PG_ID_3',
        'P_ID',
        'PROMOTION_TYPE',
        'price_ratio',
        'dayofyear'
    ]]

    fp = feature_properties()
    CB_est = pipeline_CBPoissonRegressor(
        feature_properties=fp
    )
    CB_est.fit(X.copy(), y)

    yhat = CB_est.predict(X.copy())

    mad = np.nanmean(np.abs(y - yhat))
    np.testing.assert_almost_equal(mad, 1.7185, 3)


def test_poisson_regression_default_features_notaggregated():
    np.random.seed(42)

    df = pd.read_csv("./tests/integration_test_data.csv")

    X, y = prepare_data(df)
    X = X[[
        'dayofweek',
        'L_ID',
        'PG_ID_3',
        'P_ID',
        'PROMOTION_TYPE',
        'price_ratio',
        'dayofyear'
    ]]

    fp = feature_properties()
    CB_est = pipeline_CBPoissonRegressor(
        feature_properties=fp,
        aggregate=False
    )
    CB_est.fit(X.copy(), y)

    yhat = CB_est.predict(X.copy())

    mad = np.nanmean(np.abs(y - yhat))
    np.testing.assert_almost_equal(mad, 1.7144, 3)


def test_nbinom_regression_default_features():
    np.random.seed(42)

    df = pd.read_csv("./tests/integration_test_data.csv")

    X, y = prepare_data(df)
    X = X[[
        'dayofweek',
        'L_ID',
        'PG_ID_3',
        'P_ID',
        'PROMOTION_TYPE',
        'price_ratio',
        'dayofyear'
    ]]

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


def cb_exponential_regressor_model():
    features = get_features()
    features.remove('price_ratio')
    price_features = [
        'L_ID',
        'PG_ID_1',
        'PG_ID_2',
        'PG_ID_3',
        'P_ID',
        'dayofweek',
    ]

    fp = feature_properties()
    explicit_smoothers = {('dayofyear',): SeasonalSmoother(order=3),
                          ('price_ratio',): IsotonicRegressor(increasing=False),
                         }

    plobs = [
        observers.PlottingObserver(iteration=1),
        observers.PlottingObserver(iteration=-1),
    ]

    CB_pipeline = pipeline_CBExponential(
        feature_properties=fp,
        standard_feature_groups=features,
        external_feature_groups=price_features,
        external_colname='price_ratio',
        observers=plobs,
        maximal_iterations=50,
        smoother_choice=common_smoothers.SmootherChoiceGroupBy(
            use_regression_type=True,
            use_normalization=False,
            explicit_smoothers=explicit_smoothers),
    )

    return CB_pipeline


def test_exponential_regression():
    np.random.seed(42)

    df = pd.read_csv("./tests/integration_test_data.csv")

    X, y = prepare_data(df)
    X.loc[df['price_ratio'] == np.nan, 'price_ratio'] = 1.

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
    explicit_smoothers = {('dayofyear',): SeasonalSmoother(order=3),
                          ('price_ratio',): IsotonicRegressor(increasing=False),
                         }

    plobs = [
        observers.PlottingObserver(iteration=-1)
    ]

    CB_pipeline = pipeline_CBClassifier(
        feature_properties=fp,
        feature_groups=features,
        observers=plobs,
        maximal_iterations=50,
        smoother_choice=common_smoothers.SmootherChoiceGroupBy(
            use_regression_type=True,
            use_normalization=False,
            explicit_smoothers=explicit_smoothers),
    )

    return CB_pipeline


def test_classification():
    np.random.seed(42)

    df = pd.read_csv("./tests/integration_test_data.csv")

    X, y = prepare_data(df)
    y = (y >= 3)

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
    X = X[[
        'dayofweek',
        'L_ID',
        'PG_ID_3',
        'P_ID',
        'PROMOTION_TYPE',
        'price_ratio',
        'dayofyear'
    ]]

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
    features = ['dayofweek', 'L_ID', 'PG_ID_3', 'PROMOTION_TYPE']

    fp = feature_properties()
    explicit_smoothers = {}

    plobs = [
        observers.PlottingObserver(iteration=-1)
    ]

    CB_pipeline = pipeline_CBNBinomC(
        mean_prediction_column='yhat_mean',
        feature_properties=fp,
        feature_groups=features,
        observers=plobs,
        maximal_iterations=50,
        smoother_choice=common_smoothers.SmootherChoiceGroupBy(
            use_regression_type=True,
            use_normalization=False,
            explicit_smoothers=explicit_smoothers),
    )

    return CB_pipeline


def test_width_regression_default_features():
    np.random.seed(42)

    df = pd.read_csv("./tests/integration_test_data.csv")

    X, y = prepare_data(df)
    X = X[[
        'dayofweek',
        'L_ID',
        'PG_ID_3',
        'P_ID',
        'PROMOTION_TYPE',
        'price_ratio',
        'dayofyear'
    ]]

    fp = feature_properties()
    CB_est = pipeline_CBPoissonRegressor(
        feature_properties=fp
    )
    CB_est.fit(X.copy(), y)
    yhat = CB_est.predict(X.copy())
    X['yhat_mean'] = yhat

    CB_est_width = cb_width_model()
    CB_est_width.fit(X.copy(), y)
    c = CB_est_width.predict(X.copy())
    np.testing.assert_almost_equal(c.mean(), 0.365, 3)


def test_GBS_regression_default_features():
    np.random.seed(42)

    df = pd.read_csv("./tests/integration_test_data.csv")

    X, y = prepare_data(df)
    X = X[[
        'dayofweek',
        'L_ID',
        'PG_ID_3',
        'P_ID',
        'PROMOTION_TYPE',
        'price_ratio',
        'dayofyear'
    ]]

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
