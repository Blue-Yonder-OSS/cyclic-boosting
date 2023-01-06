import pandas as pd
import numpy as np

from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline

from cyclic_boosting import binning, flags, CBPoissonRegressor,\
    common_smoothers, observers
from cyclic_boosting.smoothing.onedim import SeasonalSmoother,\
    IsotonicRegressor
from cyclic_boosting.plots import plot_analysis


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


def cb_model():
    fp = feature_properties()
    explicit_smoothers = {('dayofyear',): SeasonalSmoother(order=3),
                          ('price_ratio',): IsotonicRegressor(increasing=False),
                         }

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

    plobs = [
        observers.PlottingObserver(iteration=-1)
    ]

    est=CBPoissonRegressor(
        feature_properties=fp,
        feature_groups=features,
        observers=plobs,
        maximal_iterations=50,
        smoother_choice=common_smoothers.SmootherChoiceGroupBy(
            use_regression_type=True,
            use_normalization=False,
            explicit_smoothers=explicit_smoothers),
    )

    binner = binning.BinNumberTransformer(n_bins=100, feature_properties=fp)

    CB_est = Pipeline([("binning", binner), ("CB", est)])
    return CB_est


def test_poisson_regression():
    np.random.seed(42)

    df = pd.read_csv("./tests/integration_test_data.csv")

    X, y = prepare_data(df)

    CB_est = cb_model()
    CB_est.fit(X.copy(), y)
    # plot_CB('analysis_CB_iterlast',
    #         [CB_est[-1].observers[-1]], CB_est[-2])

    yhat = CB_est.predict(X.copy())

    mad = np.nanmean(np.abs(y - yhat))
    np.testing.assert_almost_equal(mad, 1.6997348846024691)
