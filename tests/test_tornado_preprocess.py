import warnings
from logging import WARNING, ERROR, getLogger


import numpy as np
import pandas as pd
import pytest
import os
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, \
                                  PowerTransformer, QuantileTransformer, \
                                  OrdinalEncoder, KBinsDiscretizer, \
                                  OneHotEncoder, TargetEncoder
from sklearn.feature_extraction import FeatureHasher
import statsmodels.api as sm

from cyclic_boosting.tornado.core import preprocess


# _logger = getLogger('preprocess')


def create_time_series_test_data(n_rows, n_cols, start_date=None):
    if start_date is None:
        start_date = datetime.datetime.today().date()
    else:
        start_date = pd.to_datetime(start_date).date()
    dates = pd.date_range(start=start_date, periods=n_rows, freq='D')
    df = create_non_time_series_test_data(n_rows, n_cols)
    df["date"] = dates

    return df


def create_non_time_series_test_data(n_rows, n_cols):
    data_int = np.random.randint(1, 1000, size=(n_rows, int(n_cols/2)))
    data_float = np.random.random(size=(n_rows, n_cols-int(n_cols/2)))
    data_target = np.random.random(size=(n_rows, 1))
    cols_int = [f"int_{col}" for col in range(int(n_cols/2))]
    cols_float = [f"float_{col}" for col in range(n_cols-int(n_cols/2))]
    df_int = pd.DataFrame(data_int, columns=cols_int)
    df_float = pd.DataFrame(data_float, columns=cols_float)
    df_target = pd.DataFrame(data_target, columns=["target"])
    df = pd.concat([df_int, df_float, df_target], axis=1)

    return df


def create_autocorrelated_data(n_rows, n_cols, lag, start_date=None, autocorrelation_coefficient=0.8):
    mean = 0
    std_deviation = 1
    data = [np.random.normal(mean, std_deviation)] * lag
    for _ in range(lag, n_rows):
        new_value = autocorrelation_coefficient * data[-lag] + np.random.normal(mean, std_deviation)
        data.append(new_value)
    df = create_time_series_test_data(n_rows, n_cols, start_date)
    df["target"] = data

    return df


def split_dataset(df):
    test_size = 0.2
    seed = 0
    train, valid = train_test_split(
        df,
        test_size=test_size,
        random_state=seed)
    return train, valid


def test_load_dataset():
    preprocessor = preprocess.Preprocess(opt=None)
    path = "./test_data.csv"

    # time_series data test
    test_data_time_series = create_time_series_test_data(10, 5)
    test_data_time_series.to_csv(path, index=False)
    data = preprocessor.load_dataset(path)
    data["date"] = pd.to_datetime(data["date"])
    os.remove(path)
    pd.testing.assert_frame_equal(data, test_data_time_series)

    # non time_series data test
    test_data_non_time_series = create_non_time_series_test_data(10, 5)
    test_data_non_time_series.to_csv(path, index=False)
    data = preprocessor.load_dataset(path)
    os.remove(path)
    pd.testing.assert_frame_equal(data, test_data_non_time_series)

    # excel data test
    path = "./test_data.xlsx"
    test_data_time_series = create_time_series_test_data(10, 5)
    test_data_time_series.to_excel(path, index=False)
    data = preprocessor.load_dataset(path)
    data["date"] = pd.to_datetime(data["date"])
    os.remove(path)
    pd.testing.assert_frame_equal(data, test_data_time_series)


def test_check_data():

    # time_series data test
    preprocessor = preprocess.Preprocess(opt={})
    desired = {
        "encode_category": {},
        "check_dtype": {},
        "check_cardinality": {},
        "todatetime": {},
        "lag": {},
        "rolling": {},
        "expanding": {},
        "dayofweek": {},
        "dayofyear": {},
        "standardization": {},
        "minmax": {},
        "logarithmic": {},
        "clipping": {},
        "binning": {},
        "rank": {},
        "rankgauss": {}
        }
    test_data_time_series = create_time_series_test_data(10, 5)
    preprocessor.check_data(test_data_time_series, is_time_series=True)
    preprocessors = preprocessor.get_preprocessors()
    assert all((k, v) in desired.items() for (k, v) in preprocessors.items())
    assert all((k, v) in preprocessors.items() for (k, v) in desired.items())

    # non time_series data test
    preprocessor = preprocess.Preprocess(opt={})
    desired = {
        "encode_category": {},
        "check_dtype": {},
        "check_cardinality": {},
        "standardization": {},
        "minmax": {},
        "logarithmic": {},
        "clipping": {},
        "binning": {},
        "rank": {},
        "rankgauss": {}
        }
    test_data_non_time_series = create_non_time_series_test_data(10, 5)
    preprocessor.check_data(test_data_non_time_series, is_time_series=False)
    preprocessors = preprocessor.get_preprocessors()
    assert all((k, v) in desired.items() for (k, v) in preprocessors.items())
    assert all((k, v) in preprocessors.items() for (k, v) in desired.items())


def test_apply():
    preprocessor = preprocess.Preprocess(opt={})
    # Testing with conversion type preprocessing ("todatetime")
    # and generation type preprocessing ("standardization")
    preprocessor.set_preprocessors({
        "todatetime": {},
        "standardization": {}
        })
    # Create time-series test data with the loaded state ("date" column type is object)
    test_data_time_series = create_time_series_test_data(10, 2)
    test_data_time_series["date"] = test_data_time_series["date"].astype(str)
    train, valid = split_dataset(test_data_time_series)

    desired_train = train.copy()
    desired_valid = valid.copy()
    desired_train["date"] = pd.to_datetime(desired_train["date"])
    desired_valid["date"] = pd.to_datetime(desired_valid["date"])
    desired_train_raw = desired_train.copy()
    desired_valid_raw = desired_valid.copy()
    scaler = StandardScaler()
    scaler.fit(desired_train[["float_0"]])
    desired_train["float_0_standardization"] = scaler.transform(desired_train[["float_0"]])
    desired_valid["float_0_standardization"] = scaler.transform(desired_valid[["float_0"]])
    train, valid = preprocessor.apply(train, valid, "target")
    pd.testing.assert_frame_equal(train, desired_train)
    pd.testing.assert_frame_equal(valid, desired_valid)
    pd.testing.assert_frame_equal(preprocessor.train_raw, desired_train_raw)
    pd.testing.assert_frame_equal(preprocessor.valid_raw, desired_valid_raw)


def test_todatetime():
    preprocessor = preprocess.Preprocess(opt={})
    # Create time-series test data with the loaded state ("date" column type is object)
    test_data_time_series = create_time_series_test_data(10, 2)
    test_data_time_series["date"] = test_data_time_series["date"].astype(str)
    train, valid = split_dataset(test_data_time_series)
    desired_train = train.copy()
    desired_valid = valid.copy()
    desired_train["date"] = pd.to_datetime(desired_train["date"])
    desired_valid["date"] = pd.to_datetime(desired_valid["date"])
    preprocessor.train = train.copy()
    preprocessor.valid = valid.copy()
    train, valid = preprocessor.todatetime(train, valid, "target", False)
    pd.testing.assert_frame_equal(train, desired_train)
    pd.testing.assert_frame_equal(valid, desired_valid)


def test_tolowerstr():
    preprocessor = preprocess.Preprocess(opt={})
    test_data_time_series = create_time_series_test_data(10, 2)
    test_data_time_series.rename({"int_0": "INT_0"})

    desired = test_data_time_series.copy()
    desired = desired.rename(columns={"INT_0": "int_0"})
    dataset = preprocessor.tolowerstr(test_data_time_series)
    pd.testing.assert_frame_equal(dataset, desired)


def test_dayofweek():
    preprocessor = preprocess.Preprocess(opt={})
    test_data_time_series = create_time_series_test_data(10, 2)
    train, valid = split_dataset(test_data_time_series)
    desired_train = train.copy()
    desired_valid = valid.copy()
    desired_train["dayofweek"] = desired_train["date"].dt.dayofweek
    desired_valid["dayofweek"] = desired_valid["date"].dt.dayofweek
    desired_train["dayofweek"] = desired_train["dayofweek"].astype("int64")
    desired_valid["dayofweek"] = desired_valid["dayofweek"].astype("int64")
    preprocessor.train = train.copy()
    preprocessor.valid = valid.copy()
    train, valid = preprocessor.dayofweek(train, valid, "target", False)
    pd.testing.assert_frame_equal(train, desired_train)
    pd.testing.assert_frame_equal(valid, desired_valid)


def test_dayofyear():
    preprocessor = preprocess.Preprocess(opt={})
    test_data_time_series = create_time_series_test_data(10, 2)
    train, valid = split_dataset(test_data_time_series)
    desired_train = train.copy()
    desired_valid = valid.copy()
    desired_train["dayofyear"] = desired_train["date"].dt.dayofyear
    desired_valid["dayofyear"] = desired_valid["date"].dt.dayofyear
    desired_train["dayofyear"] = desired_train["dayofyear"].astype("int64")
    desired_valid["dayofyear"] = desired_valid["dayofyear"].astype("int64")
    preprocessor.train = train.copy()
    preprocessor.valid = valid.copy()
    train, valid = preprocessor.dayofyear(train, valid, "target", False)
    pd.testing.assert_frame_equal(train, desired_train)
    pd.testing.assert_frame_equal(valid, desired_valid)


def test_create_daily_data():
    preprocessor = preprocess.Preprocess(opt={})
    start_date = datetime.datetime.today().date()
    dates = pd.date_range(start=start_date, periods=5, freq='D')
    test_data_time_series = create_non_time_series_test_data(20, 2)
    test_data_time_series["date"] = np.tile(dates, 4)
    desired = test_data_time_series.groupby("date")["target"].mean()
    desired = pd.DataFrame(desired).sort_index()
    dataset = preprocessor.create_daily_data(test_data_time_series, "target")
    pd.testing.assert_frame_equal(dataset, desired)


def test_check_corr():
    preprocessor = preprocess.Preprocess(opt={})
    lag = 5
    autocorrelated_data = create_autocorrelated_data(100, 2, lag)
    argmax_acf, argmax_pacf = preprocessor.check_corr(autocorrelated_data["target"])
    assert argmax_acf == lag
    assert argmax_pacf == lag


def test_lag():
    preprocessor = preprocess.Preprocess(opt={})
    lag = 5
    autocorrelated_data = create_autocorrelated_data(100, 2, lag)
    train, valid = split_dataset(autocorrelated_data)
    train, valid = preprocessor.lag(train, valid, "target", False)
    autocorrelated_data["lag"] = autocorrelated_data["target"].shift(lag)
    desired_train, desired_valid = split_dataset(autocorrelated_data)
    pd.testing.assert_frame_equal(train, desired_train)
    pd.testing.assert_frame_equal(valid, desired_valid)


def test_rolling():
    preprocessor = preprocess.Preprocess(opt={})
    lag = 5
    autocorrelated_data = create_autocorrelated_data(100, 2, lag)
    train, valid = split_dataset(autocorrelated_data)
    train, valid = preprocessor.rolling(train, valid, "target", False)
    lags = autocorrelated_data["target"].shift(1)
    autocorrelated_data["rolling"] = lags.rolling(lag - 1).mean()
    desired_train, desired_valid = split_dataset(autocorrelated_data)
    pd.testing.assert_frame_equal(train, desired_train)
    pd.testing.assert_frame_equal(valid, desired_valid)


def test_expanding():
    preprocessor = preprocess.Preprocess(opt={})
    lag = 5
    autocorrelated_data = create_autocorrelated_data(100, 2, lag)
    train, valid = split_dataset(autocorrelated_data)
    train, valid = preprocessor.expanding(train, valid, "target", False)
    lags = autocorrelated_data["target"].shift(1)
    autocorrelated_data["expanding"] = lags.expanding().mean()
    desired_train, desired_valid = split_dataset(autocorrelated_data)
    pd.testing.assert_frame_equal(train, desired_train)
    pd.testing.assert_frame_equal(valid, desired_valid)


def test_check_cardinality(caplog):
    caplog.set_level(WARNING)
    preprocessor = preprocess.Preprocess(opt={})
    test_data_time_series = create_time_series_test_data(10, 4)
    test_data_time_series.loc[:, "int_0"] = 0
    train, valid = split_dataset(test_data_time_series)
    train, valid = preprocessor.check_cardinality(train, valid, "target", False)
    msg = ("The cardinality of the ['int_1'] column is very high.\n"
           "    By using methods such as hierarchical grouping,\n"
           "    the cardinality can be reduced, leading to an improvement\n"
           "    in inference accuracy.")
    assert [('cyclic_boosting.tornado.core.preprocess', WARNING, msg)] == caplog.record_tuples


def test_check_dtype(caplog):
    caplog.set_level(WARNING)
    preprocessor = preprocess.Preprocess(opt={})
    test_data_time_series = create_time_series_test_data(10, 4)
    test_data_time_series.loc[:, "float_0"] = 1.0
    print(test_data_time_series)
    train, valid = split_dataset(test_data_time_series)
    train, valid = preprocessor.check_dtype(train, valid, "target", False)
    msg = ("Please check the columns ['float_0'].\n"
           "    Ensure that categorical variables are of 'int' type\n"
           "    and continuous variables are of 'float' type.")
    assert [('cyclic_boosting.tornado.core.preprocess', WARNING, msg)] == caplog.record_tuples


def test_standardization():
    preprocessor = preprocess.Preprocess(opt={})
    test_data_time_series = create_time_series_test_data(10, 2)
    train, valid = split_dataset(test_data_time_series)
    desired_train = train.copy()
    desired_valid = valid.copy()
    scaler = StandardScaler()
    scaler.fit(desired_train[["float_0"]])
    desired_train["float_0_standardization"] = scaler.transform(desired_train[["float_0"]])
    desired_valid["float_0_standardization"] = scaler.transform(desired_valid[["float_0"]])
    preprocessor.train = train.copy()
    preprocessor.valid = valid.copy()
    train, valid = preprocessor.standardization(train, valid, "target", False)
    pd.testing.assert_frame_equal(train, desired_train)
    pd.testing.assert_frame_equal(valid, desired_valid)


def test_minmax():
    preprocessor = preprocess.Preprocess(opt={})
    test_data_time_series = create_time_series_test_data(10, 2)
    train, valid = split_dataset(test_data_time_series)
    desired_train = train.copy()
    desired_valid = valid.copy()
    scaler = MinMaxScaler()
    scaler.fit(desired_train[["float_0"]])
    desired_train["float_0_minmax"] = scaler.transform(desired_train[["float_0"]])
    desired_valid["float_0_minmax"] = scaler.transform(desired_valid[["float_0"]])
    preprocessor.train = train.copy()
    preprocessor.valid = valid.copy()
    train, valid = preprocessor.minmax(train, valid, "target", False)
    pd.testing.assert_frame_equal(train, desired_train)
    pd.testing.assert_frame_equal(valid, desired_valid)


def test_logarithmic():
    preprocessor = preprocess.Preprocess(opt={})
    test_data_time_series = create_time_series_test_data(10, 2)
    train, valid = split_dataset(test_data_time_series)
    desired_train = train.copy()
    desired_valid = valid.copy()
    scaler = PowerTransformer()
    scaler.fit(desired_train[["float_0"]])
    desired_train["float_0_logarithmic"] = scaler.transform(desired_train[["float_0"]])
    desired_valid["float_0_logarithmic"] = scaler.transform(desired_valid[["float_0"]])
    preprocessor.train = train.copy()
    preprocessor.valid = valid.copy()
    train, valid = preprocessor.logarithmic(train, valid, "target", False)
    pd.testing.assert_frame_equal(train, desired_train)
    pd.testing.assert_frame_equal(valid, desired_valid)


def test_clipping():
    preprocessor = preprocess.Preprocess(opt={})
    test_data_time_series = create_time_series_test_data(10, 2)
    train, valid = split_dataset(test_data_time_series)
    desired_train = train.copy()
    desired_valid = valid.copy()
    p_l = desired_train["float_0"].quantile(0.01)
    p_u = desired_train["float_0"].quantile(0.99)
    desired_train["float_0_clipping"] = desired_train[["float_0"]].clip(p_l, p_u, axis=1)
    desired_valid["float_0_clipping"] = desired_valid[["float_0"]].clip(p_l, p_u, axis=1)
    preprocessor.train = train.copy()
    preprocessor.valid = valid.copy()
    train, valid = preprocessor.clipping(train, valid, "target", False)
    pd.testing.assert_frame_equal(train, desired_train)
    pd.testing.assert_frame_equal(valid, desired_valid)


def test_binning():
    preprocessor = preprocess.Preprocess(opt={})
    test_data_time_series = create_time_series_test_data(10, 2)
    train, valid = split_dataset(test_data_time_series)
    desired_train = train.copy()
    desired_valid = valid.copy()
    # print(int(np.log2(desired_train.shape[0]) + 1))
    scaler = KBinsDiscretizer(n_bins=4,
                              encode="ordinal",
                              strategy="uniform",
                              random_state=0,
                              subsample=200000)
    scaler.fit(desired_train[["float_0"]])
    desired_train["float_0_binning"] = scaler.transform(desired_train[["float_0"]])
    desired_valid["float_0_binning"] = scaler.transform(desired_valid[["float_0"]])
    preprocessor.train = train.copy()
    preprocessor.valid = valid.copy()
    train, valid = preprocessor.binning(train, valid, "target", False)
    pd.testing.assert_frame_equal(train, desired_train)
    pd.testing.assert_frame_equal(valid, desired_valid)


def test_rank():
    preprocessor = preprocess.Preprocess(opt={})
    test_data_time_series = create_time_series_test_data(10, 2)
    train, valid = split_dataset(test_data_time_series)
    desired_train = train.copy()
    desired_valid = valid.copy()
    scaler = QuantileTransformer(n_quantiles=8, output_distribution="uniform", random_state=0)
    scaler.fit(desired_train[["float_0"]])
    desired_train["float_0_rank"] = scaler.transform(desired_train[["float_0"]])
    desired_valid["float_0_rank"] = scaler.transform(desired_valid[["float_0"]])
    preprocessor.train = train.copy()
    preprocessor.valid = valid.copy()
    train, valid = preprocessor.rank(train, valid, "target", False)
    pd.testing.assert_frame_equal(train, desired_train)
    pd.testing.assert_frame_equal(valid, desired_valid)


def test_rankgauss():
    preprocessor = preprocess.Preprocess(opt={})
    test_data_time_series = create_time_series_test_data(10, 2)
    train, valid = split_dataset(test_data_time_series)
    desired_train = train.copy()
    desired_valid = valid.copy()
    scaler = QuantileTransformer(n_quantiles=8, output_distribution="normal", random_state=0)
    scaler.fit(desired_train[["float_0"]])
    desired_train["float_0_rankgauss"] = scaler.transform(desired_train[["float_0"]])
    desired_valid["float_0_rankgauss"] = scaler.transform(desired_valid[["float_0"]])
    preprocessor.train = train.copy()
    preprocessor.valid = valid.copy()
    train, valid = preprocessor.rankgauss(train, valid, "target", False)
    pd.testing.assert_frame_equal(train, desired_train)
    pd.testing.assert_frame_equal(valid, desired_valid)


def onehot_encording(self, train, valid, target, params_exist):
    pass


def label_encording(self, train, valid, target, params_exist):
    pass


def feature_hashing(self, train, valid, target, params_exist):
    pass


def freqency_encording(self, train, valid, target, params_exist):
    pass


def target_encording(self, train, valid, target, params_exist):
    pass


def encode_category(self, train, valid, target, params_exist):
    pass
