from logging import WARNING

import numpy as np
import pandas as pd
import random
import pytest
import os
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    KBinsDiscretizer,
    TargetEncoder,
)
from sklearn.feature_extraction import FeatureHasher

from cyclic_boosting.tornado.core import preprocess


def create_time_series_test_data(n_rows, n_cols, start_date=None):
    if start_date is None:
        start_date = datetime.datetime.today().date()
    else:
        start_date = pd.to_datetime(start_date).date()
    dates = pd.date_range(start=start_date, periods=n_rows, freq="D")
    df = create_non_time_series_test_data(n_rows, n_cols)
    df["date"] = dates

    return df


def create_non_time_series_test_data(n_rows, n_cols):
    data_category = np.array([random.sample(range(1000), n_rows) for i in range(int(n_cols / 2))]).T
    data_continuous = np.random.random(size=(n_rows, n_cols - int(n_cols / 2)))
    data_target = np.random.random(size=(n_rows, 1))
    cols_category = [f"category_{col}" for col in range(int(n_cols / 2))]
    cols_continuous = [f"continuous_{col}" for col in range(n_cols - int(n_cols / 2))]
    df_category = pd.DataFrame(data_category, columns=cols_category)
    df_continuous = pd.DataFrame(data_continuous, columns=cols_continuous)
    df_target = pd.DataFrame(data_target, columns=["target"])
    df = pd.concat([df_category, df_continuous, df_target], axis=1)

    return df


def create_autocorrelated_data(n_rows, n_cols, lag, start_date=None, autocorrelation_coefficient=0.9):
    mean = 0
    std_deviation = 1
    data = [np.random.normal(mean, std_deviation)] * lag
    for _ in range(lag, n_rows):
        new_value = autocorrelation_coefficient * data[-lag] + np.random.normal(mean, std_deviation)
        data.append(new_value)
    df = create_time_series_test_data(n_rows, n_cols, start_date)
    df["target"] = data

    return df


def create_str(n_character, last_character="z"):
    string = "".join(chr(random.choice(range(ord("a"), ord(last_character)))) for _ in range(n_character))

    return string


def create_categorical_data(n_rows, n_cols, start_date=None):
    df = create_time_series_test_data(n_rows, n_cols, start_date)
    cat = list()
    while len(cat) != n_rows:
        strings = create_str(5)
        if strings not in cat:
            cat.append(strings)
    df["cat"] = cat

    return df


def create_categorical_duplicated_data(n_rows, n_cols, start_date=None):
    df = create_time_series_test_data(n_rows, n_cols, start_date)
    cat = [create_str(2, "c") for i in range(int(n_rows * 0.6))]
    cat_duplicated = ["aa", "ab", "ba", "bb"] * int(n_rows * 0.1)
    cat += cat_duplicated
    df["cat"] = cat

    return df


def split_dataset(df):
    test_size = 0.2
    seed = 0
    train, valid = train_test_split(df, test_size=test_size, random_state=seed)
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
        "encode_category": {"label_encoding": {}},
        "check_dtype": {},
        "check_cardinality": {},
        "check_data_leakage": {},
        "todatetime": {},
        "lag": {},
        "rolling": {},
        "expanding": {},
        "dayofweek": {},
        "dayofyear": {},
    }
    test_data_time_series = create_time_series_test_data(10, 5)
    preprocessor.check_data(test_data_time_series, is_time_series=True)
    preprocessors = preprocessor.get_preprocessors()

    assert all((k, v) in desired.items() for (k, v) in preprocessors.items())
    assert all((k, v) in preprocessors.items() for (k, v) in desired.items())

    # non time_series data test
    preprocessor = preprocess.Preprocess(opt={})
    desired = {
        "encode_category": {"label_encoding": {}},
        "check_dtype": {},
        "check_cardinality": {},
        "check_data_leakage": {},
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
    preprocessor.set_preprocessors({"todatetime": {}, "standardization": {}})
    # Create time-series test data with the loaded state
    # ("date" column type is object)
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
    scaler.fit(desired_train[["continuous_0"]])
    desired_train["continuous_0_standardization"] = scaler.transform(desired_train[["continuous_0"]])
    desired_valid["continuous_0_standardization"] = scaler.transform(desired_valid[["continuous_0"]])
    train, valid = preprocessor.apply(train, valid, "target")

    pd.testing.assert_frame_equal(train, desired_train)
    pd.testing.assert_frame_equal(valid, desired_valid)
    pd.testing.assert_frame_equal(preprocessor.train_raw, desired_train_raw)
    pd.testing.assert_frame_equal(preprocessor.valid_raw, desired_valid_raw)


def test_apply_with_opt():
    preprocessor = preprocess.Preprocess(opt={"standardization": {}})
    # Tests with "todatetime" as the default preprocessing method and
    # "standardization" as an optional preprocessing method
    preprocessor.set_preprocessors(
        {
            "todatetime": {},
        }
    )
    # Create time-series test data with the loaded state
    # ("date" column type is object)
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
    scaler.fit(desired_train[["continuous_0"]])
    desired_train["continuous_0_standardization"] = scaler.transform(desired_train[["continuous_0"]])
    desired_valid["continuous_0_standardization"] = scaler.transform(desired_valid[["continuous_0"]])
    train, valid = preprocessor.apply(train, valid, "target")

    pd.testing.assert_frame_equal(train, desired_train)
    pd.testing.assert_frame_equal(valid, desired_valid)
    pd.testing.assert_frame_equal(preprocessor.train_raw, desired_train_raw)
    pd.testing.assert_frame_equal(preprocessor.valid_raw, desired_valid_raw)


def test_todatetime():
    preprocessor = preprocess.Preprocess(opt={})
    # Create time-series test data with the loaded state
    # ("date" column type is object)
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
    test_data_time_series.rename({"category_0": "CATEGORY_0"})

    desired = test_data_time_series.copy()
    desired = desired.rename(columns={"CATEGORY_0": "category_0"})
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


def test_create_time_based_average_data():
    preprocessor = preprocess.Preprocess(opt={})
    start_date = datetime.datetime.today().date()
    dates = pd.date_range(start=start_date, periods=5, freq="D")
    test_data_time_series = create_non_time_series_test_data(20, 2)
    test_data_time_series["date"] = np.tile(dates, 4)

    desired = test_data_time_series.groupby("date")["target"].mean()
    desired = pd.DataFrame(desired).sort_index()

    dataset = preprocessor.create_time_based_average_data(test_data_time_series, "target")

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
    test_data_time_series.loc[:, "category_0"] = 0
    train, valid = split_dataset(test_data_time_series)
    train, valid = preprocessor.check_cardinality(train, valid, "target", False)

    msg = (
        "The cardinality of the ['category_1'] column is very high.\n"
        "    By using methods such as hierarchical grouping,\n"
        "    the cardinality can be reduced,\n"
        "    leading to an improvement in inference accuracy."
    )
    logger_name = "cyclic_boosting.tornado.core.preprocess"
    assert [(logger_name, WARNING, msg)] == caplog.record_tuples


def test_check_dtype(caplog):
    caplog.set_level(WARNING)
    preprocessor = preprocess.Preprocess(opt={})
    test_data_time_series = create_time_series_test_data(10, 4)
    test_data_time_series.loc[:, "continuous_0"] = 1.0
    train, valid = split_dataset(test_data_time_series)
    train, valid = preprocessor.check_dtype(train, valid, "target", False)

    msg = (
        "Please check the columns ['continuous_0'].\n"
        "    Ensure that categorical variables are of 'int' type\n"
        "    and continuous variables are of 'float' type."
    )
    logger_name = "cyclic_boosting.tornado.core.preprocess"
    assert [(logger_name, WARNING, msg)] == caplog.record_tuples


def test_check_data_leakage(caplog):
    caplog.set_level(WARNING)
    preprocessor = preprocess.Preprocess(opt={})
    test_data_time_series = create_time_series_test_data(10, 4)
    test_data_time_series.loc[:, "continuous_0"] = test_data_time_series.loc[:, "target"]
    train, valid = split_dataset(test_data_time_series)
    train, valid = preprocessor.check_data_leakage(train, valid, "target", False)

    msg = (
        "Variance Inflation Factor (VIF) of the objective variable is infinite.\n"
        "This means that there is a very high association (multi-collinearity)\n"
        "between the explanatory variables and the objective variable.\n"
        "Confirmation is recommended due to the possibility of target leakage."
    )
    logger_name = "cyclic_boosting.tornado.core.preprocess"
    assert [(logger_name, WARNING, msg)] == caplog.record_tuples


def test_standardization():
    preprocessor = preprocess.Preprocess(opt={})
    test_data_time_series = create_time_series_test_data(10, 2)
    train, valid = split_dataset(test_data_time_series)

    desired_train = train.copy()
    desired_valid = valid.copy()
    scaler = StandardScaler()
    scaler.fit(desired_train[["continuous_0"]])
    desired_train["continuous_0_standardization"] = scaler.transform(desired_train[["continuous_0"]])
    desired_valid["continuous_0_standardization"] = scaler.transform(desired_valid[["continuous_0"]])

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
    scaler.fit(desired_train[["continuous_0"]])
    desired_train["continuous_0_minmax"] = scaler.transform(desired_train[["continuous_0"]])
    desired_valid["continuous_0_minmax"] = scaler.transform(desired_valid[["continuous_0"]])

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
    scaler.fit(desired_train[["continuous_0"]])
    desired_train["continuous_0_logarithmic"] = scaler.transform(desired_train[["continuous_0"]])
    desired_valid["continuous_0_logarithmic"] = scaler.transform(desired_valid[["continuous_0"]])

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
    p_l = desired_train["continuous_0"].quantile(0.01)
    p_u = desired_train["continuous_0"].quantile(0.99)
    desired_train["continuous_0_clipping"] = desired_train[["continuous_0"]].clip(p_l, p_u, axis=1)
    desired_valid["continuous_0_clipping"] = desired_valid[["continuous_0"]].clip(p_l, p_u, axis=1)

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
    scaler = KBinsDiscretizer(n_bins=4, encode="ordinal", strategy="uniform", random_state=0, subsample=200000)
    scaler.fit(desired_train[["continuous_0"]])
    desired_train["continuous_0_binning"] = scaler.transform(desired_train[["continuous_0"]])
    desired_valid["continuous_0_binning"] = scaler.transform(desired_valid[["continuous_0"]])

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
    scaler.fit(desired_train[["continuous_0"]])
    desired_train["continuous_0_rank"] = scaler.transform(desired_train[["continuous_0"]])
    desired_valid["continuous_0_rank"] = scaler.transform(desired_valid[["continuous_0"]])

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
    scaler.fit(desired_train[["continuous_0"]])
    desired_train["continuous_0_rankgauss"] = scaler.transform(desired_train[["continuous_0"]])
    desired_valid["continuous_0_rankgauss"] = scaler.transform(desired_valid[["continuous_0"]])

    preprocessor.train = train.copy()
    preprocessor.valid = valid.copy()
    train, valid = preprocessor.rankgauss(train, valid, "target", False)

    pd.testing.assert_frame_equal(train, desired_train)
    pd.testing.assert_frame_equal(valid, desired_valid)


def test_onehot_encoding():
    preprocessor = preprocess.Preprocess(opt={})
    test_data_categorical = create_categorical_data(10, 2)
    train, valid = split_dataset(test_data_categorical)

    desired = test_data_categorical.copy()
    for i in range(desired.shape[0]):
        val = desired.loc[i, "cat"]
        desired.loc[i, f"cat_{val}"] = 1
        desired.fillna(0, inplace=True)
        desired[f"cat_{val}"] = desired[f"cat_{val}"].astype(int)
    desired.drop(columns="cat", inplace=True)
    desired_train, desired_valid = split_dataset(desired)
    desired_train_raw = train.drop(columns="cat")
    desired_valid_raw = valid.drop(columns="cat")

    preprocessor.train = train.copy()
    preprocessor.valid = valid.copy()
    train, valid = preprocessor.onehot_encoding(train, valid, "target", False)
    train = train.reindex(columns=desired_train.columns)
    valid = valid.reindex(columns=desired_valid.columns)

    pd.testing.assert_frame_equal(train, desired_train)
    pd.testing.assert_frame_equal(valid, desired_valid)
    pd.testing.assert_frame_equal(preprocessor.train, desired_train_raw)
    pd.testing.assert_frame_equal(preprocessor.valid, desired_valid_raw)


def test_label_encoding():
    preprocessor = preprocess.Preprocess(opt={})
    test_data_categorical = create_categorical_data(10, 2)
    test_data_categorical.loc[9, "cat"] = np.nan
    train, valid = split_dataset(test_data_categorical)

    desired = test_data_categorical.copy()
    desired.sort_values("cat", inplace=True)
    cat_label_encoding = np.arange(10)
    cat_label_encoding[-1] = -2
    desired["cat_label_encoding"] = cat_label_encoding
    desired.drop(columns="cat", inplace=True)
    desired.sort_index(inplace=True)
    desired_train, desired_valid = split_dataset(desired)
    desired_train_raw = train.drop(columns="cat")
    desired_valid_raw = valid.drop(columns="cat")

    preprocessor.train = train.copy()
    preprocessor.valid = valid.copy()
    train, valid = preprocessor.label_encoding(train, valid, "target", False)

    pd.testing.assert_frame_equal(train, desired_train)
    pd.testing.assert_frame_equal(valid, desired_valid)
    pd.testing.assert_frame_equal(preprocessor.train, desired_train_raw)
    pd.testing.assert_frame_equal(preprocessor.valid, desired_valid_raw)


def test_feature_hashing():
    preprocessor = preprocess.Preprocess(opt={})
    test_data_categorical = create_categorical_data(10, 2)
    train, valid = split_dataset(test_data_categorical)

    desired = test_data_categorical.copy()
    fh = FeatureHasher(n_features=10, input_type="string", dtype=np.int64)
    hash_desired = fh.transform(desired["cat"].astype(str).values[:, np.newaxis].tolist())
    hash_desired = pd.DataFrame(hash_desired.todense(), columns=[f"cat_{i}" for i in range(10)])
    desired = pd.concat([desired, hash_desired], axis=1)
    desired.drop(columns="cat", inplace=True)
    desired_train, desired_valid = split_dataset(desired)
    desired_train_raw = train.drop(columns="cat")
    desired_valid_raw = valid.drop(columns="cat")

    preprocessor.train = train.copy()
    preprocessor.valid = valid.copy()
    train, valid = preprocessor.feature_hashing(train, valid, "target", False)

    pd.testing.assert_frame_equal(train, desired_train)
    pd.testing.assert_frame_equal(valid, desired_valid)
    pd.testing.assert_frame_equal(preprocessor.train, desired_train_raw)
    pd.testing.assert_frame_equal(preprocessor.valid, desired_valid_raw)


def test_frequency_encoding():
    preprocessor = preprocess.Preprocess(opt={})
    test_data_categorical = create_categorical_data(10, 2)
    train, valid = split_dataset(test_data_categorical)
    train.iloc[1, -1] = train.iloc[0, -1]

    desired_train = train.copy()
    desired_valid = valid.copy()
    desired_train["cat_frequency_encoding"] = [2, 2, 1, 1, 1, 1, 1, 1]
    desired_valid["cat_frequency_encoding"] = [np.nan, np.nan]
    desired_train.drop(columns="cat", inplace=True)
    desired_valid.drop(columns="cat", inplace=True)
    desired_train_raw = train.drop(columns="cat")
    desired_valid_raw = valid.drop(columns="cat")

    preprocessor.train = train.copy()
    preprocessor.valid = valid.copy()
    train, valid = preprocessor.frequency_encoding(train, valid, "target", False)

    pd.testing.assert_frame_equal(train, desired_train)
    pd.testing.assert_frame_equal(valid, desired_valid)
    pd.testing.assert_frame_equal(preprocessor.train, desired_train_raw)
    pd.testing.assert_frame_equal(preprocessor.valid, desired_valid_raw)


def test_target_encoding():
    preprocessor = preprocess.Preprocess(opt={})
    test_data_categorical = create_categorical_duplicated_data(100, 2)
    train, valid = split_dataset(test_data_categorical)

    desired_train = train.copy()
    desired_valid = valid.copy()
    te = TargetEncoder(random_state=0)
    X_train = np.array([desired_train["cat"].values]).T
    X_valid = np.array([desired_valid["cat"].values]).T
    y = desired_train["target"].values
    desired_train["cat"] = te.fit_transform(X_train, y)
    desired_valid["cat"] = te.transform(X_valid)
    desired_train.rename(columns={"cat": "cat_target_encoding"}, inplace=True)
    desired_valid.rename(columns={"cat": "cat_target_encoding"}, inplace=True)
    desired_train_raw = train.drop(columns="cat")
    desired_valid_raw = valid.drop(columns="cat")

    preprocessor.train = train.copy()
    preprocessor.valid = valid.copy()
    train, valid = preprocessor.target_encoding(train, valid, "target", False)

    pd.testing.assert_frame_equal(train, desired_train)
    pd.testing.assert_frame_equal(valid, desired_valid)
    pd.testing.assert_frame_equal(preprocessor.train, desired_train_raw)
    pd.testing.assert_frame_equal(preprocessor.valid, desired_valid_raw)


def test_encode_category_dtype_error():
    preprocessor = preprocess.Preprocess(opt={})
    test_data_categorical = create_categorical_duplicated_data(100, 2)
    test_data_categorical.iloc[0, -1] = 0
    train, valid = split_dataset(test_data_categorical)

    with pytest.raises(RuntimeError) as e:
        train, valid = preprocessor.encode_category(train, valid, "target", False)

    msg = "The dataset has differenct dtype in same col"
    assert str(e.value) == msg


def test_encode_category_n_encoders_error():
    preprocessor = preprocess.Preprocess(opt={})
    preprocessor.set_preprocessors({"encode_category": {"onehot_encoding": {}, "label_encoding": {}}})
    test_data_categorical = create_categorical_duplicated_data(100, 2)
    train, valid = split_dataset(test_data_categorical)

    with pytest.raises(RuntimeError) as e:
        train, valid = preprocessor.encode_category(train, valid, "target", False)

    msg = "Single encoding method should be used for categorical variables."
    assert str(e.value) == msg


def test_encode_category():
    preprocessor = preprocess.Preprocess(opt={})
    preprocessor.set_preprocessors({"encode_category": {"label_encoding": {}}})
    test_data_categorical = create_categorical_duplicated_data(100, 2)
    train, valid = split_dataset(test_data_categorical)

    desired = test_data_categorical.copy()
    values = pd.Series([0, 1, 2, 3], index=["aa", "ab", "ba", "bb"])
    desired["cat_label_encoding"] = desired["cat"].map(values)
    desired.drop(columns="cat", inplace=True)
    desired_train, desired_valid = split_dataset(desired)
    desired_train_raw = train.drop(columns="cat")
    desired_valid_raw = valid.drop(columns="cat")

    preprocessor.train = train.copy()
    preprocessor.valid = valid.copy()
    train, valid = preprocessor.encode_category(train, valid, "target", False)

    pd.testing.assert_frame_equal(train, desired_train)
    pd.testing.assert_frame_equal(valid, desired_valid)
    pd.testing.assert_frame_equal(train, preprocessor.train_raw)
    pd.testing.assert_frame_equal(valid, preprocessor.valid_raw)
    pd.testing.assert_frame_equal(preprocessor.train, desired_train_raw)
    pd.testing.assert_frame_equal(preprocessor.valid, desired_valid_raw)
