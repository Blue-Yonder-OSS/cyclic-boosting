from logging import INFO

import numpy as np
import pandas as pd
import pickle
import os
import datetime
import re
from sklearn.model_selection import train_test_split

from cyclic_boosting.tornado.core import datamodule


def create_correlated_data(n_rows, is_time_series):
    t = np.arange(0, n_rows * 0.1, 0.1)
    y0 = np.sin(2 * np.pi * t) + np.random.random(n_rows) * 0.1
    y1 = np.sin(2 * np.pi * t + 1) + np.random.random(n_rows) * 0.1
    target = np.sin(2 * np.pi * t) + np.random.random(n_rows) * 0.1
    df = pd.DataFrame(np.array([y0, y1, target]).T, columns=["y0", "y1", "target"])
    if is_time_series:
        start_date = datetime.datetime.today().date()
        dates = pd.date_range(start=start_date, periods=n_rows, freq="D")
        df["date"] = dates

    return df


def split_dataset(df):
    test_size = 0.2
    seed = 0
    train, valid = train_test_split(df, test_size=test_size, random_state=seed)
    return train, valid


def test_set_log():
    # load failure
    path = "./test_data.csv"
    log_path = "./test_data.pkl"
    data_deliverer = datamodule.TornadoDataModule(path)
    data_deliverer.set_log()
    assert data_deliverer.log_path == log_path
    assert data_deliverer.preprocessors == {}
    assert data_deliverer.features == []

    # load success
    path = "./test_data.csv"
    log_path = "./test_data_success.pkl"
    with open(log_path, "wb") as p:
        log = {"preprocessors": {"test": {}}, "features": ["test"]}
        pickle.dump(log, p)
    data_deliverer = datamodule.TornadoDataModule(path)
    data_deliverer.log_path = log_path
    data_deliverer.set_log()
    assert data_deliverer.log_path == log_path
    assert data_deliverer.preprocessors == {"test": {}}
    assert data_deliverer.features == ["test"]
    os.remove(log_path)


def test_corr_based_removal():
    # time series
    path = "./test_data.csv"
    data_deliverer = datamodule.TornadoDataModule(path)
    correlated_data = create_correlated_data(10, True)
    correlated_data["y0_drop"] = correlated_data["y0"] + np.random.random(10) * 0.5
    train, valid = split_dataset(correlated_data)

    desired = correlated_data.copy()
    desired.drop(columns="y0_drop", inplace=True)
    desired_train, desired_valid = split_dataset(desired)

    data_deliverer.train = train
    data_deliverer.valid = valid
    data_deliverer.target = "target"
    data_deliverer.corr_based_removal()
    data_deliverer.train = data_deliverer.train.reindex(columns=desired_train.columns)
    data_deliverer.valid = data_deliverer.valid.reindex(columns=desired_valid.columns)
    pd.testing.assert_frame_equal(data_deliverer.train, desired_train)
    pd.testing.assert_frame_equal(data_deliverer.valid, desired_valid)

    # non time series
    path = "./test_data.csv"
    data_deliverer = datamodule.TornadoDataModule(path)
    correlated_data = create_correlated_data(10, False)
    correlated_data["y0_drop"] = correlated_data["y0"] + np.random.random(10) * 0.5
    train, valid = split_dataset(correlated_data)

    desired = correlated_data.copy()
    desired.drop(columns="y0_drop", inplace=True)
    desired_train, desired_valid = split_dataset(desired)

    data_deliverer.train = train
    data_deliverer.valid = valid
    data_deliverer.target = "target"
    data_deliverer.corr_based_removal()
    data_deliverer.train = data_deliverer.train.reindex(columns=desired_train.columns)
    data_deliverer.valid = data_deliverer.valid.reindex(columns=desired_valid.columns)
    pd.testing.assert_frame_equal(data_deliverer.train, desired_train)
    pd.testing.assert_frame_equal(data_deliverer.valid, desired_valid)


def test_vif_based_removal():
    # time series
    path = "./test_data.csv"
    data_deliverer = datamodule.TornadoDataModule(path)
    correlated_data = create_correlated_data(100, True)
    t = np.arange(0, 100 * 0.1, 0.1)
    correlated_data["y0_drop"] = np.sin(2 * np.pi * t + 0.5) + np.random.random(100) * 0.1
    train, valid = split_dataset(correlated_data)

    desired = correlated_data.copy()
    desired.drop(columns="y0_drop", inplace=True)
    desired_train, desired_valid = split_dataset(desired)

    data_deliverer.train = train
    data_deliverer.valid = valid
    data_deliverer.target = "target"
    data_deliverer.vif_based_removal()
    data_deliverer.train = data_deliverer.train.reindex(columns=desired_train.columns)
    data_deliverer.valid = data_deliverer.valid.reindex(columns=desired_valid.columns)
    pd.testing.assert_frame_equal(data_deliverer.train, desired_train)
    pd.testing.assert_frame_equal(data_deliverer.valid, desired_valid)

    # non time series
    path = "./test_data.csv"
    data_deliverer = datamodule.TornadoDataModule(path)
    correlated_data = create_correlated_data(100, False)
    t = np.arange(0, 100 * 0.1, 0.1)
    correlated_data["y0_drop"] = np.sin(2 * np.pi * t + 0.5) + np.random.random(100) * 0.1
    train, valid = split_dataset(correlated_data)

    desired = correlated_data.copy()
    desired.drop(columns="y0_drop", inplace=True)
    desired_train, desired_valid = split_dataset(desired)

    data_deliverer.train = train
    data_deliverer.valid = valid
    data_deliverer.target = "target"
    data_deliverer.vif_based_removal()
    data_deliverer.train = data_deliverer.train.reindex(columns=desired_train.columns)
    data_deliverer.valid = data_deliverer.valid.reindex(columns=desired_valid.columns)
    pd.testing.assert_frame_equal(data_deliverer.train, desired_train)
    pd.testing.assert_frame_equal(data_deliverer.valid, desired_valid)


def test_remove_features():
    # without preprocessing log
    path = "./test_data.csv"
    data_deliverer = datamodule.TornadoDataModule(path)
    correlated_data = create_correlated_data(100, True)
    t = np.arange(0, 100 * 0.1, 0.1)
    correlated_data["y0_drop"] = np.sin(2 * np.pi * t + 0.5) + np.random.random(100) * 0.1
    train, valid = split_dataset(correlated_data)

    desired = correlated_data.copy()
    desired.drop(columns="y0_drop", inplace=True)
    desired_train, desired_valid = split_dataset(desired)

    data_deliverer.train = train
    data_deliverer.valid = valid
    data_deliverer.target = "target"
    data_deliverer.features = []
    data_deliverer.remove_features()
    data_deliverer.train = data_deliverer.train.reindex(columns=desired_train.columns)
    data_deliverer.valid = data_deliverer.valid.reindex(columns=desired_valid.columns)
    pd.testing.assert_frame_equal(data_deliverer.train, desired_train)
    pd.testing.assert_frame_equal(data_deliverer.valid, desired_valid)

    # with preprocessing log
    path = "./test_data.csv"
    data_deliverer = datamodule.TornadoDataModule(path)

    correlated_data = create_correlated_data(100, True)
    correlated_data["y0_drop"] = np.random.random(100)
    train, valid = split_dataset(correlated_data)

    desired = correlated_data.copy()
    desired.drop(columns="y0_drop", inplace=True)
    desired_train, desired_valid = split_dataset(desired)

    data_deliverer.train = train
    data_deliverer.valid = valid
    data_deliverer.target = "target"
    data_deliverer.features = ["y0", "y1", "date", "target"]
    data_deliverer.remove_features()
    data_deliverer.train = data_deliverer.train.reindex(columns=desired_train.columns)
    data_deliverer.valid = data_deliverer.valid.reindex(columns=desired_valid.columns)
    pd.testing.assert_frame_equal(data_deliverer.train, desired_train)
    pd.testing.assert_frame_equal(data_deliverer.valid, desired_valid)


def test_generate_trainset(caplog):
    caplog.set_level(INFO)
    # without parameter
    path = "./test_data.csv"
    log_path = "./test_data.pkl"
    data_deliverer = datamodule.TornadoDataModule(path)
    correlated_data = create_correlated_data(100, True)
    t = np.arange(0, 100 * 0.1, 0.1)
    correlated_data["y0_drop"] = np.sin(2 * np.pi * t + 0.5) + np.random.random(100) * 0.1
    correlated_data.to_csv(path, index=False)

    desired_n_features_original = [4]
    desired_n_features_preprocessed = [9]
    desired_n_features_selected = [6]
    desired_n_features = [desired_n_features_original, desired_n_features_preprocessed, desired_n_features_selected]

    train, valid = data_deliverer.generate_trainset(
        "target",
        test_size=0.2,
        seed=0,
        is_time_series=True,
    )

    src = pd.read_csv(path)
    desired = src.head(int(len(src) * 0.8))
    pd.testing.assert_index_equal(train.index, desired.index)

    for logger_name, log_level, message in caplog.record_tuples:
        if "datamodule" in logger_name and "->" in message:
            n_features = re.findall(r"\d+", message)
            for i, n in enumerate(n_features):
                assert int(n) in desired_n_features[i]
    assert data_deliverer.log_path == log_path
    with open(log_path, "rb") as p:
        log = pickle.load(p)
    os.remove(path)
    os.remove(log_path)

    assert all(k in data_deliverer.preprocessors.keys() for k in log["preprocessors"].keys())
    assert all(k in log["preprocessors"].keys() for k in data_deliverer.preprocessors.keys())
    assert data_deliverer.features == log["features"]


def test_generate_testset(caplog):
    caplog.set_level(INFO)
    path = "./test_data.csv"
    log_path = "./test_data.pkl"
    data_deliverer = datamodule.TornadoDataModule(path)
    correlated_data = create_correlated_data(100, True)
    t = np.arange(0, 100 * 0.1, 0.1)
    correlated_data["y0_drop"] = np.sin(2 * np.pi * t + 0.5) + np.random.random(100) * 0.1
    correlated_data.to_csv(path, index=False)

    train, valid = data_deliverer.generate_trainset(
        "target",
        test_size=0.2,
        seed=0,
        is_time_series=True,
    )
    dataset = data_deliverer.generate_testset(correlated_data).sort_index()
    desired_dataset = pd.concat([train, valid]).drop(columns="target").sort_index()

    pd.testing.assert_frame_equal(dataset, desired_dataset)
    os.remove(path)
    os.remove(log_path)
