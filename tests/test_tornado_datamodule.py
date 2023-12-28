import warnings
from logging import INFO

import numpy as np
import pandas as pd
import pickle
import pytest
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
    df = pd.DataFrame(np.array([y0, y1, target]).T,
                      columns=["y0", "y1", "target"])
    if is_time_series:
        start_date = datetime.datetime.today().date()
        dates = pd.date_range(start=start_date, periods=n_rows, freq='D')
        df["date"] = dates

    return df


def split_dataset(df):
    test_size = 0.2
    seed = 0
    train, valid = train_test_split(
        df,
        test_size=test_size,
        random_state=seed)
    return train, valid


def test_set_log():
    # load failure
    path = "./test_data.csv"
    log_path = "./test_data.pickle"
    data_deliverler = datamodule.TornadoDataModule(path)
    data_deliverler.set_log()
    assert data_deliverler.log_path == log_path
    assert data_deliverler.preprocessors == {}
    assert data_deliverler.features == []

    # load success
    path = "./test_data.csv"
    log_path = "./test_data_success.pickle"
    with open(log_path, "wb") as p:
        log = {"preprocessors": {"test": {}},
               "features": ["test"]}
        pickle.dump(log, p)
    data_deliverler = datamodule.TornadoDataModule(path)
    data_deliverler.log_path = log_path
    data_deliverler.set_log()
    assert data_deliverler.log_path == log_path
    assert data_deliverler.preprocessors == {"test": {}}
    assert data_deliverler.features == ["test"]
    os.remove(log_path)


def test_corr_based_removal():
    # time series
    path = "./test_data.csv"
    data_deliverler = datamodule.TornadoDataModule(path)
    correlated_data = create_correlated_data(10, True)
    correlated_data["y0_drop"] = (correlated_data["y0"] +
                                  np.random.random(10) * 0.5)
    train, valid = split_dataset(correlated_data)

    desired = correlated_data.copy()
    desired.drop(columns="y0_drop", inplace=True)
    desired_train, desired_valid = split_dataset(desired)

    data_deliverler.train = train
    data_deliverler.valid = valid
    data_deliverler.target = "target"
    data_deliverler.corr_based_removal()
    data_deliverler.train = data_deliverler.train.reindex(
        columns=desired_train.columns)
    data_deliverler.valid = data_deliverler.valid.reindex(
        columns=desired_valid.columns)
    pd.testing.assert_frame_equal(data_deliverler.train, desired_train)
    pd.testing.assert_frame_equal(data_deliverler.valid, desired_valid)

    # non time series
    path = "./test_data.csv"
    data_deliverler = datamodule.TornadoDataModule(path)
    correlated_data = create_correlated_data(10, False)
    correlated_data["y0_drop"] = (correlated_data["y0"] +
                                  np.random.random(10) * 0.5)
    train, valid = split_dataset(correlated_data)

    desired = correlated_data.copy()
    desired.drop(columns="y0_drop", inplace=True)
    desired_train, desired_valid = split_dataset(desired)

    data_deliverler.train = train
    data_deliverler.valid = valid
    data_deliverler.target = "target"
    data_deliverler.corr_based_removal()
    data_deliverler.train = data_deliverler.train.reindex(
        columns=desired_train.columns)
    data_deliverler.valid = data_deliverler.valid.reindex(
        columns=desired_valid.columns)
    pd.testing.assert_frame_equal(data_deliverler.train, desired_train)
    pd.testing.assert_frame_equal(data_deliverler.valid, desired_valid)


def test_vif_based_removal():
    # time series
    path = "./test_data.csv"
    data_deliverler = datamodule.TornadoDataModule(path)
    correlated_data = create_correlated_data(100, True)
    t = np.arange(0, 100 * 0.1, 0.1)
    correlated_data["y0_drop"] = (np.sin(2 * np.pi * t + 0.5) +
                                  np.random.random(100) * 0.1)
    train, valid = split_dataset(correlated_data)

    desired = correlated_data.copy()
    desired.drop(columns="y0_drop", inplace=True)
    desired_train, desired_valid = split_dataset(desired)

    data_deliverler.train = train
    data_deliverler.valid = valid
    data_deliverler.target = "target"
    data_deliverler.vif_based_removal()
    data_deliverler.train = data_deliverler.train.reindex(
        columns=desired_train.columns)
    data_deliverler.valid = data_deliverler.valid.reindex(
        columns=desired_valid.columns)
    pd.testing.assert_frame_equal(data_deliverler.train, desired_train)
    pd.testing.assert_frame_equal(data_deliverler.valid, desired_valid)

    # non time series
    path = "./test_data.csv"
    data_deliverler = datamodule.TornadoDataModule(path)
    correlated_data = create_correlated_data(100, False)
    t = np.arange(0, 100 * 0.1, 0.1)
    correlated_data["y0_drop"] = (np.sin(2 * np.pi * t + 0.5) +
                                  np.random.random(100) * 0.1)
    train, valid = split_dataset(correlated_data)

    desired = correlated_data.copy()
    desired.drop(columns="y0_drop", inplace=True)
    desired_train, desired_valid = split_dataset(desired)

    data_deliverler.train = train
    data_deliverler.valid = valid
    data_deliverler.target = "target"
    data_deliverler.vif_based_removal()
    data_deliverler.train = data_deliverler.train.reindex(
        columns=desired_train.columns)
    data_deliverler.valid = data_deliverler.valid.reindex(
        columns=desired_valid.columns)
    pd.testing.assert_frame_equal(data_deliverler.train, desired_train)
    pd.testing.assert_frame_equal(data_deliverler.valid, desired_valid)


def test_remove_features():
    # without preprocessing log
    path = "./test_data.csv"
    data_deliverler = datamodule.TornadoDataModule(path)
    correlated_data = create_correlated_data(100, True)
    t = np.arange(0, 100 * 0.1, 0.1)
    correlated_data["y0_drop"] = (np.sin(2 * np.pi * t + 0.5) +
                                  np.random.random(100) * 0.1)
    train, valid = split_dataset(correlated_data)

    desired = correlated_data.copy()
    desired.drop(columns="y0_drop", inplace=True)
    desired_train, desired_valid = split_dataset(desired)

    data_deliverler.train = train
    data_deliverler.valid = valid
    data_deliverler.target = "target"
    data_deliverler.features = []
    data_deliverler.remove_features()
    data_deliverler.train = data_deliverler.train.reindex(
        columns=desired_train.columns)
    data_deliverler.valid = data_deliverler.valid.reindex(
        columns=desired_valid.columns)
    pd.testing.assert_frame_equal(data_deliverler.train, desired_train)
    pd.testing.assert_frame_equal(data_deliverler.valid, desired_valid)

    # with preprocessing log
    path = "./test_data.csv"
    data_deliverler = datamodule.TornadoDataModule(path)

    correlated_data = create_correlated_data(100, True)
    correlated_data["y0_drop"] = np.random.random(100)
    train, valid = split_dataset(correlated_data)

    desired = correlated_data.copy()
    desired.drop(columns="y0_drop", inplace=True)
    desired_train, desired_valid = split_dataset(desired)

    data_deliverler.train = train
    data_deliverler.valid = valid
    data_deliverler.target = "target"
    data_deliverler.features = ["y0", "y1", "date", "target"]
    data_deliverler.remove_features()
    data_deliverler.train = data_deliverler.train.reindex(
        columns=desired_train.columns)
    data_deliverler.valid = data_deliverler.valid.reindex(
        columns=desired_valid.columns)
    pd.testing.assert_frame_equal(data_deliverler.train, desired_train)
    pd.testing.assert_frame_equal(data_deliverler.valid, desired_valid)


def test_generate(caplog):
    caplog.set_level(INFO)
    # without parameter
    path = "./test_data.csv"
    log_path = "./test_data.pickle"
    data_deliverler = datamodule.TornadoDataModule(path)
    correlated_data = create_correlated_data(100, True)
    t = np.arange(0, 100 * 0.1, 0.1)
    correlated_data["y0_drop"] = (np.sin(2 * np.pi * t + 0.5) +
                                  np.random.random(100) * 0.1)
    correlated_data.to_csv(path, index=False)

    desired_n_features_original = [4]
    desired_n_features_preprocessed = [30]
    desired_n_features_selected = [7, 8, 9, 10, 11]
    desired_n_features = [desired_n_features_original,
                          desired_n_features_preprocessed,
                          desired_n_features_selected]

    train, valid = data_deliverler.generate("target", True, 0.2, 0)

    for logger_name, log_level, message in caplog.record_tuples:
        if 'datamodule' in logger_name and "->" in message:
            n_features = re.findall(r"\d+", message)
            for i, n in enumerate(n_features):
                assert int(n) in desired_n_features[i]
    assert data_deliverler.log_path == log_path
    with open(log_path, "rb") as p:
        log = pickle.load(p)
    os.remove(path)
    os.remove(log_path)

    assert all(k in data_deliverler.preprocessors.keys()
               for k in log["preprocessors"].keys())
    assert all(k in log["preprocessors"].keys()
               for k in data_deliverler.preprocessors.keys())
    assert data_deliverler.features == log["features"]
