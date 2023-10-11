import pandas as pd
import numpy as np
import pytest

from sklearn.preprocessing import OrdinalEncoder
from cyclic_boosting import flags
from typing import List, Tuple, Union

import os


@pytest.fixture(scope="session")
def prepare_data() -> Tuple[pd.DataFrame, np.ndarray]:
    current_path = os.path.dirname(os.path.realpath(__file__))
    filename = "integration_test_data.csv"
    df = pd.read_csv(current_path + "/" + filename)

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


@pytest.fixture(scope="session")
def feature_properties() -> dict:
    fp = {}
    fp["P_ID"] = flags.IS_UNORDERED
    fp["PG_ID_3"] = flags.IS_UNORDERED
    fp["L_ID"] = flags.IS_UNORDERED
    fp["dayofweek"] = flags.IS_ORDERED
    fp["dayofyear"] = flags.IS_CONTINUOUS | flags.IS_LINEAR
    fp["price_ratio"] = flags.IS_CONTINUOUS | flags.HAS_MISSING | flags.MISSING_NOT_LEARNED
    fp["PROMOTION_TYPE"] = flags.IS_ORDERED
    return fp


@pytest.fixture(scope="session")
def features() -> List[Union[str, tuple]]:
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


@pytest.fixture(scope="session")
def default_features() -> List[str]:
    return ["dayofweek", "L_ID", "PG_ID_3", "P_ID", "PROMOTION_TYPE", "price_ratio", "dayofyear"]


@pytest.fixture(scope="session")
def is_plot() -> bool:
    return False


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


@pytest.fixture(scope="session")
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
