import datetime
import numpy as np
import pandas as pd
from itertools import combinations
from cyclic_boosting.tornado.core.manager import (
    TornadoManager,
    ForwardSelectionManager,
    PriorPredForwardSelectionManager,
)
from cyclic_boosting.smoothing.onedim import SeasonalSmoother, IsotonicRegressor
import pytest


@pytest.fixture(scope="module")
def manager_base_params():
    params = {
        "mfp": None,
        "is_ts": True,
        "data_interval": None,
        "task": "regression",
        "dist": "poisson",
        "model": "multiplicative",
        "X": None,
        "y": None,
        "target": None,
        "regressor": None,
        "features": list(),
        "interaction_term": list(),
        "feature_properties": dict(),
        "smoothers": dict(),
        "observers": dict(),
        "report": dict(),
        "init_model_attr": dict(),
        "model_params": dict(),
        "max_interaction": 0,
        "experiment": 0,
        "end": 0,
    }
    return params


def test_tornado_manager(prepare_data, manager_base_params):
    X, y = prepare_data
    X, y = X[:100].copy(), y[:100].copy()
    X.loc[:, "SALES"] = y.copy()
    names = {k: k.lower() for k in X.columns}
    X.rename(names, axis=1, inplace=True)

    length = 10
    data = np.arange(length).astype(np.int64)
    start_date = datetime.datetime.today().date()
    dates = pd.date_range(start=start_date, periods=length, freq="D")
    X_dummy = pd.DataFrame(
        {"int": data, "float": data.astype(np.float64), "object": data.astype(object), "date": dates}
    )
    manager = TornadoManager(is_time_series=True)

    # get_attr
    params = manager.get_attr()
    for p, desired in zip(params.items(), manager.get_attr().items()):
        assert p[1] == desired[1], p[0]

    # set_attr
    manager_base_params.update({"end": 10000, "X": X_dummy})
    manager.set_attr(manager_base_params)
    params = manager.get_attr()
    assert params["end"] == manager_base_params["end"]

    # feature_properties
    attr = "feature_properties"
    manager_base_params.update({"end": 0, "X": X})
    manager.set_attr(manager_base_params)
    manager.set_feature_property()

    # set_feature
    func = "set_feature"
    attr = "features"
    desired = ["int", "float", "object"]
    manager_base_params.update({"interaction_term": [("a", "b")]})
    manager.set_attr(manager_base_params)
    manager.set_feature(desired)
    desired.append(("a", "b"))
    res = manager.get_attr()
    is_same = res[attr] == desired
    assert is_same, f"{func}'s result is {res[attr]}, {desired}"

    # set_interaction_term
    func = "set_interaction"
    attr = "interaction_term"
    desired = [x for x in combinations(X_dummy.columns, 2)]
    manager_base_params.update(
        {"init_model_attr": {"features": list(X_dummy.columns)}, "X": X_dummy, "interaction_term": list()}
    )
    manager.set_attr(manager_base_params)
    manager.set_interaction()
    res = manager.get_attr()
    is_same = res[attr] == desired
    assert is_same, f"{func}'s result is {res[attr]}, {desired}"

    # set_smoother
    func = "set_smoother"
    attr = "smoothers"
    report_dummy = {"has_seasonality": ["seasonal"], "has_up_monotonicity": ["up"], "has_down_monotonicity": ["down"]}
    manager_base_params.update({"report": report_dummy})
    manager.set_attr(manager_base_params)
    manager.set_smoother()
    res = list(manager.get_attr()[attr].values())
    assert isinstance(res[0], SeasonalSmoother)
    assert isinstance(res[1], IsotonicRegressor)
    assert isinstance(res[2], IsotonicRegressor)

    # set observer
    manager.set_observer()

    # drop_unused_features
    func = "drop_unused_features"
    attr = "X"
    param = {
        "init_model_attr": {"X": X_dummy},
        "feature_properties": {"int": 1},
    }
    manager.set_attr(param)
    manager.drop_unused_features()
    res = manager.get_attr()
    desired = ["int"]
    is_same = list(res[attr].columns) == desired
    assert is_same, f"{func}'s result is {res[attr].columns}, {desired}"

    # clear
    func = "clear"
    manager.clear()
    assert isinstance(manager.features, list)
    assert isinstance(manager.feature_properties, dict)
    assert isinstance(manager.smoothers, dict)
    assert isinstance(manager.observers, dict)

    # basic steps
    manager.init(X.copy(), "SALES")
    manager.update()
    manager.manage()


def test_forward_selection_manager(prepare_data):
    X, y = prepare_data
    X, y = X[:100].copy(), y[:100].copy()
    X.loc[:, "SALES"] = y.copy()
    names = {k: k.lower() for k in X.columns}
    X.rename(names, axis=1, inplace=True)
    manager = ForwardSelectionManager(is_time_series=True)

    # init
    manager.init(X.copy(), "SALES")

    # feature
    desired = ["dummy"]
    manager.set_feature(desired)
    assert manager.features == desired

    # manage
    manager.manage()

    # update
    manager.update()


def test_prior_pred_forward_selection_manager(prepare_data):
    X, y = prepare_data
    X, y = X[:100].copy(), y[:100].copy()
    X.loc[:, "SALES"] = y
    names = {k: k.lower() for k in X.columns}
    X.rename(names, axis=1, inplace=True)
    manager = PriorPredForwardSelectionManager(is_time_series=True)

    # init
    manager.init(X.copy(), "SALES")

    # feature
    desired = ["dummy"]
    manager.set_feature(desired)
    assert manager.features == desired

    # manage
    manager.manage()

    # update
    manager.update()
