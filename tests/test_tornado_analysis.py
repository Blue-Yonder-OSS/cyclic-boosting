import numpy as np
import pandas as pd
from cyclic_boosting.tornado.core.analysis import TornadoAnalysisModule
import pytest

trend = "trend"
seasonal = "seasonality"
up_linear = "up-monotonicity_linearity"
down = "down-monotonicity"


@pytest.fixture(scope="module")
def prepare_time_series_dataset() -> tuple:

    np.random.seed(0)
    t = np.arange(1, 1000)
    daily_seasonality = 5 * np.sin(2 * np.pi * t / 24)
    weekly_seasonality = 5 * np.sin(2 * np.pi * t / (24 * 7))
    monthly_seasonality = 5 * np.sin(2 * np.pi * t / (24 * 30))

    ts = pd.date_range(start="2020-01-01", freq="D", periods=len(t))
    linear = np.arange(1, (len(ts) + 1), dtype=np.float64)
    down_monotonicity = linear[::-1].copy()
    daily_df = pd.DataFrame({"date": ts, seasonal: daily_seasonality, up_linear: linear, down: down_monotonicity})

    ts = pd.date_range(start="2020-01-01", freq="D", periods=len(t))
    weekly_df = pd.DataFrame({"date": ts, seasonal: weekly_seasonality, up_linear: linear, down: down_monotonicity})

    ts = pd.date_range(start="2020-01-01", freq="D", periods=len(t))
    monthly_df = pd.DataFrame({"date": ts, seasonal: monthly_seasonality, up_linear: linear, down: down_monotonicity})

    y = np.random.randn(len(t))
    ts = pd.date_range(start="2020-01-01", freq="D", periods=len(t))
    random_df = pd.DataFrame({"random": y, "date": ts})

    daily_df.iloc[10, 1] = np.nan
    weekly_df.iloc[10, 1] = np.nan
    monthly_df.iloc[10, 1] = np.nan

    return daily_df, weekly_df, monthly_df, random_df


def test_analysis_modules(prepare_time_series_dataset) -> tuple:
    daily, weekly, monthly, random = prepare_time_series_dataset

    func = {
        "has_trend": "check_trend",
        "has_seasonality": "check_seasonality",
        "has_up_monotonicity": "check_up_monotonicity",
        "has_down_monotonicity": "check_donw_monotonicity",
        "has_linearity": "check_linearity",
        "has_missing": "check_missing",
    }

    # daily dataset
    analyzer = TornadoAnalysisModule(dataset=daily, is_time_series=True, data_interval=None)
    report = analyzer.analyze()
    desired = {
        "has_trend": [seasonal, up_linear, down],
        "has_seasonality": [seasonal],
        "has_up_monotonicity": [up_linear],
        "has_down_monotonicity": [down],
        "has_linearity": [up_linear],
        "has_missing": [seasonal],
    }
    for k in desired.keys():
        if report[k] != desired[k]:
            alart = f"{func[k]}'s result is {report[k]}, expect {desired[k]}" " with daily time series data"
            assert False, alart

    # weekly dataset
    analyzer = TornadoAnalysisModule(dataset=weekly, is_time_series=True, data_interval=None)
    report = analyzer.analyze()
    desired = {
        "has_trend": [seasonal, up_linear, down],
        "has_seasonality": [seasonal],
        "has_up_monotonicity": [up_linear],
        "has_down_monotonicity": [down],
        "has_linearity": [up_linear],
        "has_missing": [seasonal],
    }
    for k in desired.keys():
        if report[k] != desired[k]:
            alart = f"{func[k]}'s result is {report[k]}, expect {desired[k]}" " with weekly time series data"
            assert False, alart

    # monthly dataset
    analyzer = TornadoAnalysisModule(dataset=monthly, is_time_series=True, data_interval=None)
    report = analyzer.analyze()
    desired = {
        "has_trend": [seasonal, up_linear, down],
        "has_seasonality": [seasonal],
        "has_up_monotonicity": [up_linear],
        "has_down_monotonicity": [down],
        "has_linearity": [up_linear],
        "has_missing": [seasonal],
    }
    for k in desired.keys():
        if report[k] != desired[k]:
            alart = f"{func[k]}'s result is {report[k]}, expect {desired[k]}" " with monthly time series data"
            assert False, alart

    analyzer = TornadoAnalysisModule(dataset=random, is_time_series=True, data_interval=None)
    report = analyzer.analyze()
    desired = {
        "has_trend": list(),
        "has_seasonality": list(),
        "has_up_monotonicity": list(),
        "has_down_monotonicity": list(),
        "has_linearity": list(),
        "has_missing": list(),
    }
    for key in desired.keys():
        if report[key] != desired[key]:
            assert False, "'check_missing' has something wrong, expect undetected"
