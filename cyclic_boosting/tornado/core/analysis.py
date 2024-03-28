"""Automatically analyze features and give flags."""

import logging

import numpy as np
import pandas as pd
import pymannkendall as mk
from scipy.fft import fft
from statsmodels.tsa.seasonal import MSTL

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(fmt="%(message)s"))
handler.terminator = ""
_logger.addHandler(handler)


class TornadoAnalysisModule:
    """TornadoAnalysisModule is a class for automatic feature analysis.

    This class automates the process of analyzing the characteristics of each
    feature in a dataset. It sets flags based on these characteristics, which
    are then used to apply appropriate treatments within a cyclic boosting
    process.

    Once the `analyze` function is executed, it extracts the features to be
    automatically analyzed, and then converts them into a time-wise average
    data using a "date" column. It analyzes the temporal characteristics of
    each feature.

    Parameters
    ----------
    dataset : pandas.DataFrame
        The dataset to be automatically analyzed for its features.

    is_time_series : bool
        Flag indicating whether the dataset is time-series data.

    data_interval : str
        The data collection interval which can be one of "hourly", "daily",
        "weekly", "monthly", or None. If None (default), the interval will be
        inferred and set automatically.

    Attributes
    ----------
    targets : list
        A list of feature names that are subject to automatic analysis.

    report : dict
        A dictionary with flag names as keys and lists of feature names that
        have been assigned these flags as values.

    amplitude : float
        The amplitude threshold for frequency components used in seasonality
        test. default is 5.0.

    corr_coefficient : float
        The correlation coefficient threshold used for linearity test. default
        is 0.95.

    Notes
    -----
    Only continuous features are targeted for automatic analysis. It is
    assumed that these continuous features are of float type.
    """

    def __init__(self, dataset, is_time_series=True, data_interval=None):
        self.dataset = dataset
        self.targets = list()
        self.report = {
            "is_unordered": list(),
            "is_continuous": list(),
            "has_trend": list(),
            "has_seasonality": list(),
            "has_up_monotonicity": list(),
            "has_down_monotonicity": list(),
            "has_linearity": list(),
            "has_missing": list(),
        }
        self.is_time_series = is_time_series
        self.amplitude = 5.0
        self.corr_coefficient = 0.95
        self.data_interval = data_interval

    def analyze(self) -> dict:
        """Run the automatic analysis.

        Performs automatic analysis on float type features within the dataset
        using the "date" column. Automatic analysis targets are float type
        features, treating them as continuous variables, and requires a "date"
        column to conduct time-based analysis. It will only execute if the
        "date" column exists.

        Returns
        -------
        dict
            A dictionary with flag names as keys and lists of feature names
            given those flags as values.
        """
        # self.gen_base_feature_property()
        self.int_or_float_feature_property()
        self.check_missing()

        cols = self.dataset.select_dtypes(include=["float"]).columns
        self.targets = [c for c in cols]
        targets = [c for c in cols]

        _logger.info(f"  Target features: {targets} \n")
        if self.is_time_series and "date" in self.dataset.columns:
            self.dataset.index = self.dataset["date"].values
            self.dataset = self.dataset[targets]
            self.calc_daily_average()
            self.check_data_interval()
            self.check_trend()
            self.check_monotonicity()
            self.check_seasonality()
            self.check_linearity()
        elif self.is_time_series and "date" not in self.dataset.columns:
            raise ValueError(
                "Dataset must have 'date' column on "
                "time-series prediction."
                "if not, set 'False' to 'is_time_series' option"
                "at Manager"
            )
        else:
            self.dataset = self.dataset[targets]

        return self.report

    def gen_base_feature_property(self) -> None:
        """Ask the user to set if each feature is categorical or continuous.

        Ask the user the property of each feature and flag it as either
        categorical or continuous, depending on the input. The input is "cat"
        for categorical features and "con" for continuous features.
        """
        cols = self.dataset.select_dtypes(include=["int", "float", "object"])
        _logger.info("is a categorical or continuous variable?")
        for col in cols:
            fp_str = input(f"please enter {col} is [cat/con] ")
            if fp_str == "cat":
                self.report["is_unordered"].append(col)
            elif fp_str == "con":
                self.report["is_continuous"].append(col)
            else:
                raise ValueError("please type 'cat' or 'con'")

    # FIXME
    def int_or_float_feature_property(self) -> None:
        """Set each feature as categorical or continuous based on data type.

        Assign a flag of categorical or continuous feature based on the data
        type of each feature. Assign the categorical feature flag to features
        of type int, and the continuous feature flag to features of type
        float.
        """
        cols = self.dataset.select_dtypes(include=["int", "float"])
        for col in cols:
            if isinstance(self.dataset[col][0], np.int64):
                self.report["is_unordered"].append(col)
            elif isinstance(self.dataset[col][0], np.float64):
                self.report["is_continuous"].append(col)

    def calc_daily_average(self) -> None:
        """Average features in the dataset by time.

        This function takes a dataset with a datetime index, computes the mean
        of features for each time, and returns the dataset sorted by the
        datetime index.
        """
        self.dataset = self.dataset.groupby(level=0).mean()
        self.dataset = self.dataset.sort_index()

    def check_data_interval(self):
        """Set the data interval and interpolate missing time data points.

        If `data_interval` is not given in advance, the minimum data interval
        is taken from the time-averaged data and set as data_interval. It then
        performs linear interpolation to fill in missing time data points for
        accurate feature analysis.
        """
        if self.data_interval is None:
            diff = self.dataset.index.to_series().diff()
            interval = diff.min()
        elif self.data_interval == "monthly":
            interval = pd.Timedelta(days=30)
        elif self.data_interval == "weekly":
            interval = pd.Timedelta(days=7)
        elif self.data_interval == "daily":
            interval = pd.Timedelta(days=1)
        elif self.data_interval == "hourly":
            interval = pd.Timedelta(hours=1)
        else:
            raise ValueError(
                "data_interval must be 'monthly', 'weekly', \
                             'daily', or 'hourly'."
            )

        if interval.days in [28, 29, 30, 31]:
            if self.dataset.index.to_series().dt.day.mode()[0] == 1:
                data_interval = "MS"
            elif self.dataset.index.to_series().dt.day.mode()[0] >= 28:
                data_interval = "M"
            self.data_interval = "monthly"
        elif interval.days == 7:
            dayofweek = ["W-MON", "W-TUE", "W-WED", "W-THU", "W-FRI", "W-SAT", "W-SUN"]
            data_interval = dayofweek[self.dataset.index.to_series().dt.dayofweek.mode()[0]]
            self.data_interval = "weekly"
        elif interval.days == 1:
            data_interval = "D"
            self.data_interval = "daily"
        elif interval.seconds == 3600:
            data_interval = "H"
            self.data_interval = "hourly"

        _logger.info(f"  Data interval is '{self.data_interval}'\n")
        _logger.info("  if it is not correct, set 'data_interval' option at TornadoDataModule\n")

        if self.data_interval in ["monthly", "weekly", "daily"]:
            self.dataset.index = [i.date() for i in self.dataset.index]
        self.dataset = self.dataset.asfreq(freq=data_interval)
        self.dataset = self.dataset.interpolate(method="linear", limit_direction="both")

    def check_trend(self) -> None:
        """Detect trends in features.

        This function uses the Mann-Kendall trend test to assess each feature
        for the presence of a trend.
        """
        self.report["has_trend"] = []
        for col in self.targets:
            decmp = self.decompose(self.dataset[col])
            decmp_trend = pd.DataFrame(decmp.trend)
            flag = mk.original_test(decmp_trend)[0] != "no trend"
            if flag:
                self.report["has_trend"].append(col)

    def fft(self, data):
        """Perform FFT (Fast Fourier Transform) analysis on time-series data.

        Parameters
        ----------
        data : pandas.Series
            The time-series data to be analyzed.

        Returns
        -------
        numpy.ndarray
            The result of the FFT analysis.
        """
        n = len(data)
        amp = fft(data.values)
        amp = (2.0 / n) * (np.abs(amp[0 : n // 2]))

        return amp

    def decompose(self, data) -> MSTL:
        """Decompose seasonal and trend components of time-series data.

        The function employs the "Multiple Seasonal-Trend decomposition using
        LOESS" (MSTL) to decompose time-series data into its seasonal and
        trend components.

        Parameters
        ----------
        data : pandas.Series
            The time-series data to be decomposed.

        Returns
        -------
        statsmodels.tsa.seasonal.DecomposeResult
            The decomposition result containing seasonal and trend components.
        """
        if self.data_interval == "monthly":
            periods = 12
        elif self.data_interval == "weekly":
            periods = (4, 52)
        elif self.data_interval == "daily":
            periods = (7, 30, 365)
        elif self.data_interval == "hourly":
            periods = (24, 24 * 7, 24 * 30, 24 * 365)
        else:
            _logger.warning(
                "Seasonality detection is only supported for\n" "    hourly, daily, weekly, and monthly data."
            )
        decomposer = MSTL(data, periods=periods)
        res = decomposer.fit()

        return res

    def check_seasonality(self) -> None:
        """Detect seasonality in features.

        Decompose a feature using Multiple Seasonal-Trend decomposition using
        LOESS (MSTL) and detect seasonality by comparing the FFT amplitude of
        each seasonal component data with the FFT amplitude of the original
        data (considered as noise).
        """
        for col in self.targets:
            is_seasonality = []
            decmp = self.decompose(self.dataset[col])
            decmp_seasonal = pd.DataFrame(decmp.seasonal)
            amp_med = np.median(self.fft(self.dataset[col]))
            for _range in decmp_seasonal:
                amp_list = self.fft(decmp_seasonal[_range])
                amp_max = np.max(amp_list)
                peak = np.abs(amp_max - amp_med) / amp_med
                if peak > self.amplitude:
                    is_seasonality.append(True)
                else:
                    is_seasonality.append(False)
            if np.any(is_seasonality):
                self.report["has_seasonality"].append(col)

    def check_monotonicity(self) -> None:
        """Detect monotonicity in features.

        Determine whether the feature is monotonically increasing,
        monotonically decreasing, or non-monotonic.
        """
        for col in self.targets:
            diff = self.dataset[col].diff().dropna()
            if np.all(diff.values > 0):
                self.report["has_up_monotonicity"].append(col)
            elif np.all(diff.values < 0):
                self.report["has_down_monotonicity"].append(col)

    def check_linearity(self) -> None:
        """Detect linearity in features.

        Determine linearity by the correlation coefficient of each feature
        with time.
        """
        # NOTE: very rough
        for col in self.targets:
            corr = np.corrcoef(np.arange(len(self.dataset[col])), self.dataset[col].values)
            if corr[0, 1] > self.corr_coefficient:
                self.report["has_linearity"].append(col)

    def check_missing(self) -> None:
        """Determine the presence of missing data."""
        count = self.dataset.isnull().sum()
        missing = count[count > 0]
        self.report["has_missing"] = [i for i in missing.index]
