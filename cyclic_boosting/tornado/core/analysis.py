import logging

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import MSTL
from scipy.fft import fft
import pymannkendall as mk
import matplotlib.pyplot as plt

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(fmt="%(message)s"))
handler.terminator = ''
_logger.addHandler(handler)


class TornadoAnalysisModule():
    def __init__(self, dataset, is_time_series=True,
                 data_interval=None):
        # super().__init__()
        # データセットはカテゴリ変数がint, 量的変数がfloatという前提をおく
        self.dataset = dataset
        self.targets = []
        self.report = {
            'is_unordered': [],
            'is_continuous': [],
            'has_trend': [],
            'has_seasonality': [],
            'has_up_monotonicity': [],
            'has_down_monotonicity': [],
            'has_linearity': [],
            'has_missing': []
        }
        self.is_time_series = is_time_series
        self.P_THRESH = 0.05
        self.W_THRESH = 5.0
        self.C_THRESH = 0.95
        self.data_interval = data_interval

    def analyze(self) -> dict:
        # self.gen_base_feature_property()
        self.int_or_float_feature_property()
        self.check_missing()

        cols = self.dataset.select_dtypes(include=['float']).columns
        self.targets = [c for c in cols]
        targets = [c for c in cols]
        _logger.info(f"Auto analysis target {targets}")
        if 'date' in self.dataset.columns:
            self.dataset.index = self.dataset["date"].values
            self.dataset = self.dataset[targets]
            self.calc_daily_average()
            self.check_data_interval()
            self.check_trend()
            self.check_monotonicity()
            self.check_seasonality()
            self.check_linearity()
        else:
            if self.is_time_series:
                raise ValueError("Dataset must have 'date' column on "
                                 "time-series prediction")
            self.dataset = self.dataset[targets]

        return self.report

    def gen_base_feature_property(self) -> None:
        cols = self.dataset.select_dtypes(include=['int', 'float', 'object'])
        _logger.info('is a categorical or continuous variable?')
        for col in cols:
            fp_str = input(f"please enter {col} is [cat/con] ")
            if fp_str == 'cat':
                self.report['is_unordered'].append(col)
            elif fp_str == 'con':
                self.report['is_continuous'].append(col)
            else:
                raise ValueError("please type 'cat' or 'con'")

    # FIXME
    # この関数は入力の手間をなくすためだけのものであり本質的に自動化を行っているわけではない.修正の必要あり
    def int_or_float_feature_property(self) -> None:
        cols = self.dataset.select_dtypes(include=['int', 'float', 'object'])
        for col in cols:
            if isinstance(self.dataset[col][0], np.int64):
                self.report['is_unordered'].append(col)
            elif isinstance(self.dataset[col][0], np.float64):
                self.report['is_continuous'].append(col)
            else:
                raise ValueError("整数または小数ではない")

    def calc_daily_average(self) -> None:
        self.dataset = self.dataset.groupby(level=0).mean()
        self.dataset = self.dataset.sort_index()

    def check_data_interval(self):
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
            raise ValueError("data_interval must be 'monthly', 'weekly', \
                             'daily', or 'hourly'.")

        if interval.days in [28, 29, 30, 31]:
            if self.dataset.index.to_series().dt.day.mode()[0] == 1:
                data_interval = "MS"
            elif self.dataset.index.to_series().dt.day.mode()[0] >= 28:
                data_interval = "M"
            self.data_interval = "monthly"
        elif interval.days == 7:
            dayofweek = [
                "W-MON", "W-TUE", "W-WED", "W-THU", "W-FRI", "W-SAT", "W-SUN"]
            data_interval = dayofweek[self.dataset.index.to_series().dt.
                                      dayofweek.mode()[0]]
            self.data_interval = "weekly"
        elif interval.days == 1:
            data_interval = "D"
            self.data_interval = "daily"
        elif interval.seconds == 3600:
            data_interval = "H"
            self.data_interval = "hourly"
        _logger.info(f"Data interval is '{self.data_interval}'. If not, give\n"
                     "    the data_interval option in the TornadoDataModule.")
        if self.data_interval in ["monthly", "weekly", "daily"]:
            self.dataset.index = [i.date() for i in self.dataset.index]
        self.dataset = self.dataset.asfreq(freq=data_interval)
        self.dataset = self.dataset.interpolate(method='linear',
                                                limit_direction='both')

    def check_trend(self) -> None:
        # 帰無仮説：n個のサンプルx1,x2,...xnが独立で同一の確率分布に従う
        # 　　　　　つまり、トレンド性なし
        # P値が0.05を超えた場合、帰無仮説を棄却⇒トレンド性あり
        self.report['has_trend'] = []
        for col in self.targets:
            flag = mk.original_test(self.dataset[col])[0] != "no trend"
            if flag:
                self.report['has_trend'].append(col)

    def fft(self, data):
        n = len(data)
        amp = fft(data.values)
        amp = (2.0 / n) * (np.abs(amp[0:n // 2]))

        return amp

    def decompose(self, data) -> MSTL:
        if self.data_interval == "monthly":
            periods = (12)
        elif self.data_interval == "weekly":
            periods = (4, 52)
        elif self.data_interval == "daily":
            periods = (7, 30, 365)
        elif self.data_interval == "hourly":
            periods = (24, 24*7, 24*30, 24*365)
        else:
            _logger.warning("Seasonality detection is only supported for\n"
                            "    hourly, daily, weekly, and monthly data.")
        decomposer = MSTL(data, periods=periods)
        res = decomposer.fit()

        return res

    def check_seasonality(self) -> None:
        for col in self.targets:
            is_seasonality = []
            decmp = self.decompose(self.dataset[col])
            decmp_seasonal = pd.DataFrame(decmp.seasonal)
            amp_med = np.median(self.fft(self.dataset[col]))
            for _range in decmp_seasonal:
                amp_list = self.fft(decmp_seasonal[_range])
                amp_max = np.max(amp_list)
                peak = np.abs(amp_max - amp_med) / amp_med
                if peak > self.W_THRESH:
                    is_seasonality.append(True)
                else:
                    is_seasonality.append(False)
            if np.any(is_seasonality):
                self.report['has_seasonality'].append(col)

    def check_monotonicity(self) -> None:
        for col in self.targets:
            diff = self.dataset[col].diff().dropna()
            if np.all(diff.values > 0):
                self.report['has_up_monotonicity'].append(col)
            elif np.all(diff.values < 0):
                self.report['has_down_monotonicity'].append(col)

    def check_linearity(self) -> None:
        # NOTE: very rough
        for col in self.targets:
            corr = np.corrcoef(np.arange(len(self.dataset[col])),
                               self.dataset[col].values)
            if corr[0, 1] > self.C_THRESH:
                self.report['has_linearity'].append(col)

    def check_missing(self) -> None:
        count = self.dataset.isnull().sum()
        missing = count[count > 0]
        self.report['has_missing'] = [i for i in missing.index]
