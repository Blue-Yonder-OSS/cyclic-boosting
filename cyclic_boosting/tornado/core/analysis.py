import numpy as np
from statsmodels.tsa.seasonal import MSTL
from scipy.fft import fft
import pymannkendall as mk


class TornadoAnalysisModule():
    def __init__(self, dataset, is_time_series=True):
        # super().__init__()
        # データセットはカテゴリ変数がint, 量的変数がfloatという前提をおく
        self.dataset = dataset
        self.targets = []
        self.report = {
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

    def analyze(self) -> dict:
        if self.is_time_series:
            if 'date' not in self.dataset.columns:
                raise ValueError("Dataset must be included 'date' column on \
                                 time-series prediction")
            cols = self.dataset.select_dtypes(include=['float']).columns
            self.targets = [c for c in cols]
            targets = [c for c in cols]
            print(f"Auto analysis target {targets}")
            self.dataset.index = self.dataset["date"].values
            self.dataset = self.dataset[targets]
            self.check_missing()
            self.calc_daily_average()
            self.check_trend()
            self.check_monotonicity()
            self.check_seasonality()
            self.check_linearity()

        return self.report

    def calc_daily_average(self) -> None:
        self.dataset = self.dataset.groupby(level=0).mean()
        self.dataset = self.dataset.sort_index()

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
        decomposer = MSTL(data, periods=(7, 30, 365))
        res = decomposer.fit()

        return res

    def check_seasonality(self) -> None:
        for col in self.targets:
            is_seasonality = []
            decmp = self.decompose(self.dataset[col])
            amp_med = np.median(self.fft(self.dataset[col]))
            for _range in ['7', '30', '365']:
                amp_list = self.fft(decmp.seasonal[f'seasonal_{_range}'])
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
