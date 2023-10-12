import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import MSTL
from scipy.fft import fft
from scipy.stats import norm


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
            self.calc_daily_ave()
            self.check_trend()
            self.check_monotonicity()
            self.check_seasonality()
            self.check_linearity()

        return self.report

    def calc_daily_ave(self) -> None:
        dates = np.sort(self.dataset.index.unique())
        size = (len(dates), len(self.dataset.columns))
        dataset = pd.DataFrame(np.zeros(size),
                               columns=self.dataset.columns,
                               index=dates)
        for date in dates:
            daily_ave = self.dataset[self.dataset.index == date].mean()
            dataset.loc[date, :] = daily_ave
        self.dataset = dataset.copy()

    def mann_kendall(self, data) -> bool:
        data = data.values[~np.isnan(data)].reshape(-1)
        n = len(data)

        # calculate S
        s = 0
        for k in range(n - 1):
            for j in range(k + 1, n):
                s += np.sign(data[j] - data[k])

        # calculate the unique data
        unique_data = np.unique(data)

        # calculate the var(s)
        tp = np.zeros(unique_data.shape)
        for i in range(len(unique_data)):
            tp[i] = np.sum(unique_data[i] == data)
        var_s = (n * (n - 1) * (2 * n + 5) +
                 np.sum(tp * (tp - 1) * (2 * tp + 5))) / 18
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s == 0:
            z = 0
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)

        # calculate the p_value
        p = 2 * (1 - norm.cdf(abs(z)))  # two tail test
        h = abs(z) > norm.ppf(1 - self.P_THRESH / 2)
        if (z != 0) and (p < self.P_THRESH) and h:
            return True
        else:
            return False

    def check_trend(self) -> None:
        # 帰無仮説：n個のサンプルx1,x2,...xnが独立で同一の確率分布に従う
        # 　　　　　つまり、トレンド性なし
        # P値が0.05を超えた場合、帰無仮説を棄却⇒トレンド性あり
        self.report['has_trend'] = []
        for col in self.targets:
            flag = self.mann_kendall(self.dataset[col])
            if flag:
                self.report['has_trend'].append(col)

    def fft(self, data):
        n = len(data)
        amp = fft(data.values)
        amp = 2.0/n * np.abs(amp[0:n//2])

        return amp

    def decompose(self, data) -> MSTL:
        decomposer = MSTL(data, periods=(7, 30, 356))
        res = decomposer.fit()

        return res

    def check_seasonality(self) -> None:
        for col in self.targets:
            is_seasonality = []
            decmp = self.decompose(self.dataset[col])
            amp_med = np.median(self.fft(self.dataset[col]))
            for _range in ['7', '30', '356']:
                amp_list = self.fft(decmp.seasonal[f'seasonal_{_range}'])
                amp_max = np.max(amp_list)
                peak = np.abs(amp_max-amp_med)/amp_med
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
