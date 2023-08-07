import numpy as np
from statsmodels.tsa.seasonal import MSTL
from statsmodels.tsa.stattools import adfuller
from scipy.fft import fft
from scipy.signal import find_peaks


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
        self.W_THRESH = 1.5
        self.C_THRESH = 0.95

    def analyze(self) -> dict:
        if self.is_time_series:
            if 'date' not in self.dataset.columns:
                raise ValueError("Dataset must be included 'date' column on \
                                 time-series prediction")
            cols = self.dataset.select_dtypes(include='float').columns
            self.targets = [c for c in cols]
            targets = [c for c in cols]
            print(f"Auto analysis target {targets}")
            targets.append('date')
            dataset = self.dataset[targets]
            dataset = dataset.sort_values('date')
            # self.check_trend()
            self.check_monotonicity()
            self.check_seasonality()
            self.check_linearity()
            self.check_missing()

        return self.report

    def adf(self, data) -> list:
        ps = []
        for reg in ['n', 'c', 'ct', 'ctt']:
            _, pvalue, _, _, _, _ = adfuller(x=data,
                                             regression=reg,
                                             autolag='AIC')
            ps.append(pvalue)

        return ps

    def check_trend(self) -> None:
        # P値が0.05を超えた場合、「データが定常性をもつ＝トレンドをもたない」
        # という帰無仮説を棄却

        self.report['has_trend'] = []
        for col in self.targets:
            ps = self.adf(self.dataset[col])
            flag = np.all(np.array(ps) > self.P_THRESH)
            if flag:
                self.report['has_trend'].append(col)

    def fft(self, data) -> list:
        amp = fft(data.values, 1)
        peaks, _ = find_peaks(amp, prominence=0.5)
        if len(peaks) >= 1:
            peaks = sorted(peaks)
            duration = [peaks[i+1] - peaks[i] for i in range(0, len(peaks)-1)]
            return duration
        else:
            return []

    def decompose(self, data) -> MSTL:
        decomposer = MSTL(data, periods=(7, 30, 356))
        res = decomposer.fit()

        return res

    def check_seasonality(self) -> None:
        for col in self.targets:
            is_seasonality = []
            decmp = self.decompose(self.dataset[col])
            for _range in ['7', '30', '356']:
                duration = self.fft(decmp.seasonal[f'seasonal_{_range}'])
                if len(duration) > 0:
                    if np.std(duration) < self.W_THRESH:
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
            corr = self.dataset[['date', col]].corr()
            if corr.iloc[0, 1] > self.C_THRESH:
                self.report['has_linearity'].append(col)

    def check_missing(self) -> None:
        count = self.dataset.isnull().sum()
        missing = count[count > 0]
        self.report['has_missing'] = [i for i in missing.index]
