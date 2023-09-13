# Add some comments
#
#
#

import abc
import six
import copy
import itertools
import pandas as pd
import numpy as np

from cyclic_boosting import flags, common_smoothers, observers, binning
from cyclic_boosting.plots import plot_analysis
from cyclic_boosting.pipelines import pipeline_CBPoissonRegressor

from cyclic_boosting.smoothing.onedim import SeasonalSmoother, IsotonicRegressor
from .analysis import TornadoAnalysisModule


# @six.add_metaclass(abc.ABCMeta)
class TornadoModuleForVariableSelection():
    # some comments

    def __init__(self, manual_feature_property=None,
                 is_time_series=True) -> None:
        # super().__init__()
        self.X = None
        self.y = None
        self.target = None
        self.mfp = manual_feature_property
        self.is_ts = is_time_series
        self.feature_properties = {}
        self.features = []
        self.next_features = []
        self.sorted_features = []
        self.interaction_term = []
        self.smoothers = {}
        self.observers = {}
        self.CB_pipline = {}
        self.report = {}
        self.type = "single"
        self.max_interaction = 0
        self.experiment = -1
    
    def int_or_float_feature_property(self) -> None:
        cols = self.X.select_dtypes(include=['int', 'float', 'object'])
        for col in cols:
            if type(self.X[col][0]) == np.int64:
                self.feature_properties[col] = flags.IS_UNORDERED
            elif type(self.X[col][0]) == np.float64:
                self.feature_properties[col] = flags.IS_CONTINUOUS
            else:
                raise ValueError("整数または小数ではない")

    def set_feature_property(self) -> None:

        # カテゴリ変数の尺度の違い、質的変数が季節性をもつかなどを自動で抽出するのが難しい
        # 質的変数、量的変数、欠損値の有無などは自動で抽出して、他は間違いを許容する
        # datetimeなど、モデルが処理できない変数はおとす

        if self.mfp is None:
            self.int_or_float_feature_property()
            analyzer = TornadoAnalysisModule(self.X, is_time_series=self.is_ts)
            self.report = analyzer.analyze()
            for key, cols in self.report.items():
                if key == 'has_seasonality':
                    flag = flags.IS_SEASONAL
                elif key == 'has_linearity':
                    flag = flags.IS_LINEAR
                elif key == 'has_up_monotonicity':
                    flag = flags.IS_MONOTONIC
                elif key == 'has_down_monotonicity':
                    flag = flags.IS_MONOTONIC
                elif key == 'has_missing':
                    flag = flags.HAS_MISSING
                else:
                    continue

                for col in cols:
                    self.feature_properties[col] |= flag

        else:
            self.feature_properties = self.mfp

        print(self.report)

    def set_feature(self) -> None:

        if self.type == "single":
            #TODO
            #ここで毎回all_featuresを作っているところは修正したい
            all_features = []
            for feature in self.feature_properties.keys():
                all_features.append(feature)
            for term in self.interaction_term:
                all_features.append(term)
            self.features = [all_features[self.experiment]]
        if self.type == "multiple":
            if self.experiment == 0:
                self.features = [list(self.sorted_features.keys())[0]]
            else:
                self.features = self.next_features
                print(f"features: {self.features}")
            pass

    def get_features(self, features):
        #ここで次に動かすfeatureを書きたい。もしかしたら最初とそれ以降で場合分けがいるかも
        self.next_features = features
        pass


    def create_interaction_term(self, size=2) -> None:
        if size <= 1:
            raise ValueError("interaction size must be more than 2")
        elif size >= 3:
            print("WARNING: many interaction terms might cause long training")

        # interaction term
        # combination = []
        for s in range(2, size+1):
            comb = itertools.combinations(self.feature_properties.keys(), s)
            # combination += [c for c in comb]
            self.interaction_term += [c for c in comb]

        self.max_interaction = len(self.feature_properties.keys()) + len(self.interaction_term)


    def set_smoother(self) -> None:
        smoothers = {}
        for key, cols in self.report.items():
            if len(cols) > 0:
                if key == 'has_seasonality':
                    for col in cols:
                        smoothers[(col,)] = SeasonalSmoother(order=3)
                elif key == 'has_up_monotonicity':
                    for col in cols:
                        smoothers[(col,)] = IsotonicRegressor(increasing=True)
                elif key == 'has_down_monotonicity':
                    for col in cols:
                        smoothers[(col,)] = IsotonicRegressor(increasing=False)
        self.smoothers = smoothers

    def set_observer(self) -> None:
        self.observers = [
            observers.PlottingObserver(iteration=1),
            observers.PlottingObserver(iteration=-1),
        ]

    def init(self, dataset, target) -> None:
        self.target = target.lower()
        self.y = copy.deepcopy(np.asarray(dataset[self.target]))
        self.X = copy.deepcopy(dataset.drop(self.target, axis=1))
        if not self.is_ts:
            self.X = self.X.drop('date', axis=1)
        self.set_feature_property()
        self.create_interaction_term()

    def reset(self, dataset, target) -> None:
        self.y = copy.deepcopy(np.asarray(dataset[self.target]))
        self.X = copy.deepcopy(dataset.drop(self.target, axis=1))
        if not self.is_ts:
            self.X = self.X.drop('date', axis=1)

    def update(self) -> None:
        self.set_feature()
        self.set_smoother()
        self.set_observer()

    def clear(self) -> None:
        self.features = []
        self.smoothers = {}
        self.observers = {}

    # @abc.abstractmethod
    def build(self) -> pipeline_CBPoissonRegressor:

        # タスクによってbuildするモデルを切り変える
        self.CB_pipeline = pipeline_CBPoissonRegressor(
            feature_properties=self.feature_properties,
            feature_groups=self.features,
            observers=self.observers,
            maximal_iterations=50,
            smoother_choice=common_smoothers.SmootherChoiceGroupBy(
                use_regression_type=True,
                use_normalization=False,
                explicit_smoothers=self.smoothers,
            ),
        )

        return self.CB_pipeline

    def manage(self) -> bool:

        # 交互作用項の設定、平滑化関数の設定の変更を反映
        if self.experiment != self.max_interaction - 1:
            # 設定変更するコードを起動する
            self.experiment += 1
            self.clear()
            self.update()
            return True
        else:
            return False
    
    def set_to_multiple(self, features):
        self.sorted_features = features
        self.max_interaction = len(features)
        self.type = "multiple"
        self.experiment = -1
        self.features = [list(self.sorted_features.keys())[0]]
    
    def manage_selection(self) -> bool:
        #TODO
        #基本的にmanageと同じ機能を実装するが、sorted_CODsを受け取らないといけなさそう
        if self.experiment != self.max_interaction - 1:
            # 設定変更するコードを起動する
            self.experiment += 1
            self.clear()
            self.update_for_selection()
            return True
        else:
            return False
        pass

