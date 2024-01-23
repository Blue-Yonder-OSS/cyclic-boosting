# Add some comments
#
#
#
import logging

import abc
import six
import itertools
import numpy as np

from cyclic_boosting import flags, common_smoothers, observers

# from cyclic_boosting.plots import plot_analysis
from cyclic_boosting.pipelines import (
    pipeline_CBPoissonRegressor as PoissonRegressor,
    pipeline_CBNBinomRegressor as NBinomRegressor,
    pipeline_CBNBinomC as NBinomC,
    pipeline_CBMultiplicativeQuantileRegressor as MultiplicativeQuantileRegressor,
    pipeline_CBAdditiveQuantileRegressor as AdditiveQuantileRegressor,
)

from cyclic_boosting.smoothing.onedim import SeasonalSmoother, IsotonicRegressor
from .analysis import TornadoAnalysisModule

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(fmt="%(message)s"))
handler.terminator = ""
_logger.addHandler(handler)


@six.add_metaclass(abc.ABCMeta)
class TornadoModuleBase:
    def __init__(
        self,
        manual_feature_property=None,
        is_time_series=True,
        data_interval=None,
        task="regression",
        dist="poisson",
        model="multiplicative",
    ) -> None:
        self.mfp = manual_feature_property
        self.is_ts = is_time_series
        self.data_interval = data_interval
        self.task = task
        self.dist = dist
        self.model = model
        self.X = None
        self.y = None
        self.target = None
        self.regressor = None
        self.features = list()
        self.interaction_term = list()
        self.feature_properties = dict()
        self.smoothers = dict()
        self.observers = dict()
        self.report = dict()
        self.init_model_attr = dict()
        self.model_params = dict()
        self.max_interaction = 0
        self.experiment = 0
        self.end = 0

    def get_params(self) -> dict:
        class_vars = dict()
        for attr_name, value in self.__dict__.items():
            if not callable(value) and not attr_name.startswith("__"):
                class_vars[attr_name] = value

        return class_vars

    def set_params(self, params: dict) -> None:
        class_vars = self.__dict__
        for attr_name, value in params.items():
            class_vars[attr_name] = value

    def gen_base_feature_property(self) -> None:
        cols = self.X.select_dtypes(include=["int", "float", "object"])
        print("is a categorical or continuous variable?")
        for col in cols:
            fp_str = input(f"please enter {col} is [cat/con] ")
            if fp_str == "cat":
                self.feature_properties[col] = flags.IS_UNORDERED
            elif fp_str == "con":
                self.feature_properties[col] = flags.IS_CONTINUOUS
            else:
                raise ValueError("please type 'cat' or 'con'")

    # FIXME
    # この関数は入力の手間をなくすためだけのものであり本質的に自動化を行っているわけではない.修正の必要あり
    def int_or_float_feature_property(self) -> None:
        cols = self.X.select_dtypes(include=["int", "float", "object"])
        for col in cols:
            if isinstance(self.X[col][0], np.int64):
                self.feature_properties[col] = flags.IS_UNORDERED
            elif isinstance(self.X[col][0], np.float64):
                self.feature_properties[col] = flags.IS_CONTINUOUS
            else:
                raise ValueError("整数または小数ではない")

    def set_feature_property(self, fp=None, verbose=True) -> None:
        if self.mfp is None and fp is None:
            # self.gen_base_feature_property()
            self.int_or_float_feature_property()
            analyzer = TornadoAnalysisModule(self.X,
                                             is_time_series=self.is_ts,
                                             data_interval=self.data_interval,
                                             )
            self.report = analyzer.analyze()
            for check_point, analyzed_features in self.report.items():
                if check_point == "has_seasonality":
                    flag = flags.IS_SEASONAL
                elif check_point == "has_linearity":
                    flag = flags.IS_LINEAR
                elif check_point == "has_up_monotonicity":
                    flag = flags.IS_MONOTONIC
                elif check_point == "has_down_monotonicity":
                    flag = flags.IS_MONOTONIC
                elif check_point == "has_missing":
                    flag = flags.HAS_MISSING
                else:
                    continue

                for feature in analyzed_features:
                    self.feature_properties[feature] |= flag
        elif fp is not None:
            self.feature_properties = fp
        else:
            self.feature_properties = self.mfp

        if verbose:
            for key, value in self.report.items():
                _logger.info(f"    {key}: {value}\n")

    def set_feature(self, feature: list = None) -> None:
        if feature is None:
            raise ValueError("feature doesn't not defined")
        self.features = feature

    def set_interaction_term(self, n_comb=2) -> None:
        if n_comb <= 1:
            raise ValueError("interaction size must be more than 2")
        elif n_comb >= 3:
            _logger.warning("many interaction terms might cause long training")

        for s in range(2, n_comb + 1):
            feature = self.init_model_attr["features"]
            comb = itertools.combinations(feature, s)
            self.interaction_term += [c for c in comb]

        self.max_interaction = len(self.interaction_term)

    def set_smoother(self, smoothers: dict = None) -> None:
        if smoothers is None:
            smoothers = dict()
            for key, cols in self.report.items():
                if len(cols) > 0:
                    if key == "has_seasonality":
                        for col in cols:
                            smoothers[(col,)] = SeasonalSmoother(order=3)
                    elif key == "has_up_monotonicity":
                        for col in cols:
                            smoothers[(col,)] = IsotonicRegressor(increasing=True)
                    elif key == "has_down_monotonicity":
                        for col in cols:
                            smoothers[(col,)] = IsotonicRegressor(increasing=False)
            self.smoothers = smoothers

    def set_observer(self) -> None:
        self.observers = [
            observers.PlottingObserver(iteration=1),
            observers.PlottingObserver(iteration=-1),
        ]

    def drop_unused_features(self) -> None:
        target = list()
        for col in self.init_model_attr["X"].columns:
            if col not in self.feature_properties.keys():
                target.append(col)
        self.X = self.init_model_attr["X"].drop(target, axis=1)

    def clear(self) -> None:
        self.features = list()
        self.feature_properties = dict()
        self.smoothers = dict()
        self.observers = dict()

    def init(self, dataset, target, n_comb=2) -> None:
        self.target = target.lower()
        self.y = np.asarray(dataset[self.target])
        self.X = dataset.drop(self.target, axis=1)

        if not self.is_ts:
            self.X = self.X.drop("date", axis=1)

        # set and save model parameter
        self.set_feature_property()
        self.init_model_attr["X"] = self.X
        self.init_model_attr["y"] = self.y
        self.init_model_attr["features"] = list(self.feature_properties.keys())
        self.init_model_attr["feature_properties"] = self.feature_properties

        # NOTE: run after feature_properties is into init_model_params
        self.set_interaction_term(n_comb=n_comb)
        self.init_model_attr["interaction"] = self.interaction_term

        # NOTE: run after X and feature_property are into init_model_params
        self.drop_unused_features()

    def build(self):
        # set param
        param = {
            "feature_properties": self.feature_properties,
            "feature_groups": self.features,
            "observers": self.observers,
            "maximal_iterations": 50,
            "smoother_choice": common_smoothers.SmootherChoiceGroupBy(
                use_regression_type=True,
                use_normalization=False,
                explicit_smoothers=self.smoothers,
            ),
        }

        # additional or update param
        for k, p in self.model_params.items():
            param[k] = p

        # build
        if self.task == "regression":
            if self.dist == "qpd":
                if self.model == "multiplicative":
                    self.regressor = MultiplicativeQuantileRegressor(**param)
                elif self.model == "additive":
                    self.regressor = AdditiveQuantileRegressor(**param)
                else:
                    raise ValueError(f"{self.mode} mode is not exist")

            elif self.dist == "poisson":
                self.regressor = PoissonRegressor(**param)
            elif self.dist == "nbinom":
                self.regressor = NBinomRegressor(**param)
            elif self.dist == "nbinomc":
                param["mean_prediction_column"] = "yhat_mean"
                param["feature_properties"]["yhat_mean_feature"] = flags.IS_CONTINUOUS | flags.HAS_MISSING | flags.MISSING_NOT_LEARNED
                param["feature_groups"].append("yhat_mean_feature")
                return NBinomC(**param)

            else:
                raise ValueError(f"{self.dist} mode is not exist")

        return self.regressor

    @abc.abstractmethod
    def update(self) -> None:
        pass

    @abc.abstractmethod
    def manage(self) -> bool:
        pass


class TornadoModule(TornadoModuleBase):
    def __init__(self,
                 manual_feature_property=None,
                 is_time_series=True,
                 data_interval=None,
                 ) -> None:
        super().__init__(manual_feature_property, is_time_series, data_interval)
        self.mode = "multiple"

    def set_feature(self) -> None:
        # single feature
        for feature in self.feature_properties.keys():
            self.features.append(feature)

        # interaction term
        point = self.experiment - 1
        self.features.append(self.interaction_term[point])

    def set_interaction_term(self, n_comb=2) -> None:
        if n_comb <= 1:
            raise ValueError("interaction size must be more than 2")
        elif n_comb >= 3:
            ("WARNING: many interaction terms might cause long training")

        for s in range(2, n_comb + 1):
            feature = self.init_model_attr["features"]
            comb = itertools.combinations(feature, s)
            self.interaction_term += [c for c in comb]

        self.max_interaction = len(feature) + len(self.interaction_term)

    def update(self) -> None:
        self.set_feature()
        self.set_feature_property(self.feature_properties)
        self.set_smoother()
        self.set_observer()

    def manage(self) -> bool:
        self.end = self.max_interaction
        if self.experiment <= self.end - 1:
            self.clear()
            self.update()
            self.experiment += 1
            return True
        else:
            return False


class ForwardSelectionModule(TornadoModuleBase):
    def __init__(self, manual_feature_property=None,
                 is_time_series=True,
                 data_interval=None,
                 dist="poisson",
                 ) -> None:
        super().__init__(manual_feature_property,
                         is_time_series,
                         data_interval,
                         dist=dist,
                         )
        self.first_round = "single_regression_analysis"
        self.second_round = "multiple_regression_analysis"
        self.mode = self.first_round
        self.hold_setting = {"features": list(), "feature_properties": dict()}
        self.target_features = list()
        self.model_params = dict()

    def update(self) -> None:
        if self.mode == self.first_round:
            ix = self.experiment

            name = "features"
            base = self.init_model_attr[name]
            interaction = self.interaction_term
            feature = base + interaction
            self.set_feature([feature[ix]])

            name = "feature_properties"
            if isinstance(feature[ix], tuple):
                param = {k: self.init_model_attr[name][k] for k in feature[ix]}
            else:
                param = {feature[ix]: self.init_model_attr[name][feature[ix]]}
            self.set_feature_property(param, verbose=False)

            self.set_smoother()

            self.set_observer()

            self.drop_unused_features()

        elif self.mode == self.second_round:
            ix = self.experiment

            name = "features"
            feature = self.target_features[ix]
            self.hold_setting[name].append(feature)
            self.set_feature(self.hold_setting[name])

            name = "feature_properties"
            tmp = list()
            for feature in self.features:
                if isinstance(feature, tuple):
                    for f in feature:
                        tmp.append(f)
                else:
                    tmp.append(feature)
            unique_feature = list(set(tmp))
            param = dict()
            for feature in unique_feature:
                property = self.init_model_attr[name][feature]
                param[feature] = property

            self.set_feature_property(param, verbose=False)

            self.set_smoother()

            self.set_observer()

            self.drop_unused_features()

    def manage(self) -> bool:
        if self.mode == self.first_round:
            single_size = len(self.init_model_attr["features"])
            interaction_size = len(self.interaction_term)
            self.end = single_size + interaction_size
        else:
            self.end = len(self.target_features)

        if self.experiment <= self.end - 1:
            self.clear()
            self.update()
            self.experiment += 1
            return True
        else:
            return False


class BFForwardSelectionModule(TornadoModuleBase):
    def __init__(self,
                 manual_feature_property=None,
                 is_time_series=True,
                 dist="qpd",
                 ) -> None:
        super().__init__(manual_feature_property, is_time_series, dist=dist)
        self.first_round = "prior_prediction_with_single_variables"
        self.second_round = "interaction_search"
        self.mode = self.first_round
        self.model_params = {"quantile": 0.5, "maximal_iterations": 1}

    def update(self) -> None:
        if self.mode == self.first_round:
            feature = self.init_model_attr["features"]
            self.set_feature(feature)

            fp = self.init_model_attr["feature_properties"]
            self.set_feature_property(fp, verbose=False)

            self.set_smoother()

            self.set_observer()

            self.drop_unused_features()

        elif self.mode == self.second_round:
            ix = self.experiment

            name = "prior_prediction_column"
            prior = self.model_params[name]

            name = "interaction"
            interaction = self.init_model_attr[name][ix]
            self.set_feature([interaction, prior])

            name = "feature_properties"
            fp = self.init_model_attr[name]
            param = {feature: fp[feature] for feature in interaction}
            param[prior] = flags.IS_CONTINUOUS
            self.set_feature_property(param, verbose=False)

            self.set_smoother()

            self.set_observer()

            self.drop_unused_features()

    def manage(self) -> bool:
        if self.mode == self.first_round:
            self.end = 1
        else:
            self.end = self.max_interaction

        if self.experiment <= self.end - 1:
            self.clear()
            self.update()
            self.experiment += 1
            return True
        else:
            return False
