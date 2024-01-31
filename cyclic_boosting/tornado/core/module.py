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
        max_iter=10,
        task="regression",
        dist="poisson",
        model="multiplicative",
    ) -> None:
        self.manual_feature_property = manual_feature_property
        self.is_time_series = is_time_series
        self.data_interval = data_interval
        self.max_iter = max_iter
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
        self.experiment = 0
        self.end = 0

    def get_params(self) -> dict:
        class_vars = dict()
        for attr_name, value in self.__dict__.items():
            if not callable(value) and not attr_name.startswith("__"):
                class_vars[attr_name] = value

        return class_vars

    def set_params(self, params: dict) -> None:
        for attr_name, value in params.items():
            self.__dict__[attr_name] = value

    def set_feature_property(self, fp=None, verbose=True) -> None:
        if fp is None:
            analyzer = TornadoAnalysisModule(self.X,
                                             is_time_series=self.is_time_series,
                                             data_interval=self.data_interval,
                                             )
            self.report = analyzer.analyze()
            for prop, analyzed_features in self.report.items():
                if prop == "is_unordered":
                    flag = flags.IS_UNORDERED
                elif prop == "is_continuous":
                    flag = flags.IS_CONTINUOUS
                elif prop == "has_seasonality":
                    flag = flags.IS_SEASONAL
                elif prop == "has_linearity":
                    flag = flags.IS_LINEAR
                elif prop == "has_up_monotonicity":
                    flag = flags.IS_MONOTONIC
                elif prop == "has_down_monotonicity":
                    flag = flags.IS_MONOTONIC
                elif prop == "has_missing":
                    flag = flags.HAS_MISSING
                else:
                    continue

                for feature in analyzed_features:
                    if feature not in self.feature_properties.keys():
                        self.feature_properties[feature] = flag
                    else:
                        self.feature_properties[feature] |= flag
        else :
            self.feature_properties = {k: v for k, v in fp.items()}

        if verbose:
            for key, value in self.report.items():
                _logger.info(f"    {key}: {value}\n")

    def set_smoother(self, smoothers: dict = None) -> None:
        if smoothers is None:
            smoothers = dict()
            for prop, features in self.report.items():
                if len(features) > 0:
                    if prop == "has_seasonality":
                        for col in features:
                            smoothers[(col,)] = SeasonalSmoother(order=3)
                    elif prop == "has_up_monotonicity":
                        for col in features:
                            smoothers[(col,)] = IsotonicRegressor(increasing=True)
                    elif prop == "has_down_monotonicity":
                        for col in features:
                            smoothers[(col,)] = IsotonicRegressor(increasing=False)
            self.smoothers = smoothers

    def set_observer(self) -> None:
        self.observers = [
            observers.PlottingObserver(iteration=1),
            observers.PlottingObserver(iteration=-1),
        ]

    def drop_unused_features(self) -> None:
        target = list()
        for feature in self.init_model_attr["X"].columns:
            if feature not in self.feature_properties.keys():
                target.append(feature)
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

        if not self.is_time_series:
            self.X = self.X.drop("date", axis=1)

        # set and save model parameter
        if self.manual_feature_property:
            self.set_feature_property(fp=self.manual_feature_property)
        else:
            self.set_feature_property()
        self.init_model_attr["X"] = self.X
        self.init_model_attr["y"] = self.y
        self.features = list(self.feature_properties.keys())
        self.init_model_attr["features"] = list(self.feature_properties.keys())
        self.init_model_attr["feature_properties"] = self.feature_properties

        # NOTE: run after feature_properties is into init_model_params
        self.set_interaction(n_comb=n_comb)
        self.init_model_attr["interaction"] = self.interaction_term

        # NOTE: run after X and feature_property are into init_model_params
        self.drop_unused_features()

    def build(self):
        # set param
        param = {
            "feature_properties": self.feature_properties,
            "feature_groups": self.features,
            "observers": self.observers,
            "maximal_iterations": self.max_iter,
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

            elif self.dist in ["poisson", "nbinom"]:
                self.regressor = PoissonRegressor(**param)
            elif self.dist == "nbinomc":
                param["mean_prediction_column"] = "yhat_mean"
                param["feature_properties"]["yhat_mean_feature"] = flags.IS_CONTINUOUS | flags.HAS_MISSING | flags.MISSING_NOT_LEARNED
                param["feature_groups"].append("yhat_mean_feature")
                return NBinomC(**param)

            else:
                raise ValueError(f"{self.dist} mode is not exist")

        return self.regressor

    @abc.abstractmethod
    def set_feature(self) -> None:
        pass

    @abc.abstractmethod
    def set_interaction(self, n_comb=2) -> None:
        pass

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
                 max_iter=10,
                 dist="poisson",
                 model=None,
                 ) -> None:
        super().__init__(
            manual_feature_property,
            is_time_series,
            data_interval,
            max_iter=max_iter,
            dist=dist,
            model=model,
            )
        self.first_round = "dummy"
        self.second_round = "interaction search"
        self.mode = self.second_round

    def set_feature(self, feature: list) -> None:
        idx = self.experiment
        feature.append(self.interaction_term[idx])
        self.features = feature

    def set_interaction(self, n_comb=2) -> None:
        if n_comb <= 1:
            raise ValueError("interaction size must be more than 2")
        elif n_comb >= 3:
            ("WARNING: many interaction terms might cause long training")

        for s in range(2, n_comb + 1):
            feature = self.init_model_attr["features"]
            comb = itertools.combinations(feature, s)
            self.interaction_term += [c for c in comb]

    def update(self) -> None:
        self.set_feature([x for x in self.features])
        self.set_feature_property(self.feature_properties, verbose=False)
        self.set_smoother()
        self.set_observer()

    def manage(self) -> bool:
        self.end = len(self.interaction_term)
        if self.experiment < self.end:
            self.update()
            self.experiment += 1
            return True
        else:
            return False


class ForwardSelectionModule(TornadoModule):
    def __init__(self,
                 manual_feature_property=None,
                 is_time_series=True,
                 data_interval=None,
                 max_iter=10,
                 dist="poisson",
                 ) -> None:
        super().__init__(
            manual_feature_property,
            is_time_series,
            data_interval,
            max_iter=max_iter,
            dist=dist,
            )
        self.first_round = "single_regression_analysis"
        self.second_round = "multiple_regression_analysis"
        self.mode = self.first_round
        self.target_features = list()
        self.model_params = dict()

    def set_feature(self, feature: list = None) -> None:
        if feature is None:
            raise ValueError("feature doesn't not defined")
        self.features = feature

    def update(self) -> None:
        if self.mode == self.first_round:
            idx = self.experiment

            base = self.init_model_attr["features"]
            interaction = self.interaction_term
            feature = base + interaction
            self.set_feature([feature[idx]])

            name = "feature_properties"
            if isinstance(feature[idx], tuple):
                param = {k: self.init_model_attr[name][k] for k in feature[idx]}
            else:
                param = {feature[idx]: self.init_model_attr[name][feature[idx]]}
            self.set_feature_property(param, verbose=False)
            self.set_smoother()
            self.set_observer()
            self.drop_unused_features()

        elif self.mode == self.second_round:
            idx = self.experiment
            feature = self.target_features[idx]
            next_features = [x for x in self.features]
            next_features.append(feature)
            self.set_feature(next_features)

            prop = {k: v for k, v in self.feature_properties.items()}
            if isinstance(feature, tuple):
                next_prop = [x for x in feature]
            elif isinstance(feature, str):
                next_prop = [feature]
            else:
                ValueError()

            for feature in next_prop:
                if feature not in prop.keys():
                    prop[feature] = self.init_model_attr["feature_properties"][feature]
            self.set_feature_property(prop, verbose=False)

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

        if self.experiment < self.end:
            self.update()
            self.experiment += 1
            return True
        else:
            return False


class PriorPredForwardSelectionModule(TornadoModule):
    def __init__(self,
                 manual_feature_property=None,
                 is_time_series=True,
                 max_iter=10,
                 dist="qpd",
                 model="additive",
                 ) -> None:
        super().__init__(
            manual_feature_property,
            is_time_series,
            max_iter=max_iter,
            dist=dist,
            model=model,
            )
        self.first_round = "prior_prediction_with_single_variables"
        self.second_round = "interaction_search"
        self.mode = self.first_round
        self.model_params = {"quantile": 0.5, "maximal_iterations": 1}

    def set_feature(self, feature: list = None) -> None:
        if feature is None:
            raise ValueError("feature doesn't not defined")
        self.features = feature

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
            idx = self.experiment
            single_interaction = self.init_model_attr["interaction"][idx]
            self.set_feature([single_interaction])

            feature_properties = self.init_model_attr["feature_properties"]
            prop = {f_name: feature_properties[f_name] for f_name in single_interaction}
            prior = self.model_params["prior_prediction_column"]
            prop[prior] = flags.IS_CONTINUOUS  # to skip drop_unused_features step
            self.set_feature_property(prop, verbose=False)

            self.set_smoother()
            self.set_observer()
            self.drop_unused_features()

    def manage(self) -> bool:
        if self.mode == self.first_round:
            self.end = 1
        else:
            self.end = len(self.interaction_term)

        if self.experiment < self.end:
            self.update()
            self.experiment += 1
            return True
        else:
            return False
