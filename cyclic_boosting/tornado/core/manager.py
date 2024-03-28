"""Handle model settings."""

from __future__ import annotations

import abc
import itertools
import logging
from typing import TYPE_CHECKING

import numpy as np
import six

from cyclic_boosting import common_smoothers, flags, observers
from cyclic_boosting.pipelines import (
    pipeline_CBAdditiveQuantileRegressor as AdditiveQuantileRegressor,
)
from cyclic_boosting.pipelines import pipeline_CBNBinomC as NBinomC
from cyclic_boosting.pipelines import pipeline_CBPoissonRegressor as PoissonRegressor
from cyclic_boosting.smoothing.onedim import IsotonicRegressor, SeasonalSmoother

from .analysis import TornadoAnalysisModule

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(fmt="%(message)s"))
handler.terminator = ""
_logger.addHandler(handler)


@six.add_metaclass(abc.ABCMeta)
class ManagerBase:
    """Base class handling model settings in the recursive learning.

    Attributes
    ----------
    manual_feature_property : dict or None
        Feature property set manually. Dfault is None.

    is_time_series : bool
        Time-series data or not. Default is True.

    data_interval : str or None
        The data collection interval which can be one of "hourly", "daily",
        "weekly", "monthly", or None. If None (default), the interval will be
        inferred and set automatically.

    combination : int
        Maximum number of features included in the combination of
        interaction terms created from the given features. Default is 2.

    max_iter : int
        Maximal number of iteration. This is used in each estimator. Default
        is 10.

    task : str
        Type of task. Default is 'regression'.

    dist : str
        Type of distribution. 'poisson', 'nbinom' or 'qpd'. Default is
        'poisson'.

    model : str
        Type of model. 'additive' or 'multiplicative'. Default is
        'multiplicative'. This is used only when `dist` is 'qpd'.

    X : pandas.DataFrame
        Training data with explanatory variables

    y : numpy.ndarray
        Training data with target variable

    target : str
        Name of the target valiable

    regressor : sklearn.pipeline.Pipeline
        Pipeline with steps including binning and CB

    features : list
        List of names of explanatory variables

    interaction_term : list
        Two-dimensional list with lists of interaction terms

    feature_properties : dict
        Dictionary with feature names as keys and properties as values

    smoothers : dict
        Dictionary with feature names as keys and smoothers as values

    observers : list
        List of :class:`PlottingObserver`.

    report : dict
        Results of automatic analysis. Properties and smoothers are set based
        on this.

    init_model_attr : dict
        Attributes of the initial model that will be the base for recursive
        training. A dictionary with attribute names as keys and attribute
        values as values.

    model_params : dict
        Parameters of the model. A dictionary with parameter names as keys and
        parameter values as values.

    experiment : int
        Number of experiments

    end : int
        Maximum number of experiments
    """

    def __init__(
        self,
        manual_feature_property=None,
        is_time_series=True,
        data_interval=None,
        combination=2,
        max_iter=10,
        task="regression",
        dist="poisson",
    ) -> None:
        self.manual_feature_property = manual_feature_property
        self.is_time_series = is_time_series
        self.data_interval = data_interval
        self.combination = combination
        self.max_iter = max_iter
        self.task = task
        self.dist = dist
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

    def get_attr(self) -> dict:
        """Get class attributes.

        Get the attributes of the class not starting with "__" as a dictionary.

        Returns
        -------
        dict
            Dictionary with class attribute names as keys and class attribute
            as values.
        """
        class_vars = dict()
        for attr_name, value in self.__dict__.items():
            if not callable(value) and not attr_name.startswith("__"):
                class_vars[attr_name] = value

        return class_vars

    def set_attr(self, params: dict) -> None:
        """Set class attributes.

        Parameters
        ----------
        params : dict
            Parameters to be set. Dictionary with attribute names as keys and
            attribute as values.
        """
        for attr_name, value in params.items():
            self.__dict__[attr_name] = value

    def set_feature_property(self, fp=None, verbose=True) -> None:
        """Set the feature properties for the model.

        Parameters
        ----------
        fp : dict
            Feature properties to be set. If None, features are automatically
            analyzed and the properties are set. Default is None.

        verbose : bool
            Whether to display standard output or not. Default is True.
        """
        if fp is None:
            analyzer = TornadoAnalysisModule(
                self.X,
                is_time_series=self.is_time_series,
                data_interval=self.data_interval,
            )

            _logger.info("[START] Auto analysis \n")
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
            _logger.info("[END] Auto analysis \n\n")
        else:
            self.feature_properties = {k: v for k, v in fp.items()}

        if verbose:
            _logger.info("Feature properties\n")
            for key, value in self.report.items():
                _logger.info(f"    {key}: {value}\n")
            _logger.info("\n")

    def set_smoother(self, smoothers: dict = None) -> None:
        """Set smoothers for the model.

        Smoothers are set according to the properties of the features.

        Parameters
        ----------
        smoothers : dict
            Smoothers to be set. Default is None.
        """
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
        """Set observers for plotting on the model."""
        self.observers = [
            observers.PlottingObserver(iteration=1),
            observers.PlottingObserver(iteration=-1),
        ]

    def drop_unused_features(self) -> None:
        """Drop unused features in the model."""
        target = list()
        for feature in self.init_model_attr["X"].columns:
            if feature not in self.feature_properties.keys():
                target.append(feature)
        self.X = self.init_model_attr["X"].drop(target, axis=1)

    def clear(self) -> None:
        """Clear the model settings."""
        self.features = list()
        self.feature_properties = dict()
        self.smoothers = dict()
        self.observers = dict()

    def init(self, dataset, target) -> None:
        """Set initial settings for the model.

        Set up the model, generate interaction terms, and drop unneeded
        features. The model settings here are held as the initial model
        settings that are used as the base for recursive learning.

        Parameters
        ----------
        dataset : pandas.DataFrame
            Dataset for learning

        target : str
            Name of the target valiable
        """
        self.target = target.lower()
        self.y = np.asarray(dataset[self.target])
        self.X = dataset.drop(self.target, axis=1)

        # set and save model parameter
        if self.manual_feature_property:
            self.set_feature_property(fp=self.manual_feature_property)
        else:
            self.set_feature_property()
        if self.is_time_series:
            # instead, dayofweek and dayofyear are generated
            self.X = self.X.drop("date", axis=1)
        self.init_model_attr["X"] = self.X
        self.init_model_attr["y"] = self.y
        self.features = list(self.feature_properties.keys())
        self.init_model_attr["features"] = list(self.feature_properties.keys())
        self.init_model_attr["feature_properties"] = self.feature_properties

        # NOTE: run after feature_properties is into init_model_params
        self.set_interaction(combination=self.combination)
        self.init_model_attr["interaction"] = self.interaction_term

        # NOTE: run after X and feature_property are into init_model_params
        self.drop_unused_features()

    def build(self) -> Pipeline:
        """Build the model.

        A model is built depending on the type of task (regression,
        classification, , , ), the type of probability distribution assumed
        (Poisson distribution, negative binomial distribution, QPD, , , ), and
        the type of model (multiplicative model, additive model).

        Returns
        -------
        sklearn.pipeline.Pipeline
            Estimator built with specific settings.
        """
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
                self.regressor = AdditiveQuantileRegressor(**param)

            elif self.dist in ["poisson", "nbinom"]:
                self.regressor = PoissonRegressor(**param)
            elif self.dist == "nbinomc":
                param["mean_prediction_column"] = "yhat_mean"
                param["feature_properties"]["yhat_mean_feature"] = (
                    flags.IS_CONTINUOUS | flags.HAS_MISSING | flags.MISSING_NOT_LEARNED
                )
                param["feature_groups"].append("yhat_mean_feature")
                return NBinomC(**param)

            else:
                raise ValueError(f"{self.dist} mode is not exist")

        return self.regressor

    @abc.abstractmethod
    def set_feature(self) -> None:
        """Abstract methods to set features."""
        pass

    @abc.abstractmethod
    def set_interaction(self, combination=2) -> None:
        """Abstract methods to set interaction terms."""
        pass

    @abc.abstractmethod
    def update(self) -> None:
        """Abstract methods to update settings."""
        pass

    @abc.abstractmethod
    def manage(self) -> bool:
        """Abstract methods to manage model settings."""
        pass


class TornadoManager(ManagerBase):
    """Class handling the model settings in ordinary Tornado.

    Controls the settings of the model in recursive learning in ordinary
    Tornado. This class inherits from :class:`TornadoModuleBase`.

    Attributes
    ----------
    first_round : str
        Name of the first round in the learning process. Default is "dummy"
        because Recursive learning is performed only once in ordinary Tornado.

    second_round : str
        Name of the second round in the learning process. Default is
        "interaction search".

    mode : str
        Current round

    Notes
    -----
    Some attributes are inherited from the base class and can be configured
    when creating a class instance. See the Attributes of base class for a
    description.
    """

    def __init__(
        self,
        manual_feature_property=None,
        is_time_series=True,
        data_interval=None,
        combination=2,
        max_iter=10,
        dist="poisson",
    ) -> None:
        super().__init__(
            manual_feature_property,
            is_time_series,
            data_interval,
            combination=combination,
            max_iter=max_iter,
            dist=dist,
        )
        self.first_round = "dummy"
        self.second_round = "interaction search"
        self.mode = self.second_round

    def set_feature(self, feature: list) -> None:
        """Set feature for the model.

        Parameters
        ----------
        feature : list
            List containing feature names.
        """
        idx = self.experiment
        feature.append(self.interaction_term[idx])
        self.features = feature

    def set_interaction(self, combination=2) -> None:
        """Set interaction terms for the model.

        Parameters
        ----------
        combination : int
            Maximum number of features included in the combination of
            interaction terms created from the given features. Default is 2.
        """
        if combination <= 1:
            raise ValueError("interaction size must be more than 2")
        elif combination >= 3:
            ("WARNING: many interaction terms might cause long training")

        for s in range(2, combination + 1):
            feature = self.init_model_attr["features"]
            comb = itertools.combinations(feature, s)
            self.interaction_term += [c for c in comb]

    def update(self) -> None:
        """Update model settings."""
        self.set_feature([x for x in self.features])
        self.set_feature_property(self.feature_properties, verbose=False)
        self.set_smoother()
        self.set_observer()

    def manage(self) -> bool:
        """Manage model settings in the recursive learning.

        Returns
        -------
        bool
            Whether to continue recursive learning or not.
        """
        self.end = len(self.interaction_term)
        if self.experiment < self.end:
            self.update()
            self.experiment += 1
            return True
        else:
            return False


class ForwardSelectionManager(TornadoManager):
    """Class handling the model settings in forward selection of Tornado.

    Controls the settings of the model in recursive learning in forward
    feature selection of Tornado. This class inherits from
    :class:`TornadoModule`.

    Attributes
    ----------
    first_round : str
        Name of the first round in the learning process. Default is
        "single_regression_analysis".

    second_round : str
        Name of the second round in the learning process. Default is
        "multiple_regression_analysis".

    mode : str
        Current round

    target_features : list
        List of features for variable selection

    Notes
    -----
    Some attributes are inherited from the base class and can be configured
    when creating a class instance. See the Attributes of base class for a
    description.
    """

    def __init__(
        self,
        manual_feature_property=None,
        is_time_series=True,
        data_interval=None,
        combination=2,
        max_iter=10,
        dist="poisson",
    ) -> None:
        super().__init__(
            manual_feature_property,
            is_time_series,
            data_interval,
            combination=combination,
            max_iter=max_iter,
            dist=dist,
        )
        self.first_round = "single_regression_analysis"
        self.second_round = "multiple_regression_analysis"
        self.mode = self.first_round
        self.target_features = list()
        self.model_params = dict()

    def set_feature(self, feature: list = None) -> None:
        """Set feature for the model.

        Parameters
        ----------
        feature : list
            List containing feature names.
        """
        if feature is None:
            raise ValueError("feature doesn't not defined")
        self.features = feature

    def update(self) -> None:
        """Update model settings.

        If `mode` is the first round of 'single_regression_analysis', one
        feature from all single features and interaction terms are set in the
        model at a time. If `mode` is the second round of
        'multiple_regression_analysis', one feature from all single features
        and interaction terms are added and set into the model one by one.
        """
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
        """Manage model settings in the recursive learning.

        Returns
        -------
        bool
            Whether to continue recursive learning or not.
        """
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


class PriorPredForwardSelectionManager(TornadoManager):
    """Handle the model setting in forward selection with prior prediction.

    Control the settings of the model in recursive learning in forward
    selection with prior prediction in Tornado. This class inherits from
    :class:`TornadoModule`.

    Attributes
    ----------
    first_round : str
        Name of the first round in the learning process. Default is
        "prior_prediction_with_single_variables".

    second_round : str
        Name of the second round in the learning process. Default is
        "interaction_search".

    mode : str
        Current round

    Notes
    -----
    Some attributes are inherited from the base class and can be configured
    when creating a class instance. See the Attributes of base class for a
    description.
    """

    def __init__(
        self,
        manual_feature_property=None,
        is_time_series=True,
        combination=2,
        max_iter=10,
        dist="poisson",
    ) -> None:
        super().__init__(
            manual_feature_property,
            is_time_series,
            combination=combination,
            max_iter=max_iter,
            dist=dist,
        )
        self.first_round = "prior_prediction_with_single_variables"
        self.second_round = "interaction_search"
        self.mode = self.first_round
        self.model_params = {"quantile": 0.5, "maximal_iterations": 1}

    def set_feature(self, feature: list = None) -> None:
        """Set feature for the model.

        Parameters
        ----------
        feature : list
            List containing feature names.
        """
        if feature is None:
            raise ValueError("feature doesn't not defined")
        self.features = feature

    def update(self) -> None:
        """Update model settings.

        If `mode` is the first round of
        `prior_prediction_with_single_variables', all single features are set
        in the model. If `mode` is the second round of `interaction_search',
        two features are set in the model : one feature is the prediction
        result in round 1 and the other feature is one from all the
        interaction terms.
        """
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
        """Manage model settings in the recursive learning.

        Returns
        -------
        bool
            Whether to continue recursive learning or not.
        """
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
