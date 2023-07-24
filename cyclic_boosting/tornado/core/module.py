# Add some comments
#
#
#

import abc
import six
import pandas as pd

from cyclic_boosting import flags, common_smoothers, observers, binning
from cyclic_boosting.plots import plot_analysis
from cyclic_boosting.pipelines import pipeline_CBPoissonRegressor

from cyclic_boosting.smoothing.onedim import SeasonalSmoother, IsotonicRegressor


# @six.add_metaclass(abc.ABCMeta)
class TornadoModule():
    # some comments

    def __init__(self, ) -> None:
        super().__init__()
        self.feature_property = {}
        self.features = []
        self.smoothers = {}
        self.observers = {}
        self.CB_pipline = {}

    # @abc.abstractmethod
    def set_feature_property(self) -> None:

        self.feature_properties = {
            "P_ID": flags.IS_UNORDERED,
            "PG_ID_3": flags.IS_UNORDERED,
            "L_ID": flags.IS_UNORDERED,
            "dayofweek": flags.IS_ORDERED,
            "dayofyear": flags.IS_CONTINUOUS | flags.IS_LINEAR,
            "price_ratio": flags.IS_CONTINUOUS
            | flags.HAS_MISSING
            | flags.MISSING_NOT_LEARNED,
            "PROMOTION_TYPE": flags.IS_ORDERED,
        }
 
        # raise NotImplementedError("implement in subclass")

    # @abc.abstractmethod
    def set_feature(self) -> None:
        self.features = [
            "dayofweek",
            "L_ID",
            "PG_ID_3",
            "P_ID",
            "PROMOTION_TYPE",
            "price_ratio",
            "dayofyear",
            ("P_ID", "L_ID"),
        ]
        # raise NotImplementedError("implement in subclass")

    # @abc.abstractmethod
    def set_smoother(self) -> None:
        self.smoothers = {
            ("dayofyear",): SeasonalSmoother(order=3),
            ("price_ratio",): IsotonicRegressor(increasing=False),
        }
        # raise NotImplementedError("implement in subclass")

    # @abc.abstractmethod
    def set_observer(self) -> None:
        self.observers = [
            observers.PlottingObserver(iteration=1),
            observers.PlottingObserver(iteration=-1),
        ]
        # raise NotImplementedError("implement in subclass")

    # @abc.abstractmethod
    def build(self) -> pipeline_CBPoissonRegressor:

        self.set_feature_property()
        self.set_feature()
        self.set_smoother()
        self.set_observer()

        self.CB_pipeline = pipeline_CBPoissonRegressor(
            feature_properties=self.feature_propertie,
            feature_groups=self.features,
            observers=self.observers,
            maximal_iterations=50,
            smoother_choice=common_smoothers.SmootherChoiceGroupBy(
                use_regression_type=True,
                use_normalization=False,
                explicit_smoothers=self.smoothers,
            ),
        )
        # raise NotImplementedError("implement in subclass")

        return self.CB_pipeline


# class TornadoRegressor(TornadoModule):
