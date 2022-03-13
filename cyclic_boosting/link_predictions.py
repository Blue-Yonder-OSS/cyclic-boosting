import numpy as np
import pandas as pd
from numexpr import evaluate

from cyclic_boosting.features import FeatureTypes


class UpdateMixin(object):
    def include_price_contrib(self, pred):
        self.df["exponents"] += pred
        self.price_feature_seen = True

    def include_factor_contrib(self, pred, influence_category):
        self.df["factors"] += pred
        if influence_category is not None:
            if influence_category in self.df.columns:
                self.df[influence_category] += pred
            else:
                self.df[influence_category] = pred

    def remove_price_contrib(self, pred):
        self.df["exponents"] -= pred

    def remove_factor_contrib(self, pred, influence_category):
        self.df["factors"] -= pred
        if influence_category is not None:
            if influence_category in self.df.columns:
                self.df[influence_category] -= pred
            else:
                self.df[influence_category] = pred

    def update_predictions(self, pred, feature, influence_categories=None):
        if feature.feature_type == FeatureTypes.external:
            self.include_price_contrib(pred)
        else:
            self.include_factor_contrib(
                pred, get_influence_category(feature, influence_categories)
            )

    def remove_predictions(self, pred, feature, influence_categories=None):
        if feature.feature_type == FeatureTypes.external:
            self.remove_price_contrib(pred)
        else:
            self.remove_factor_contrib(
                pred, get_influence_category(feature, influence_categories)
            )


class CBLinkPredictions(UpdateMixin):
    """Support for prediction of type log(p) = factors + base * exponents"""

    def __init__(self, predictions, exponents, base):
        self.df = pd.DataFrame(
            {
                "factors": predictions,
                "prior_exponents": exponents,
                "exponents": np.zeros_like(exponents, dtype=np.float64),
                "base": base,
            }
        )

    def predict_link(self):
        return self.factors() + self.exponents() * self.base()

    def factors(self):
        return self.df["factors"].values

    def exponents(self):
        x = self.df["exponents"].values  # noqa
        return self.df["prior_exponents"].values * evaluate("exp(x)")

    def prior_exponents(self):
        return self.df["prior_exponents"].values

    def base(self):
        return self.df["base"].values


class CBLinkPredictionsFactors(UpdateMixin):
    """Support for prediction of type log(p) = factors"""

    def __init__(self, predictions):
        self.df = pd.DataFrame({"factors": predictions})

    def predict_link(self):
        return self.factors()

    def factors(self):
        return self.df["factors"].values


def get_output_columns_influence_categories(cb_est, influence_categories):
    cols = set()
    if influence_categories is not None:
        for feature in cb_est.features:
            if feature.feature_type is None:
                cols.add(get_influence_category(feature, influence_categories))
    return list(cols)


def get_influence_category(feature, influence_categories):
    influence_category = None
    if influence_categories is not None:
        influence_category = influence_categories.get(feature.feature_group, None)
        if (
            influence_category is None
        ):  # this is due to the flaws in create features, i would propose to only allow tuple
            if len(feature.feature_group) == 1:
                fg = feature.feature_group[0]
                influence_category = influence_categories.get(fg, None)
        if influence_category is None:
            raise KeyError(
                f"Please add {feature.feature_group} to influence_categories"
            )
    return influence_category
