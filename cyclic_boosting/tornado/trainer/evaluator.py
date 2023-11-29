import abc
import six
import numpy as np
from .metrics import (
    mean_deviation,
    mean_absolute_deviation,
    mean_square_error,
    mean_absolute_error,
    median_absolute_error,
    weighted_absolute_percentage_error,
    symmetric_mean_absolute_percentage_error,
    coefficient_of_determination,
    F,
    mean_pinball_loss,
)


@six.add_metaclass(abc.ABCMeta)
class EvaluatorBase:
    def __init__(self) -> None:
        self.result = {}
        self.eval_funcs = {}

    def eval_all(self, y, yhat, est, verbose=True, digit=5):
        # eval
        for metrics, func in self.eval_funcs.items():
            if metrics not in self.result.keys():
                self.result[metrics] = []

            if not metrics == "F":
                result = func(y, yhat)
            else:
                result = func(y, yhat, k=len(est["CB"].feature_groups))
            self.result[metrics].append(result)

            if verbose:
                print(f"[{metrics}]: {round(result, digit)}")

    def mean_metrics(self, digit=5):
        for metrics, values in self.result.items():
            print(f"[{metrics}]: {round(np.mean(values), digit)}")

    def clear(self):
        self.result = {}


class Evaluator(EvaluatorBase):
    def __init__(self) -> None:
        self.result = {}
        self.eval_funcs = {
            "MD": mean_deviation,
            "MAD": mean_absolute_deviation,
            "MSE": mean_square_error,
            "MAE": mean_absolute_error,
            "MedAE": median_absolute_error,
            "WMAPE": weighted_absolute_percentage_error,
            "SMAPE": symmetric_mean_absolute_percentage_error,
            "COD": coefficient_of_determination,
            "F": F,
        }


class QuantileEvaluator(EvaluatorBase):
    def __init__(self) -> None:
        self.result = {}
        self.eval_funcs = {"PINBALL": mean_pinball_loss}
