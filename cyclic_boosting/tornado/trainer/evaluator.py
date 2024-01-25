import logging

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

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(fmt="%(message)s"))
handler.terminator = ""
_logger.addHandler(handler)


@six.add_metaclass(abc.ABCMeta)
class EvaluatorBase:
    def __init__(self) -> None:
        self.result = dict()
        self.eval_funcs = dict()

    def eval(self, y, yhat, est, verbose=True, digit=5):
        for metrics, func in self.eval_funcs.items():
            if metrics not in ["F", "COD"]:
                result = func(y, yhat)
            else:
                result = func(y, yhat, k=len(est["CB"].feature_groups))

            self.result[metrics] = result

            if verbose:
                _logger.info(f"[{metrics}]: {round(result, digit)}\n")

        return self.result

    def mean_metrics(self, digit=5):
        for metrics, values in self.result.items():
            _logger.info(f"[{metrics}]: {round(np.mean(values), digit)}\n")

    def clear(self):
        self.result = dict()


class Evaluator(EvaluatorBase):
    def __init__(self) -> None:
        self.result = dict()
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
        self.result = dict()
        self.eval_funcs = {"PINBALL": mean_pinball_loss,
                           "COD": coefficient_of_determination,
                           }

    def eval(self, y, yhat, q, est, verbose=True, digit=5):
        for metrics, func in self.eval_funcs.items():
            if metrics == "PINBALL":
                result = func(y, yhat, q)
            else:
                result = func(y, yhat, k=len(est["CB"].feature_groups))

            self.result[metrics] = result

            if verbose:
                _logger.info(f"[{metrics}]: {round(result, digit)}\n")

        return self.result
