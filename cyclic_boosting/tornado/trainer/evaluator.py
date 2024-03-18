"""Evaluation of the model of Tornado's training."""

import abc
import logging

import numpy as np
import six

from .metrics import (
    F,
    coefficient_of_determination,
    mean_absolute_deviation,
    mean_absolute_error,
    mean_deviation,
    mean_pinball_loss,
    mean_square_error,
    median_absolute_error,
    symmetric_mean_absolute_percentage_error,
    weighted_absolute_percentage_error,
)

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(fmt="%(message)s"))
handler.terminator = ""
_logger.addHandler(handler)


@six.add_metaclass(abc.ABCMeta)
class EvaluatorBase:
    """Base class for model evaluation.

    Includes processing common to :class:`Evaluator` and
    :class:`QuantileEvaluator`.

    Attributes
    ----------
    result : dict
        Evaluation result

    eval_funcs : dict
        Evaluation function
    """

    def __init__(self) -> None:
        self.result = dict()
        self.eval_funcs = dict()

    def eval(self, y, yhat, est, verbose=True, digit=5):
        """Perform the evaluation.

        Apply evaluation functions in order according to `eval_func`. Also,
        make standard output.

        Parameters
        ----------
        y : numpy.nddary
            Ground truth

        yhat : numpy.nddary
            Predicted value

        est : sklearn.pipeline.Pipeline
            The model to be saved.

        verbose : bool
            Whether to display standard output or not. Default is True.

        digit : int
            Number of digits of standard output. Default is 5.

        Returns
        -------
        dict
            Results of evaluation
        """
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
        """Display the mean of the evaluation results in the standard output.

        digit : int
            Number of digits of standard output. Default is 5.
        """
        for metrics, values in self.result.items():
            _logger.info(f"[{metrics}]: {round(np.mean(values), digit)}\n")

    def clear(self):
        """Initialize evaluation results."""
        self.result = dict()


class Evaluator(EvaluatorBase):
    """Evaluate the Tornado model.

    Evaluate a standard Tornado model or a ForwardTrainer model. This class
    inherits from the base class :class:`EvaluatorBase`.

    Attributes
    ----------
    result : dict
        Evaluation result

    eval_funcs : dict
        Evaluation function
    """

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
    """Evaluate the Tornado model.

    Evaluate a model of a QPDForwardTrainer. This class inherits from the base
    class :class:`EvaluatorBase`.

    Attributes
    ----------
    result : dict
        Evaluation result

    eval_funcs : dict
        Evaluation function
    """

    def __init__(self) -> None:
        self.result = dict()
        self.eval_funcs = {
            "PINBALL": mean_pinball_loss,
            "COD": coefficient_of_determination,
        }

    def eval(self, y, yhat, q, est, verbose=True, digit=5):
        """Perform the evaluation.

        Apply evaluation functions in order according to `eval_func`. Also,
        make standard output.

        Parameters
        ----------
        y : numpy.nddary
            Ground truth

        yhat : numpy.nddary
            Predicted value

        q : float
            Quantile

        est : sklearn.pipeline.Pipeline
            The model to be saved.

        verbose : bool
            Whether to display standard output or not. Default is True.

        digit : int
            Number of digits of standard output. Default is 5.

        Returns
        -------
        dict
            Results of evaluation
        """
        for metrics, func in self.eval_funcs.items():
            if metrics == "PINBALL":
                result = func(y, yhat, q)
            else:
                result = func(y, yhat, k=len(est["CB"].feature_groups))

            self.result[metrics] = result
            if verbose:
                _logger.info(f"[{metrics}]: {round(result, digit)}\n")

        return self.result
