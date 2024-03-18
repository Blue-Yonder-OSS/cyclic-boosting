import numpy as np

from cyclic_boosting.tornado.trainer.evaluator import Evaluator, QuantileEvaluator
from cyclic_boosting.pipelines import pipeline_CBPoissonRegressor, pipeline_CBMultiplicativeQuantileRegressor
from cyclic_boosting import flags


def test_evaluator() -> None:
    yhat = np.arange(1, 10)
    y = yhat - (1 / 2 * yhat)

    estimator = pipeline_CBPoissonRegressor(feature_groups=["dummy"], feature_properties={"dummy": flags.IS_CONTINUOUS})

    evaluator = Evaluator()
    try:
        _ = evaluator.eval(y, yhat, estimator)
    except Exception as e:
        assert False, e


def test_quantile_evaluator() -> None:
    yhat = np.arange(1, 10)
    y = yhat - (1 / 2 * yhat)
    quantile = 0.5

    estimator = pipeline_CBMultiplicativeQuantileRegressor(
        quantile=quantile, feature_groups=["dummy"], feature_properties={"dummy": flags.IS_CONTINUOUS}
    )

    evaluator = QuantileEvaluator()
    try:
        _ = evaluator.eval(y, yhat, quantile, estimator)
    except Exception as e:
        assert False, e
