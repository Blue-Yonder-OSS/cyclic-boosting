import glob
import os
import shutil
import numpy as np
import pandas as pd
from cyclic_boosting.tornado.trainer.logger import ForwardLogger, PriorPredForwardLogger
from cyclic_boosting.pipelines import (
    pipeline_CBPoissonRegressor,
)
from cyclic_boosting import flags, observers
import pytest


@pytest.fixture(scope="module")
def prepare_log_data() -> dict:
    log_data = {
        "iter": 100,
        "features": ["dummy"],
        "feature_properties": {"dummy": flags.IS_CONTINUOUS},
        "smoothers": dict(),
        "metrics": {"COD": 1.0, "F": 1.0, "PINBALL": 1.0},
    }
    return log_data


@pytest.fixture(scope="module")
def prepare_tornado_modules() -> tuple:
    # manager = ForwardSelectionManager()
    manager_attr = {
        "end": 100,
        "experiment": 1,
        "mode": "first",
        "features": ["dummy"],
        "feature_properties": {"dummy": flags.IS_CONTINUOUS},
        "smoothers": dict(),
    }
    col = "dummy"
    estimator = pipeline_CBPoissonRegressor(
        feature_groups=[col],
        feature_properties={col: flags.IS_CONTINUOUS},
        observers=[
            observers.PlottingObserver(iteration=1),
            observers.PlottingObserver(iteration=-1),
        ],
    )
    X = pd.DataFrame(np.arange(10), columns=[col])
    y = np.arange(10)
    estimator.fit(X, y)
    return manager_attr, estimator


def test_forward_logger(prepare_log_data, prepare_tornado_modules) -> None:
    log_data = prepare_log_data
    manager_attr, est = prepare_tornado_modules

    # parameters
    save_dir = "./save_dir"
    f_name = "temp"
    criterion = "COD"
    first_round, second_round = "first", "second"
    eval = log_data["metrics"]
    bench_mark = {k: (v - 1) for k, v in log_data["metrics"].items()}

    logger = ForwardLogger(save_dir=save_dir, criterion=criterion)

    desired = {
        "iter": 0,
        "save_dir": save_dir,
        "criterion": criterion,
        "first_round": first_round,
        "second_round": second_round,
        "log_data": dict(),
        "CODs": dict(),
        "model_dir": None,
    }
    assert logger.get_attr() == desired

    params = {"log_data": log_data}
    desired.update(params)
    logger.set_attr(params)
    assert logger.get_attr()["log_data"] == log_data
    del desired

    logger.make_dir()
    assert os.path.isdir(save_dir)

    logger.make_model_dir()
    model_dir = os.path.join(save_dir, f"model_{log_data['iter']}")
    assert os.path.isdir(model_dir)

    ext = ".pkl"
    logger.save_model(est, f_name + ext)
    assert os.path.exists(f_name + ext)
    os.remove(f_name + ext)

    ext = ".txt"
    logger.save_metrics(f_name + ext)
    assert os.path.exists(f_name + ext)
    os.remove(f_name + ext)

    logger.save_setting(f_name + ext)
    assert os.path.exists(f_name + ext)
    os.remove(f_name + ext)

    ext = ".pdf"
    logger.save_plot(est, f_name + ext)
    assert os.path.exists(f_name + ext)
    os.remove(f_name + ext)

    logger.save(est)
    assert len(glob.glob(f"{model_dir}/*")) == 4
    shutil.rmtree(model_dir)

    logger.reset_count()
    assert logger.get_attr()["iter"] == 0

    try:
        logger.output(eval, None)
    except Exception as error:
        assert False, error

    params = {
        "iter": 100,
    }
    logger.set_attr(params)
    logger.hold(est, eval, manager_attr, verbose=True, save=True)
    assert log_data == logger.get_attr()["log_data"]
    ext = ".pkl"
    os.remove(os.path.join(save_dir, f_name + ext))

    # check for validation
    params = {"criterion": criterion, "bench_mark": {"metrics": bench_mark}}
    logger.set_attr(params)
    assert logger.validate(eval), f"{criterion} logic is wrong"

    criterion = "PINBALL"
    bench_mark = {k: (v + 1) for k, v in log_data["metrics"].items()}
    params = {"criterion": criterion, "bench_mark": {"metrics": bench_mark}}
    logger.set_attr(params)
    assert logger.validate(eval), f"{criterion} logic is wrong"

    # check for 2 steps round training
    # first round
    param = {
        "iter": 100,
        "first_round": first_round,
        "bench_mark": {"metrics": bench_mark},
    }
    logger.set_attr(param)
    try:
        logger.log(est, eval, manager_attr)
    except Exception as error:
        assert False, error

    # second round
    param = {
        "second_round": second_round,
        "bench_mark": {"metrics": bench_mark},
    }
    logger.set_attr(param)
    param = {
        "mode": second_round,
    }
    manager_attr.update(param)
    try:
        logger.log(est, eval, manager_attr)
    except Exception as error:
        assert False, error


def test_prior_pred_forward_logger(prepare_log_data, prepare_tornado_modules) -> None:
    log_data = prepare_log_data
    manager_attr, est = prepare_tornado_modules

    # parameters
    save_dir = "./save_dir"
    f_name = "temp"
    criterion = "COD"
    first_round, second_round = "first", "second"
    eval = log_data["metrics"]
    bench_mark = {k: (v - 1) for k, v in log_data["metrics"].items()}

    logger = PriorPredForwardLogger(save_dir=save_dir, criterion=criterion)

    param = {
        "mode": first_round,
        "experiment": 1,
        "end": 1,
        "features": ["dummy"],
    }
    manager_attr.update(param)
    try:
        logger.output(eval, manager_attr)
    except Exception as error:
        assert False, error

    log_data = prepare_log_data
    params = {
        "iter": 100,
    }
    logger.set_attr(params)
    logger.hold(est, eval, manager_attr, verbose=True, save=True)
    assert log_data == logger.get_attr()["log_data"]
    ext = ".pkl"
    os.remove(os.path.join(save_dir, f_name + ext))

    # check for validation
    criterion = "COD"
    params = {"criterion": criterion, "bench_mark": {"metrics": bench_mark}}
    logger.set_attr(params)
    logger.validate(eval)
    assert logger.validate(eval), f"{criterion} logic is wrong"

    criterion = "PINBALL"
    bench_mark = {k: (v + 1) for k, v in log_data["metrics"].items()}
    params = {"criterion": criterion, "bench_mark": {"metrics": bench_mark}}
    logger.set_attr(params)
    logger.validate(eval)
    assert logger.validate(eval), f"{criterion} logic is wrong"

    # check for 2 steps round training
    # first round
    param = {
        "iter": 100,
        "first_round": first_round,
        "bench_mark": {"metrics": bench_mark},
    }
    logger.set_attr(param)
    try:
        logger.log(est, eval, manager_attr)
    except Exception as error:
        assert False, error

    # second round
    param = {
        "second_round": second_round,
        "bench_mark": {"metrics": bench_mark},
    }
    logger.set_attr(param)
    param = {
        "mode": second_round,
    }
    manager_attr.update(param)
    try:
        logger.log(est, eval, manager_attr)
    except Exception as error:
        assert False, error
