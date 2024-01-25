import logging

import abc
import copy
import six
import numpy as np
import pandas as pd

from .evaluator import Evaluator, QuantileEvaluator
from .logger import Logger, BFForwardLogger
from cyclic_boosting.quantile_matching import QPD_RegressorChain

# from typing import List, Union


_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(fmt="%(message)s"))
handler.terminator = ""
_logger.addHandler(handler)


@six.add_metaclass(abc.ABCMeta)
class TornadoBase:
    def __init__(self, DataModule, TornadoModule):
        self.data_deliveler = DataModule
        self.manager = TornadoModule

    @abc.abstractmethod
    def fit(self,
            target,
            test_size=0.2,
            seed=0, save_dir="./models",
            log_policy="COD",
            verbose=True,
            ):
        # write main fitting steps using Estimator, Evaluator, Logger, DataModule
        pass

    @abc.abstractmethod
    def tornado(self, X):
        # write cyclic model training steps with handling by TornadoModule
        pass

    @abc.abstractmethod
    def predict(self, X):
        # write prediction steps
        pass


class Tornado(TornadoBase):
    def __init__(self, DataModule, TornadoModule):
        super().__init__(DataModule, TornadoModule)
        self.estimator = None

    def fit(self,
            target,
            test_size=0.2,
            seed=0,
            save_dir="./models",
            log_policy="COD",
            verbose=True,
            ):
        # build logger and evaluator
        mng_params = self.manager.get_params()
        round2 = mng_params["second_round"]
        logger = Logger(save_dir, log_policy, ["SKIP", round2])
        evaluator = Evaluator()

        # create dataset
        train, validation = self.data_deliveler.generate(
            target,
            self.manager.is_ts,
            test_size,
            seed=seed,
            )

        # initialize model setting
        self.manager.init(train, target)

        # train
        self.estimator = self.tornado(target, validation, logger, evaluator, verbose)

    def tornado(self, target, valid_data, logger, evaluator, verbose):
        while self.manager.manage():
            estimator = self.manager.build()

            # train
            X = copy.deepcopy(self.manager.X)
            y = copy.deepcopy(self.manager.y)
            _ = estimator.fit(X, y)

            # validation
            y_valid = np.asarray(valid_data[target])
            X_valid = valid_data.drop(target, axis=1)
            y_pred = estimator.predict(X_valid)

            # log
            eval_history = evaluator.eval(y_valid, y_pred, estimator, verbose)
            mng_attr = self.manager.get_params()
            logger.log(estimator, eval_history, mng_attr)
            self.manager.clear()

            # update param
            if mng_attr["mode"] == mng_attr["second_round"]:
                keys = ["features", "feature_properties"]
                update_params = {k: logger.get_params()["log_data"][k] for k in keys}
                self.manager.set_params(update_params)

        return estimator

    def predict(self, X):
        X = self.data_deliveler.generate(X)
        pred = self.estimator.predict(X)

        return pred


class ForwardTrainer(TornadoBase):
    def __init__(self, DataModule, TornadoModule):
        super().__init__(DataModule, TornadoModule)
        self.estimator = None

    def fit(self,
            target,
            test_size=0.2,
            seed=0,
            save_dir="./models",
            log_policy="COD",
            verbose=True,
            ):
        # build logger and evaluator
        mng_params = self.manager.get_params()
        round1 = mng_params["first_round"]
        round2 = mng_params["second_round"]
        logger = Logger(save_dir, log_policy, [round1, round2])
        evaluator = Evaluator()

        # create dataset
        train, validation = self.data_deliveler.generate(
            target,
            self.manager.is_ts,
            test_size,
            seed=seed,
        )

        # initialize training setting
        self.manager.init(train, target)

        # single variable regression analysis to search valid features
        _logger.info(f"\n=== [ROUND] {round1} ===\n")
        _ = self.tornado(target, validation, logger, evaluator, verbose)

        # pick up
        threshold = 2  # basically, 2 would be better for threshold
        logger_params = logger.get_params()

        valid_feature = dict()
        for feature, eval_result in logger_params["CODs"].items():
            if eval_result["F"] > threshold:
                valid_feature[feature] = eval_result

        base_feature = mng_params["init_model_attr"]["features"]
        interaction = [x for x in valid_feature.keys() if isinstance(x, tuple)]
        explanatory_variables = base_feature + interaction

        # model setting for multiple variable regression
        update_params = {
            "mode": round2,
            "experiment": 0,
            "target_features": explanatory_variables,
            "X": mng_params["init_model_attr"]["X"].copy(),
        }
        self.manager.set_params(update_params)

        # clearing for next training
        logger.reset_count()
        evaluator.clear()

        # multiple variable regression (main training)
        _logger.info(f"\n=== [ROUND] {round2} ===\n")
        self.estimator = self.tornado(target, validation, logger, evaluator, verbose)

    def tornado(self, target, valid_data, logger, evaluator, verbose):
        while self.manager.manage():
            estimator = self.manager.build()

            # train
            X = copy.deepcopy(self.manager.X)
            y = copy.deepcopy(self.manager.y)
            _ = estimator.fit(X, y)

            # validation
            y_valid = np.asarray(valid_data[target])
            X_valid = valid_data.drop(target, axis=1)
            y_pred = estimator.predict(X_valid)
            evaluator.eval(y_valid, y_pred, estimator, verbose)

            # log
            eval_history = evaluator.eval(y_valid, y_pred, estimator, verbose)
            mng_attr = self.manager.get_params()
            logger.log(estimator, eval_history, mng_attr)
            self.manager.clear()

            # update param
            if mng_attr["mode"] == mng_attr["second_round"]:
                keys = ["features", "feature_properties"]
                update_params = {k: logger.get_params()["log_data"][k] for k in keys}
                self.manager.set_params(update_params)

        return estimator

    def predict(self, X):
        X = self.data_deliveler.generate(X)
        pred = self.estimator.predict(X)

        return pred


class QPDForwardTrainer(TornadoBase):
    def __init__(
            self,
            DataModule,
            TornadoModule,
            quantile=0.1,
            bound="U",
            lower=0.0,
            upper=1.0,
            ):
        super().__init__(DataModule, TornadoModule)
        self.quantile = [quantile, 0.5, 1 - quantile]
        self.bound = bound
        self.lower = lower
        self.upper = upper
        self.loss = 0.0
        self.est_qpd = None

        if quantile >= 0.5:
            raise ValueError("quantile must be quantile < 0.5")

    def fit(self,
            target,
            test_size=0.2,
            seed=0,
            save_dir="./models",
            log_policy="PINBALL",
            verbose=True,
            ):
        # build logger and evaluator
        mng_params = self.manager.get_params()
        round1 = mng_params["first_round"]
        round2 = mng_params["second_round"]
        logger = BFForwardLogger(save_dir, log_policy, [round1, round2])
        evaluator = QuantileEvaluator()

        # create dataset
        train, validation = self.data_deliveler.generate(
            target,
            self.manager.is_ts,
            test_size,
            seed=seed,
        )

        # initialize training setting
        self.manager.init(train, target)

        # single variables model for interaction search
        _logger.info(f"\n=== [ROUND] {round1} ===\n")
        args = {"quantile": self.quantile[0]}
        base_model = self.tornado(target, validation, logger, evaluator, verbose, args)

        X_train = mng_params["init_model_attr"]["X"].copy()
        X_valid = validation.drop(target, axis=1)
        pred_train = base_model.predict(X_train.copy())
        pred_valid = base_model.predict(X_valid.copy())

        # change mode
        col = "prior_pred"
        X_train[col] = pred_train
        init_model_attr = mng_params["init_model_attr"]
        init_model_attr["X"] = X_train.copy()
        model_params = {k: v for k, v in mng_params["model_params"].items()}
        model_params["prior_prediction_column"] = col
        update_params = {
            "mode": round2,
            "experiment": 0,
            "X": X_train.copy(),
            "model_params": model_params,
            "init_model_attr": init_model_attr
            }
        self.manager.set_params(update_params)
        validation[col] = pred_valid
        logger.reset_count()
        evaluator.clear()

        # brute force model search
        _logger.info(f"\n=== [ROUND] {round2} ===\n")
        _ = self.tornado(target, validation, logger, evaluator, verbose, args)
        _logger.info(f"\nDetect {len(logger.valid_interactions)} interactions\n")

        # model setting for QPD estimation
        _logger.info("\n=== [ROUND] QPD estimator training ===\n")
        base = [x for x in init_model_attr["feature_properties"].keys()]
        interaction = logger.get_params()["valid_interactions"]
        features = base + interaction

        # build 3 models for QPD
        X_train = mng_params["init_model_attr"]["X"]
        model_params = {
            "feature_properties": init_model_attr["feature_properties"],
            "feature_groups": features,
            }
        est_quantiles = list()
        for quantile in self.quantile:
            model_params["quantile"] = quantile
            update_params = {"X": X_train, "model_params": model_params}
            self.manager.set_params(update_params)
            self.manager.drop_unused_features()
            est = self.manager.build()
            est_quantiles.append(est)

        # train
        X = copy.deepcopy(mng_params["init_model_attr"]["X"])
        y = copy.deepcopy(mng_params["init_model_attr"]["y"])
        self.est_qpd = QPD_RegressorChain(
            est_lowq=est_quantiles[0],
            est_median=est_quantiles[1],
            est_highq=est_quantiles[2],
            bound=self.bound,
            l=self.lower,
            u=self.upper,
        )
        _ = self.est_qpd.fit(X.copy(), y)

    def tornado(self, target, valid_data, logger, evaluator, verbose, args):
        while self.manager.manage():
            estimator = self.manager.build()

            X = copy.deepcopy(self.manager.X)
            y = copy.deepcopy(self.manager.y)
            _ = estimator.fit(X, y)

            y_valid = np.asarray(valid_data[target])
            X_valid = valid_data.loc[:, X.columns]
            y_pred = estimator.predict(X_valid)

            eval_history = evaluator.eval(
                y_valid,
                y_pred,
                args["quantile"],
                estimator,
                verbose=False,
                )
            mng_attr = self.manager.get_params()

            # log
            logger.log(estimator, eval_history, mng_attr, verbose=verbose)
            self.manager.clear()

        return estimator

    def predict(self, X, quantile="median") -> np.ndarray:
        X = self.data_deliveler.generate(X)

        est_ix = {"lower": 0, "median": 1, "upper": 2}
        est = self.estimators[est_ix[quantile]]
        quantile_values = est.predict(X)

        return quantile_values
