import logging

import abc
import copy
import six
import numpy as np
<<<<<<< HEAD
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import nbinom, poisson
from scipy.stats._distn_infrastructure import rv_frozen     # 型ヒントのためにクラスをimportしている...
=======
import pickle
import os
>>>>>>> tornado

from .evaluator import Evaluator, QuantileEvaluator
from .logger import ForwardLogger, PriorPredForwardLogger
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
            seed=0,
            save_dir="./models",
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
        mng_attr = self.manager.get_params()
        round2 = mng_attr["second_round"]
        logger = ForwardLogger(save_dir, log_policy, ["SKIP", round2])
        evaluator = Evaluator()

        # create dataset
        train, validation = self.data_deliveler.generate(
            target,
            self.manager.is_time_series,
            test_size,
            seed=seed,
            )

        # initialize model setting
        self.manager.init(train, target)

        # train
        logger_params = self.tornado(target, validation, logger, evaluator, verbose)

        # best model
        best_model_name = f'model_{logger_params["log_data"]["iter"]}'
        model_dir = logger_params["model_dir"]
        model_path = os.path.join(model_dir, best_model_name)
        with open(model_path, "rb") as f:
            self.estimator = pickle.load(f)

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

        return logger.get_params()

    def predict(self, X):
        X = self.data_deliveler.generate(X)
        pred = self.estimator.predict(X)

        return pred

    def predict_proba(self, target, output="proba") -> Union[rv_frozen, tuple(list, list)]:
        X, _ = self.data_deliveler.generate(target, is_time_series=self.manager.is_ts, pred=True)
        # X = self.data_deliveler.generate(X)
        pred = self.estimator.predict(X.copy())

        mng_params = self.manager.get_params()
        dist = mng_params["dist"]

        if dist == "poisson":
            pd_func = poisson(pred)

        elif dist == "nbinom":
            X_train = copy.deepcopy(self.manager.X)
            y_train = copy.deepcopy(self.manager.y)
            self.manager.set_params({"dist": "nbinomc"})

            features = self.estimator[1].feature_groups
            tmp = list()
            for feature in features:
                if isinstance(feature, tuple):
                    for f in feature:
                        tmp.append(f)
                else:
                    tmp.append(feature)
            unique_feature = list(set(tmp))
            param = dict()
            for feature in unique_feature:
                property = mng_params["init_model_attr"]["feature_properties"][feature]
                param[feature] = property
            self.manager.set_feature_property(param, verbose=False)

            nbinomc = self.manager.build()
            X_train['yhat_mean'] = self.estimator.predict(X_train.copy())
            X_train["yhat_mean_feature"] = X_train['yhat_mean']
            X["yhat_mean"] = pred
            X["yhat_mean_feature"] = X["yhat_mean"]
            nbinomc.fit(X_train.copy(), np.float64(y_train))
            c = nbinomc.predict(X)
            var = X['yhat_mean'] + c * X['yhat_mean'] * X['yhat_mean']

            p = np.minimum(np.where(var > 0, pred / var, 1 - 1e-8), 1 - 1e-8)
            n = np.where(var > 0, pred * p / (1 - p), 1)

            pd_func = nbinom(n, p)

        if output == "func":
            return pd_func

        elif output == "proba":
            args = np.array(pd_func.__dict__["args"]).T
            pmfs = []
            xs = []
            for i in range(args.shape[0]):
                x = np.linspace(nbinom.ppf(0.01, *args[i]),
                                nbinom.ppf(0.99, *args[i])).astype(int)
                pmf = nbinom.pmf(x, *args[i])
                xs.append(x)
                pmfs.append(pmf)
            return pmfs, xs


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
        mng_attr = self.manager.get_params()
        round1 = mng_attr["first_round"]
        round2 = mng_attr["second_round"]
        logger = ForwardLogger(save_dir, log_policy, [round1, round2])
        evaluator = Evaluator()

        # create dataset
        train, validation = self.data_deliveler.generate(
            target,
            self.manager.is_time_series,
            test_size,
            seed=seed,
        )

        # initialize training setting
        self.manager.init(train, target)

        # single variable regression analysis to search valid features
        _logger.info(f"\n=== [ROUND] {round1} ===\n")
        logger_params = self.tornado(target, validation, logger, evaluator, verbose)

        # pick up
        threshold = 2  # basically, 2 would be better for threshold

        valid_feature = dict()
        for feature, eval_result in logger_params["CODs"].items():
            if eval_result["F"] > threshold:
                valid_feature[feature] = eval_result

        mng_attr = self.manager.get_params()
        base_feature = mng_attr["init_model_attr"]["features"]
        interaction = [x for x in valid_feature.keys() if isinstance(x, tuple)]
        explanatory_variables = base_feature + interaction

        # model setting for multiple variable regression
        update_params = {
            "mode": round2,
            "experiment": 0,
            "target_features": explanatory_variables,
            "X": mng_attr["init_model_attr"]["X"].copy(),
        }
        self.manager.set_params(update_params)

        # clearing for next training
        logger.reset_count()
        evaluator.clear()

        # multiple variable regression (main training)
        _logger.info(f"\n=== [ROUND] {round2} ===\n")
        logger_params = self.tornado(target, validation, logger, evaluator, verbose)

        # best model
        best_model_name = f'model_{logger_params["log_data"]["iter"]}'
        model_dir = logger_params["model_dir"]
        model_path = os.path.join(model_dir, best_model_name)
        with open(model_path, "rb") as f:
            self.estimator = pickle.load(f)

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

        return logger.get_params()

    def predict(self, X):
        X = self.data_deliveler.generate(X)
        pred = self.estimator.predict(X)

        return pred

    def predict_proba(self, target, output="proba") -> Union[rv_frozen, tuple(list, list)]:
        X, _ = self.data_deliveler.generate(target, is_time_series=self.manager.is_ts, pred=True)
        # X = self.data_deliveler.generate(X)
        pred = self.estimator.predict(X.copy())

        mng_params = self.manager.get_params()
        dist = mng_params["dist"]

        if dist == "poisson":
            pd_func = poisson(pred)

        elif dist == "nbinom":
            X_train = copy.deepcopy(self.manager.X)
            y_train = copy.deepcopy(self.manager.y)
            self.manager.set_params({"dist": "nbinomc"})

            features = self.estimator[1].feature_groups
            tmp = list()
            for feature in features:
                if isinstance(feature, tuple):
                    for f in feature:
                        tmp.append(f)
                else:
                    tmp.append(feature)
            unique_feature = list(set(tmp))
            param = dict()
            for feature in unique_feature:
                property = mng_params["init_model_attr"]["feature_properties"][feature]
                param[feature] = property
            self.manager.set_feature_property(param, verbose=False)

            nbinomc = self.manager.build()
            X_train['yhat_mean'] = self.estimator.predict(X_train.copy())
            X_train["yhat_mean_feature"] = X_train['yhat_mean']
            X["yhat_mean"] = pred
            X["yhat_mean_feature"] = X["yhat_mean"]
            nbinomc.fit(X_train.copy(), np.float64(y_train))
            c = nbinomc.predict(X)
            var = X['yhat_mean'] + c * X['yhat_mean'] * X['yhat_mean']

            p = np.minimum(np.where(var > 0, pred / var, 1 - 1e-8), 1 - 1e-8)
            n = np.where(var > 0, pred * p / (1 - p), 1)

            pd_func = nbinom(n, p)

            self.manager.set_params({"dist": "nbinom"})

        if output == "func":
            return pd_func

        elif output == "proba":
            args = np.array(pd_func.__dict__["args"]).T
            pmfs = []
            xs = []
            for i in range(args.shape[0]):
                x = np.linspace(nbinom.ppf(0.01, *args[i]),
                                nbinom.ppf(0.99, *args[i])).astype(int)
                pmf = nbinom.pmf(x, *args[i])
                xs.append(x)
                pmfs.append(pmf)
            return pmfs, xs


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
        mng_attr = self.manager.get_params()
        round1 = mng_attr["first_round"]
        round2 = mng_attr["second_round"]
        logger = PriorPredForwardLogger(save_dir, log_policy, [round1, round2])
        evaluator = QuantileEvaluator()

        # create dataset
        train, validation = self.data_deliveler.generate(
            target,
            self.manager.is_time_series,
            test_size,
            seed=seed,
        )

        # initialize training setting
        self.manager.init(train, target)

        # single variables model for interaction search
        _logger.info(f"\n=== [ROUND] {round1} ===\n")
        args = {"quantile": self.quantile[0]}
        base_model = self.tornado(target, validation, logger, evaluator, verbose, args)

        X_train = mng_attr["init_model_attr"]["X"].copy()
        X_valid = validation.drop(target, axis=1)
        pred_train = base_model.predict(X_train.copy())
        pred_valid = base_model.predict(X_valid.copy())

        # change mode
        col = "prior_pred"
        X_train[col] = pred_train
        init_model_attr = mng_attr["init_model_attr"]
        init_model_attr["X"] = X_train.copy()
        model_params = {k: v for k, v in mng_attr["model_params"].items()}
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
        X_train = mng_attr["init_model_attr"]["X"]
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
        X = copy.deepcopy(mng_attr["init_model_attr"]["X"])
        y = copy.deepcopy(mng_attr["init_model_attr"]["y"])
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
