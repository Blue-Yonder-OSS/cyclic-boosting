import logging

import abc
import copy
import six
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import nbinom, poisson
from scipy.stats._distn_infrastructure import rv_frozen     # 型ヒントのためにクラスをimportしている...

from .evaluator import Evaluator, QuantileEvaluator
from .logger import Logger, BFForwardLogger
from cyclic_boosting.quantile_matching import J_QPD_S

from typing import List, Union


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
        evaluator = Evaluator()
        logger = Logger(save_dir, log_policy)

        # create dataset
        dataset = self.data_deliveler.generate()
        train, validation = train_test_split(dataset,
                                             test_size=test_size,
                                             random_state=seed,
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
            evaluator.eval(y_valid, y_pred, estimator, verbose)

            # log
            logger.log(estimator, evaluator, self.manager)

        return estimator

    def predict(self, X):
        X = self.data_deliveler.generate(X)
        pred = self.estimator.predict(X)

        return pred

    def predict_proba(self, target, output="proba") -> Union[rv_frozen, np.ndarray]:   # 型ヒントのためにクラスをimportしている...
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
        mng_params = self.manager.get_params()
        round1 = mng_params["first_round"]
        round2 = mng_params["second_round"]
        logger = Logger(save_dir, log_policy, [round1, round2])
        evaluator = Evaluator()

        # create dataset
        dataset = self.data_deliveler.generate()
        train, validation = train_test_split(dataset,
                                             test_size=test_size,
                                             random_state=seed,
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
            "X": mng_params["init_model_attr"]["X"],
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

            X = copy.deepcopy(self.manager.X)
            y = copy.deepcopy(self.manager.y)
            _ = estimator.fit(X, y)

            y_valid = np.asarray(valid_data[target])
            X_valid = valid_data.drop(target, axis=1)
            y_pred = estimator.predict(X_valid)
            evaluator.eval(y_valid, y_pred, estimator, verbose)

            logger.log(estimator, evaluator, self.manager)

            # update param
            mng_params = self.manager.get_params()
            if mng_params["mode"] == mng_params["second_round"]:
                keys = ["features", "feature_properties"]
                logger_params = logger.get_params()
                update_params = {k: logger_params["log_data"][k] for k in keys}
                self.manager.set_params(update_params)

        return estimator

    def predict(self, X):
        X = self.data_deliveler.generate(X)
        pred = self.estimator.predict(X)

        return pred

    def predict_proba(self, target, output="proba") -> Union[rv_frozen, np.ndarray]:   # 型ヒントのためにクラスをimportしている...
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
    def __init__(self, DataModule, TornadoModule, quantile=0.1):
        super().__init__(DataModule, TornadoModule)
        self.quantile = quantile
        self.alpha = [quantile, 0.5, 1 - quantile]
        self.loss = 0.0
        self.estimators = list()

        if self.quantile >= 0.5:
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
        dataset = self.data_deliveler.generate()
        train, validation = train_test_split(dataset,
                                             test_size=test_size,
                                             random_state=seed)

        # initialize training setting
        combination = 2
        self.manager.init(train, target, n_comb=combination)

        # single variables model for interaction search
        _logger.info(f"\n=== [ROUND] {round1} ===\n")
        args = {"quantile": 0.5}
        base_model = self.tornado(target, validation, logger, evaluator, verbose, args)

        #  prediction with base single features
        X_original = copy.deepcopy(mng_params["init_model_attr"]["X"])
        X_valid = validation.drop(target, axis=1)
        pred_train = base_model.predict(X_original.copy())
        pred_valid = base_model.predict(X_valid.copy())

        # change mode
        col = "prior_pred"
        X_original = mng_params["init_model_attr"]["X"]
        X_original[col] = pred_train
        model_params_new = mng_params["model_params"]
        model_params_new["prior_prediction_column"] = col
        update_params = {"mode": round2,
                         "experiment": 0,
                         "X": X_original,
                         "model_params": model_params_new,
                         }
        self.manager.set_params(update_params)
        validation[col] = pred_valid

        # clearing for next training
        logger.reset_count()
        evaluator.clear()

        # brute force model search
        _logger.info(f"\n=== [ROUND] {round2} ===\n")
        _ = self.tornado(target, validation, logger, evaluator, verbose, args)
        _logger.info(f"\nDetect {len(logger.valid_interactions)} interactions\n")

        # model setting for QPD estimation
        _logger.info("\n=== [ROUND] QPD estimator training ===\n")
        fp = mng_params["init_model_attr"]["feature_properties"]
        base = list(fp.keys())
        logger_params = logger.get_params()
        feature = base + logger_params["valid_interactions"]

        # build 3 models for QPD
        X_original = mng_params["init_model_attr"]["X"]
        model_params_new = {"feature_properties": fp, "feature_groups": feature}
        for a in self.alpha:
            model_params_new["quantile"] = a
            update_params = {"X": X_original, "model_params": model_params_new}
            self.manager.set_params(update_params)
            self.manager.drop_unused_features()
            est = self.manager.build()
            self.estimators.append(est)

        # train
        X = copy.deepcopy(mng_params["init_model_attr"]["X"])
        y = copy.deepcopy(mng_params["init_model_attr"]["y"])
        names = {0: "lower quantile", 1: "median", 2: "upper quantile"}
        for i, est in enumerate(self.estimators):
            _logger.info(f"\n=== {names[i]} model ===\n")
            _ = est.fit(X.copy(), y)

    def tornado(self, target, valid_data, logger, evaluator, verbose, args):
        while self.manager.manage():
            estimator = self.manager.build()

            X = copy.deepcopy(self.manager.X)
            y = copy.deepcopy(self.manager.y)
            _ = estimator.fit(X, y)

            y_valid = np.asarray(valid_data[target])
            X_valid = valid_data.loc[:, X.columns]
            y_pred = estimator.predict(X_valid)
            evaluator.eval(y_valid, y_pred, args["quantile"], estimator, verbose=False)

            logger.log(estimator, evaluator, self.manager, verbose=verbose)

        return estimator

    def predict(self, X, quantile="median") -> np.ndarray:
        X = self.data_deliveler.generate(X)

        est_ix = {"lower": 0, "median": 1, "upper": 2}
        est = self.estimators[est_ix[quantile]]
        quantile_values = est.predict(X)

        return quantile_values

    def predict_proba(self, X, output="proba") -> Union[List[J_QPD_S], pd.DataFrame]:
        X = self.data_deliveler.generate(X)

        quantile_values = list()
        for est in self.estimators:
            pred = est.predict(X)
            quantile_values.append(pred)

        low = np.array(quantile_values[0])
        median = np.array(quantile_values[1])
        high = np.array(quantile_values[2])

        # handling for cross switching
        if (np.any(low > median)) or np.any((high < median)):
            _logger.warning(
                "The SPT values are not monotonically increasing,"
                "  each SPT will be replaced by mean value")
            idx = (
                np.where((low > median), True, False) +
                np.where((high < median), True, False)
            )
            low[idx] = np.nanmean(low)
            median[idx] = np.nanmean(median)
            high[idx] = np.nanmean(high)

        individual_qpds = list()
        lower_quantile = self.alpha[0]
        for lq, mq, hq in zip(low, median, high):
            dist = J_QPD_S(lower_quantile, lq, mq, hq)
            individual_qpds.append(dist)

        if output == "func":
            return individual_qpds

        elif output == "proba":
            ix = np.arange(0.05, 1.0, 0.05)
            proba_df = pd.DataFrame(columns=ix)
            for i in ix:
                proba = [qpd.ppf(i) for qpd in individual_qpds]
                proba_df[i] = proba

            return proba_df
