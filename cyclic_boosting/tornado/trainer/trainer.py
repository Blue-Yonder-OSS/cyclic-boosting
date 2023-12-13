import logging

import abc
import copy
import six
import numpy as np
from sklearn.model_selection import train_test_split

from .evaluator import Evaluator, QuantileEvaluator
from .logger import Logger, BFForwardLogger
from cyclic_boosting.quantile_matching import J_QPD_S, J_QPD_B


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
    def fit(self, target, test_size=0.2, seed=0, save_dir="./models", log_policy="COD", verbose=True):
        pass

    @abc.abstractmethod
    def predict(self, X):
        pass

    def train(self, target, valid_data, logger, evaluator, verbose):
        while self.manager.manage():
            estimator = self.manager.build()

            # train
            X = copy.deepcopy(self.manager.X)
            y = copy.deepcopy(self.manager.y)
            _ = estimator.fit(X, y)

            # validation
            y_valid = np.asarray(valid_data[target])
            X_valid = valid_data.drop(target, axis=1)
            yhat = estimator.predict(X_valid)
            evaluator.eval(y_valid, yhat, estimator, verbose)

            # log
            logger.log(estimator, evaluator, self.manager)


class Tornado(TornadoBase):
    def __init__(self, DataModule, TornadoModule):
        super().__init__(DataModule, TornadoModule)
        self.estimator = None

    def fit(self, target, test_size=0.2, seed=0, save_dir="./models", log_policy="COD", verbose=True):
        # build logger and evaluator
        evaluator = Evaluator()
        logger = Logger(save_dir, log_policy)

        # create dataset
        dataset = self.data_deliveler.generate()
        train, validation = train_test_split(dataset, test_size=test_size, random_state=seed)

        # initialize model setting
        self.manager.init(train, target)

        # train
        self.estimator = self.train(target, validation, logger, evaluator, verbose)

    def predict(self, X):
        X = self.data_deliveler.generate(X)
        pred = self.estimator.predict(X)

        return pred


class ForwardTrainer(TornadoBase):
    def __init__(self, DataModule, TornadoModule):
        super().__init__(DataModule, TornadoModule)
        self.estimator = None

    def fit(self, target, test_size=0.2, seed=0, save_dir="./models", log_policy="COD", verbose=True):
        # build logger and evaluator
        mng_params = self.manager.get_params()
        round1 = mng_params["first_round"]
        round2 = mng_params["second_round"]
        logger = Logger(save_dir, log_policy, [round1, round2])
        evaluator = Evaluator()

        # create dataset
        dataset = self.data_deliveler.generate()
        train, validation = train_test_split(dataset, test_size=test_size, random_state=seed)

        # initialize training setting
        self.manager.init(train, target)

        # sigle variable regression analysis to search valid features
        _logger.info(f"\n=== [ROUND] {round1} ===\n")
        _ = self.train(target, validation, logger, evaluator, verbose)

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
        self.estimator = self.train(target, validation, logger, evaluator, verbose)

    def train(self, target, valid_data, logger, evaluator, verbose):
        while self.manager.manage():
            estimator = self.manager.build()

            # train
            X = copy.deepcopy(self.manager.X)
            y = copy.deepcopy(self.manager.y)
            _ = estimator.fit(X, y)

            # validation
            y_valid = np.asarray(valid_data[target])
            X_valid = valid_data.drop(target, axis=1)
            yhat = estimator.predict(X_valid)
            evaluator.eval(y_valid, yhat, estimator, verbose)

            # log
            logger.log(estimator, evaluator, self.manager)

            # update
            mng_params = self.manager.get_params()
            if mng_params["mode"] == mng_params["second_round"]:
                keys = ["features", "feature_properties"]
                logger_params = logger.get_params()
                update_params = {k: logger_params["log_data"][k] for k in keys}
                self.manager.set_params(update_params)

        return estimator

    def predict(self, X):
        # create dataset
        X = self.data_deliveler.generate(X)
        pred = self.estimator.predict(X)

        return pred


class QPDForwardTrainer(TornadoBase):
    def __init__(self, DataModule, TornadoModule, quantile=0.1):
        super().__init__(DataModule, TornadoModule)
        self.quantile = quantile
        self.alpha = [quantile, 0.5, 1 - quantile]
        self.loss = 0.0
        self.estimators = []

        if self.quantile >= 0.5:
            raise ValueError("quantile must be quantile < 0.5")

    def fit(self, target, test_size=0.2, seed=0, save_dir="./models", log_policy="PINBALL", verbose=True):
        # build logger and evaluator
        mng_params = self.manager.get_params()
        round1 = mng_params["first_round"]
        round2 = mng_params["second_round"]
        logger = BFForwardLogger(save_dir, log_policy, [round1, round2])
        evaluator = QuantileEvaluator()

        # create dataset
        dataset = self.data_deliveler.generate()
        train, validation = train_test_split(dataset, test_size=test_size, random_state=seed)

        # initialize training setting
        combination = 2
        self.manager.init(train, target, n_comb=combination)

        # sigle variables model for interaction search
        _logger.info(f"\n=== [ROUND] {round1} ===\n")
        args = {"quantile": 0.5}
        base_model = self.train(target, validation, logger, evaluator, verbose, args)

        #  prediction with base sigle features
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
        update_params = {"mode": round2, "experiment": 0, "X": X_original, "model_params": model_params_new}
        self.manager.set_params(update_params)
        validation[col] = pred_valid

        # clearing for next training
        logger.reset_count()
        evaluator.clear()

        # brute force model search
        _logger.info(f"\n=== [ROUND] {round2} ===\n")
        _ = self.train(target, validation, logger, evaluator, verbose, args)
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

    def train(self, target, valid_data, logger, evaluator, verbose, args):
        while self.manager.manage():
            estimator = self.manager.build()

            # train
            X = copy.deepcopy(self.manager.X)
            y = copy.deepcopy(self.manager.y)
            _ = estimator.fit(X, y)

            # validation
            y_valid = np.asarray(valid_data[target])
            X_valid = valid_data.loc[:, X.columns]
            y_pred = estimator.predict(X_valid)
            evaluator.eval(y_valid, y_pred, args["quantile"], verbose=False)

            # log
            logger.log(estimator, evaluator, self.manager, verbose=verbose)

        return estimator

    def predict(self, X, output="ppf", ix=0.5):
        # create dataset
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
            _logger.warning("The SPT values are not monotonically increasing, each SPT will be replaced by mean value")
            idx = np.where((low > median), True, False) + np.where((high < median), True, False)
            low[idx] = np.nanmean(low)
            median[idx] = np.nanmean(median)
            high[idx] = np.nanmean(high)

        qpd = list()
        lower_quantile = self.alpha[0]
        for lq, mq, hq in zip(low, median, high):
            dist = J_QPD_S(lower_quantile, lq, mq, hq)
            qpd.append(dist)

        if output == "proba":
            return qpd

        elif output == "ppf":
            proba = list()
            for dist in qpd:
                proba.append(dist.ppf(ix))
            return np.array(proba)
