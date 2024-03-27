"""Control Recursive Learning in Tornado."""

from __future__ import annotations

import abc
import copy
import logging
import os
import pickle
from typing import TYPE_CHECKING, Union

import numpy as np
import pandas as pd
import six
from scipy.stats import nbinom, poisson

from cyclic_boosting.quantile_matching import QPD_RegressorChain

from .evaluator import Evaluator, QuantileEvaluator
from .logger import ForwardLogger, PriorPredForwardLogger

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline
    from quantile_matching import J_QPD_S, J_QPD_B


_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(fmt="%(message)s"))
handler.terminator = ""
_logger.addHandler(handler)


@six.add_metaclass(abc.ABCMeta)
class TornadoBase:
    """Base class to control learning in Tornado.

    Attributes
    ----------
    manager : One of the classes in the manager.py
        Base class is :class:`TornadoManager`.
    """

    def __init__(self, TornadoManager):
        self.manager = TornadoManager

    @abc.abstractmethod
    def fit(
        self,
        target,
        train,
        validation,
        save_dir="./models",
        criterion="COD",
        verbose=True,
    ) -> None:
        """Abstract method for model fitting."""
        # write main fitting steps using Estimator, Evaluator, Logger
        pass

    @abc.abstractmethod
    def tornado(self, X):
        """Abstract method for model training."""
        # write cyclic model training steps with handling by TornadoManager
        pass

    @abc.abstractmethod
    def predict(self, X):
        """Abstract method for predict with the trained model."""
        # write prediction steps
        pass


class InteractionSearchModel(TornadoBase):
    """Class that controls learning in Tornado.

    This class fits a model with single features as the base, then adds
    interaction terms one by one at each iteration to fit the model, and
    updates the model when it is better than the model of the previous
    iteration. This process is repeated to search for the best model. This
    class inherits from :class:`TornadoBase`.

    Attributes
    ----------
    manager : One of the manager classes
        Class with TornadoManager as base class.

    estimator : sklearn.pipeline.Pipeline
        Pipeline with steps including binning and CB.

    nbinomc : sklearn.pipeline.Pipeline
        Estimator for parameter c of the negative binomial distribution.
    """

    def __init__(self, TornadoManager):
        super().__init__(TornadoManager)
        self.estimator = None
        self.nbinomc = None

    def fit(
        self,
        target,
        train,
        validation,
        save_dir="./models",
        criterion="COD",
        verbose=True,
    ) -> None:
        """Control the sequence of learning of the model.

        Prepare instances and datasets for training, then train the model.

        Parameters
        ----------
        target : str
            Name of the target valiable.

        train : pandas.DataFrame
            Train set.

        validation : pandas.DataFrame
            Validation set.

        save_dir : str
            Path of the directory where the model will be stored. Default is
            "./models". Under this, a directory containing the model number
            (number of iterations) is created, in which information about the
            model is stored.

        criterion : str
            Best Model Evaluation Policy. 'COD' or 'PINBALL'. Default is 'COD'.

        verbose : bool
            Whether to display standard output or not. Default is True.
        """
        target = target.lower()
        evaluator = Evaluator()
        mgr_attr = self.manager.get_attr()
        training_round = ["SKIP", mgr_attr["second_round"]]
        logger = ForwardLogger(criterion, save_dir, training_round)

        # recursive interaction search
        self.manager.init(train, target)
        stage = "\n=== [{0}] {1} ===\n"
        status = "START"
        _logger.info(stage.format(status, mgr_attr["second_round"]))

        logger_attr = self.tornado(
            target,
            validation,
            logger,
            evaluator,
            verbose,
        )
        best_model_path = os.path.join(
            logger_attr["model_dir"],
            f'model_{logger_attr["log_data"]["iter"]}.pkl',
        )
        with open(best_model_path, "rb") as f:
            self.estimator = pickle.load(f)

        status = "END"
        _logger.info(stage.format(status, mgr_attr["second_round"]))

        # c parameter estimator for proba prediction
        mgr_attr = self.manager.get_attr()
        dist = mgr_attr["dist"]
        if dist == "nbinom":
            estimator_params = self.estimator[-1].__dict__
            update_params = {
                "dist": "nbinomc",
                "feature_properties": estimator_params["feature_properties"],
                "features": estimator_params["feature_groups"],
                "observers": estimator_params["observers"],
                "smoothers": estimator_params["smoother_choice"].explicit_smoothers,
            }
            self.manager.set_attr(update_params)

            self.nbinomc = self.manager.build()

            X_train = copy.deepcopy(self.manager.X)
            y_train = copy.deepcopy(self.manager.y)
            X_train["yhat_mean"] = self.estimator.predict(X_train.copy())
            X_train["yhat_mean_feature"] = X_train["yhat_mean"]
            self.nbinomc.fit(X_train.copy(), np.float64(y_train))

            self.manager.set_attr({"dist": "nbinom"})

    def tornado(self, target, validation, logger, evaluator, verbose) -> Pipeline:
        """Recursive learning in Tornado.

        Recursive learning is performed by changing the explanatory variables
        in an ordinary Tornado. The process includes building, fitting,
        validating, and updating the model at each iteration. The process is
        repeated to search for the best model.

        Parameters
        ----------
        target : str
            Name of the target valiable.

        validation : pandas.DataFrame
            Validation set.

        logger : logger.Logger
            Instances to manage learning logs.

        evaluator : evaluator.Evaluator
            Instance to evaluate models

        verbose : bool
            Whether to display standard output or not.

        Returns
        -------
        sklearn.pipeline.Pipeline
            Tornado estimator. Pipeline with steps including binning and CB.
        """
        while self.manager.manage():
            estimator = self.manager.build()

            # train
            X = copy.deepcopy(self.manager.X)
            y = copy.deepcopy(self.manager.y)
            _ = estimator.fit(X, y)

            # validation
            y_valid = np.asarray(validation[target])
            X_valid = validation.drop(target, axis=1)
            y_pred = estimator.predict(X_valid)

            # log
            eval_history = evaluator.eval(y_valid, y_pred, estimator, verbose=False)
            mgr_attr = self.manager.get_attr()
            logger.log(estimator, eval_history, mgr_attr, verbose=verbose)
            self.manager.clear()

            # update param
            if mgr_attr["mode"] == mgr_attr["second_round"]:
                keys = ["features", "feature_properties"]
                update_params = {k: logger.get_attr()["log_data"][k] for k in keys}
                self.manager.set_attr(update_params)

        return logger.get_attr()

    def predict(self, X) -> np.ndarray:
        """Predict using the best model explored in Tornado.

        Parameters
        ---------
        X : pandas.DataFrame
            Dataset to predict.

        Returns
        -------
        numpy.ndarray
            Predicted values.
        """
        y_pred = self.estimator.predict(X)

        return y_pred

    def predict_proba(
        self,
        X,
        output="pmf",
        range=None,
    ) -> Union[list, pd.DataFrame]:
        """Probability estimates using the best model explored in Tornado.

        List containing instances of a fitted method collection or
        PMF (probability mass function) of a specific probability distribution
        is returned depending on `output`.

        Parameters
        ---------
        X : pandas.DataFrame
            Dataset to predict.

        output : str
            Output type. 'pmf' or 'func'. Default is 'pmf'.

        range : list
            List containing the lower and upper bounds of the probability
            distribution. If None, the lower bound is the minimum of the 0.05
            quantile of all samples, and the upper bound is the maximum of the
            0.95 quantile of all samples. Default is None.
                For example: [100, 200]

        Returns
        -------
        list of scipy.stats._distn_infrastructure.rv_frozen or pandas.DataFrame
            list of scipy.stats._distn_infrastructure.rv_frozen
                List of instance. Each instance is fitted method collection.
                For more information,
                https://docs.scipy.org/doc/scipy/reference/stats.html
            pandas.DataFrame
                Predicted probability distribution for each sample.
        """
        y_pred = self.estimator.predict(X.copy())

        mgr_attr = self.manager.get_attr()
        dist = mgr_attr["dist"]

        if dist == "poisson":
            pd_func = list()
            for mu in y_pred:
                pd_func.append(poisson(mu))

        elif dist == "nbinom":
            X["yhat_mean"] = y_pred
            X["yhat_mean_feature"] = X["yhat_mean"]
            c = self.nbinomc.predict(X)
            var = X["yhat_mean"] + c * X["yhat_mean"] * X["yhat_mean"]

            ps = np.minimum(np.where(var > 0, y_pred / var, 1 - 1e-8), 1 - 1e-8)
            ns = np.where(var > 0, y_pred * ps / (1 - ps), 1)

            pd_func = list()
            for n, p in np.stack((ns, ps)).T:
                pd_func.append(nbinom(n, p))

        if output == "func":
            return pd_func

        elif output == "pmf":
            min_xs = list()
            max_xs = list()
            if range is None:
                for dist in pd_func:
                    min_xs.append(dist.ppf(0.05))
                    max_xs.append(dist.ppf(0.95))
                min_x = min(min_xs)
                max_x = max(max_xs)
            else:
                min_x = range[0]
                max_x = range[1]
            x = np.linspace(start=min_x, stop=max_x, num=100).astype(int)
            pmfs = list()
            for dist in pd_func:
                pmfs.append(dist.pmf(x))

            return pd.DataFrame(pmfs, columns=x)


class ForwardSelectionModel(TornadoBase):
    """Class that controls learning using Forward Selection method in Tornado.

    Learning in this class is divided into two steps. The first step is to
    perform a single regression analysis and select valid features from the
    coefficient of determination (COD) and F-value. Then, the second step is
    to fit the model based on the selected features, adding one interaction
    term at each iteration and updating the model if it is better than the
    model from the previous iteration. This process is repeated to find the
    best model. This class inherits from :class:`TornadoBase`.

    Attributes
    ----------
    manager : One of the manager classes
        Class with TornadoManager as base class.

    estimator : sklearn.pipeline.Pipeline
        Pipeline with steps including binning and CB.

    nbinomc : sklearn.pipeline.Pipeline
        Estimator for parameter c of the negative binomial distribution.
    """

    def __init__(self, TornadoManager):
        super().__init__(TornadoManager)
        self.estimator = None
        self.nbinomc = None

    def fit(
        self,
        target,
        train,
        validation,
        save_dir="./models",
        criterion="COD",
        verbose=True,
    ) -> None:
        """Control the sequence of model learning using Forward Selection.

        Prepare instances and datasets for training, then train the model.
        Fitting in this class includes two recursive training steps by Tornado
        : the first step is a single regression analysis and the second step
        is a multiple regression analysis.

        Parameters
        ----------
        target : str
            Name of the target valiable.

        train : pandas.DataFrame
            Train set.

        validation : pandas.DataFrame
            Validation set.

        save_dir : str
            Path of the directory where the model will be stored. Default is
            "./models". Under this, a directory containing the model number
            (number of iterations) is created, in which information about the
            model is stored.

        criterion : str
            Best Model Evaluation Policy. 'COD' or 'PINBALL'. Default is 'COD'.

        verbose : bool
            Whether to display standard output or not. Default is True.
        """
        target = target.lower()
        evaluator = Evaluator()
        mgr_attr = self.manager.get_attr()
        training_round = [mgr_attr["first_round"], mgr_attr["second_round"]]
        logger = ForwardLogger(criterion, save_dir, training_round)

        # 1. rucursive single variable regression
        # This step is to find valid features
        self.manager.init(train, target)
        stage = "\n=== [{0}] {1} ===\n"
        status = "START"
        _logger.info(stage.format(status, mgr_attr["first_round"]))

        logger_attr = self.tornado(
            target,
            validation,
            logger,
            evaluator,
            verbose,
        )
        valid_features = list()
        for feature, eval_result in logger_attr["CODs"].items():
            if eval_result["F"] > 2:
                valid_features.append(feature)

        next_setting = {
            "experiment": 0,
            "mode": mgr_attr["second_round"],
            "target_features": valid_features,
            "X": mgr_attr["init_model_attr"]["X"].copy(),
        }
        self.manager.set_attr(next_setting)
        logger.reset_count()
        evaluator.clear()

        status = "END"
        _logger.info(stage.format(status, mgr_attr["first_round"]))

        # 2. multiple variable regression with forward selection
        status = "START"
        _logger.info(stage.format(status, mgr_attr["second_round"]))

        logger_attr = self.tornado(
            target,
            validation,
            logger,
            evaluator,
            verbose,
        )

        best_model_path = os.path.join(
            logger_attr["model_dir"],
            f'model_{logger_attr["log_data"]["iter"]}.pkl',
        )
        with open(best_model_path, "rb") as f:
            self.estimator = pickle.load(f)

        status = "END"
        _logger.info(stage.format(status, mgr_attr["second_round"]))

        # c parameter estimator for proba prediction
        mgr_attr = self.manager.get_attr()
        dist = mgr_attr["dist"]
        if dist == "nbinom":
            estimator_params = self.estimator[-1].__dict__
            update_params = {
                "dist": "nbinomc",
                "feature_properties": estimator_params["feature_properties"],
                "features": estimator_params["feature_groups"],
                "observers": estimator_params["observers"],
                "smoothers": estimator_params["smoother_choice"].explicit_smoothers,
            }
            self.manager.set_attr(update_params)

            self.nbinomc = self.manager.build()

            X_train = copy.deepcopy(self.manager.X)
            y_train = copy.deepcopy(self.manager.y)
            X_train["yhat_mean"] = self.estimator.predict(X_train.copy())
            X_train["yhat_mean_feature"] = X_train["yhat_mean"]
            self.nbinomc.fit(X_train.copy(), np.float64(y_train))

            self.manager.set_attr({"dist": "nbinom"})

    def tornado(self, target, validation, logger, evaluator, verbose) -> Pipeline:
        """Recursive learning in Tornado.

        Recursive learning is performed by changing the explanatory variables
        in the learning with Tornado's Forward Selection method. The process
        includes building, fitting, validating, and updating the model at each
        iteration. The process is repeated to search for the best model.

        Parameters
        ----------
        target : str
            Name of the target valiable.

        validation : pandas.DataFrame
            Validation set.

        logger : logger.Logger
            Instances to manage learning logs.

        evaluator : evaluator.Evaluator
            Instance to evaluate models.

        verbose : bool
            Whether to display standard output or not.

        Returns
        -------
        sklearn.pipeline.Pipeline
            Tornado estimator. Pipeline with steps including binning and CB.
        """
        while self.manager.manage():
            estimator = self.manager.build()

            # train
            X = copy.deepcopy(self.manager.X)
            y = copy.deepcopy(self.manager.y)
            _ = estimator.fit(X, y)

            # validation
            y_valid = np.asarray(validation[target])
            X_valid = validation.drop(target, axis=1)
            y_pred = estimator.predict(X_valid)
            evaluator.eval(y_valid, y_pred, estimator, verbose)

            # log
            eval_history = evaluator.eval(y_valid, y_pred, estimator, verbose=False)
            mgr_attr = self.manager.get_attr()
            logger.log(estimator, eval_history, mgr_attr)
            self.manager.clear()

            # update param
            if mgr_attr["mode"] == mgr_attr["second_round"]:
                keys = ["features", "feature_properties"]
                update_params = {k: logger.get_attr()["log_data"][k] for k in keys}
                self.manager.set_attr(update_params)

        return logger.get_attr()

    def predict(self, X) -> np.ndarray:
        """Predict using the best model explored in Tornado.

        Parameters
        ---------
        X : pandas.DataFrame
            Dataset to predict.

        Returns
        -------
        numpy.ndarray
            Predicted values.
        """
        y_pred = self.estimator.predict(X)

        return y_pred

    def predict_proba(
        self,
        X,
        output="pmf",
        range=None,
    ) -> Union[list, pd.DataFrame]:
        """Probability estimates using the best model explored in Tornado.

        List containing instances of a fitted method collection or
        PMF (probability mass function) of a specific probability distribution
        is returned depending on `output`.

        Parameters
        ---------
        X : pandas.DataFrame
            Dataset to predict.

        output : str
            Output type. 'pmf' or 'func'. Default is 'pmf'.

        range : list
            List containing the lower and upper bounds of the probability
            distribution. If None, the lower bound is the minimum of the 0.05
            quantile of all samples, and the upper bound is the maximum of the
            0.95 quantile of all samples. Default is None.
                For example: [100, 200]

        Returns
        -------
        list of scipy.stats._distn_infrastructure.rv_frozen or pandas.DataFrame
            list of scipy.stats._distn_infrastructure.rv_frozen
                List of instance. Each instance is fitted method collection.
                For more information,
                https://docs.scipy.org/doc/scipy/reference/stats.html
            pandas.DataFrame
                Predicted probability distribution for each sample.
        """
        y_pred = self.estimator.predict(X.copy())

        mgr_attr = self.manager.get_attr()
        dist = mgr_attr["dist"]

        if dist == "poisson":
            pd_func = list()
            for mu in y_pred:
                pd_func.append(poisson(mu))

        elif dist == "nbinom":
            X["yhat_mean"] = y_pred
            X["yhat_mean_feature"] = X["yhat_mean"]
            c = self.nbinomc.predict(X)
            var = X["yhat_mean"] + c * X["yhat_mean"] * X["yhat_mean"]

            ps = np.minimum(np.where(var > 0, y_pred / var, 1 - 1e-8), 1 - 1e-8)
            ns = np.where(var > 0, y_pred * ps / (1 - ps), 1)

            pd_func = list()
            for n, p in np.stack((ns, ps)).T:
                pd_func.append(nbinom(n, p))

        if output == "func":
            return pd_func

        elif output == "pmf":
            min_xs = list()
            max_xs = list()
            if range is None:
                for dist in pd_func:
                    min_xs.append(dist.ppf(0.05))
                    max_xs.append(dist.ppf(0.95))
                min_x = min(min_xs)
                max_x = max(max_xs)
            else:
                min_x = range[0]
                max_x = range[1]
            x = np.linspace(start=min_x, stop=max_x, num=100).astype(int)
            pmfs = list()
            for dist in pd_func:
                pmfs.append(dist.pmf(x))

            return pd.DataFrame(pmfs, columns=x)


class PriorPredInteractionSearchModel(TornadoBase):
    """Class that controls learning using Forward Selection method in Tornado.

    Learning for this class is divided into several steps.
    The first step is a multiple regression analysis with a single set of features
    (without interaction terms). Then, in the second step, a multiple regression
    analysis is performed with only two variables: the predictions of the multiple
    regression analysis and the interaction term, one in each iteration,
    to determine whether the interaction term is valid or not, based on whether
    the model improves or not.
    By repeating this process, the interaction terms are selected for
    model learning. Finally, the best model is generated by the better combination of
    interaction terms with a single features and the
    selected interaction terms. This class inherits from :class:`TornadoBase`.

    Attributes
    ----------
    manager : One of the manager classes
        Class with TornadoManager as base class.

    estimator : sklearn.pipeline.Pipeline
        Pipeline with steps including binning and CB.

    nbinomc : sklearn.pipeline.Pipeline
        Estimator for parameter c of the negative binomial distribution.
    """

    def __init__(self, TornadoManager):
        super().__init__(TornadoManager)
        self.estimator = None
        self.nbinomc = None

    def fit(
        self,
        target,
        train,
        validation,
        save_dir="./models",
        criterion="COD",
        verbose=True,
    ) -> None:
        """Control the sequence of model learning using Forward Selection.

        Prepare instances and datasets for training, then train the model.
        Fitting in this class contains several steps: the first is a
        single-round multiple regression analysis and the second is a
        recursive multiple regression analysis. And finally, the best model is
        generated by fitting with a three quartile regressor using the
        features selected in this process.

        Parameters
        ----------
        target : str
            Name of the target valiable.

        train : pandas.DataFrame
            Train set.

        validation : pandas.DataFrame
            Validation set.

        save_dir : str
            Path of the directory where the model will be stored. Default is
            "./models". Under this, a directory containing the model number
            (number of iterations) is created, in which information about the
            model is stored.

        criterion : str
            Best Model Evaluation Policy. 'COD' or 'PINBALL'. Default is
            'PINBALL'.

        verbose : bool
            Whether to display standard output or not. Default is True.
        """
        target = target.lower()
        evaluator = Evaluator()
        mgr_attr = self.manager.get_attr()
        training_round = [mgr_attr["first_round"], mgr_attr["second_round"]]
        logger = PriorPredForwardLogger(criterion, save_dir, training_round)

        # prior prediction for quick interaction search
        self.manager.init(train, target)
        stage = "\n=== [{0}] {1} ===\n"
        status = "START"
        _logger.info(stage.format(status, mgr_attr["first_round"]))

        base_model = self.tornado(
            target,
            validation,
            logger,
            evaluator,
            verbose,
        )

        X_train_init = mgr_attr["init_model_attr"]["X"].copy()  # use at final step
        X_train = X_train_init.copy()
        X_valid = validation.drop(target, axis=1)
        pred_train = base_model.predict(X_train.copy())
        pred_valid = base_model.predict(X_valid.copy())

        col = "prior_pred"
        X_train[col] = pred_train
        validation[col] = pred_valid
        init_model_attr = mgr_attr["init_model_attr"]
        init_model_attr["X"] = X_train.copy()
        model_params = {k: v for k, v in mgr_attr["model_params"].items()}
        model_params["prior_prediction_column"] = col
        next_setting = {
            "mode": mgr_attr["second_round"],
            "experiment": 0,
            "X": X_train.copy(),
            "model_params": model_params,
            "init_model_attr": init_model_attr,
        }
        self.manager.set_attr(next_setting)
        logger.reset_count()
        evaluator.clear()

        status = "END"
        _logger.info(stage.format(status, mgr_attr["first_round"]))

        # quick interaction search with prior pred
        status = "START"
        _logger.info(stage.format(status, mgr_attr["second_round"]))

        _ = self.tornado(
            target,
            validation,
            logger,
            evaluator,
            verbose,
        )
        _logger.info(f"\nDetect {len(logger.valid_interactions)} interactions\n")
        status = "END"
        _logger.info(stage.format(status, mgr_attr["second_round"]))

        # model setting for estimation
        base = [x for x in init_model_attr["feature_properties"].keys()]
        interaction = logger.get_attr()["valid_interactions"]
        features = base + interaction

        # best model fitting
        init_model_attr["X"] = X_train_init.copy()
        y_train = copy.deepcopy(mgr_attr["init_model_attr"]["y"])
        model_params = {
            "feature_properties": init_model_attr["feature_properties"],
            "feature_groups": features,
        }
        setting = {"X": X_train_init, "model_params": model_params, "init_model_attr": init_model_attr}
        self.manager.set_attr(setting)
        self.estimator = self.manager.build()
        _ = self.estimator.fit(X_train, y_train)

        # c parameter estimator for proba prediction
        mgr_attr = self.manager.get_attr()
        dist = mgr_attr["dist"]
        if dist == "nbinom":
            estimator_params = self.estimator[-1].__dict__
            estimator_params["feature_properties"]
            update_params = {
                "dist": "nbinomc",
                "feature_properties": estimator_params["feature_properties"],
                "features": estimator_params["feature_groups"],
                "observers": estimator_params["observers"],
                "smoothers": estimator_params["smoother_choice"].explicit_smoothers,
            }
            self.manager.set_attr(update_params)

            self.nbinomc = self.manager.build()

            X_train = copy.deepcopy(self.manager.X)
            y_train = copy.deepcopy(self.manager.y)
            X_train["yhat_mean"] = self.estimator.predict(X_train.copy())
            X_train["yhat_mean_feature"] = X_train["yhat_mean"]
            self.nbinomc.fit(X_train.copy(), np.float64(y_train))

            self.manager.set_attr({"dist": "nbinom"})

    def tornado(self, target, validation, logger, evaluator, verbose) -> Pipeline:
        """Recursive learning in Tornado.

        Recursive learning is performed by changing the explanatory variables
        in learning with QPD in Tornado. The process includes building,
        fitting, validating, and updating the model at each iteration. The
        process is repeated to search for the best model.

        Parameters
        ----------
        target : str
            Name of the target valiable.

        validation : pandas.DataFrame
            Validation data.

        logger : logger.Logger
            Instances to manage learning logs.

        evaluator : evaluator.Evaluator
            Instance to evaluate models.

        verbose : bool
            Whether to display standard output or not.

        Returns
        -------
        sklearn.pipeline.Pipeline
            Tornado estimator. Pipeline with steps including binning and CB.
        """
        while self.manager.manage():
            estimator = self.manager.build()

            X = copy.deepcopy(self.manager.X)
            y = copy.deepcopy(self.manager.y)
            _ = estimator.fit(X, y)

            y_valid = np.asarray(validation[target])
            X_valid = validation.loc[:, X.columns]
            y_pred = estimator.predict(X_valid)

            eval_history = evaluator.eval(y_valid, y_pred, estimator, verbose=False)
            mgr_attr = self.manager.get_attr()

            # log
            logger.log(estimator, eval_history, mgr_attr, verbose=verbose)
            self.manager.clear()

        return estimator

    def predict(self, X, quantile="median") -> np.ndarray:
        """Predict using the best model explored in Tornado.

        Parameters
        ---------
        X : pandas.DataFrame
            Dataset to predict.

        Returns
        -------
        numpy.ndarray
            Predicted values.
        """
        y_pred = self.estimator.predict(X)

        return y_pred

    def predict_proba(
        self,
        X,
        output="pdf",
        range=None,
    ) -> Union[list, pd.DataFrame]:
        """Probability estimates using the best model explored in Tornado.

        List containing instances of a fitted method collection or
        PMF (probability mass function) of a specific probability distribution
        is returned depending on `output`.

        Parameters
        ---------
        X : pandas.DataFrame
            Dataset to predict.

        output : str
            Output type. 'pmf' or 'func'. Default is 'pmf'.

        range : list
            List containing the lower and upper bounds of the probability
            distribution. If None, the lower bound is the minimum of the 0.05
            quantile of all samples, and the upper bound is the maximum of the
            0.95 quantile of all samples. Default is None.
                For example: [100, 200]

        Returns
        -------
        list of scipy.stats._distn_infrastructure.rv_frozen or pandas.DataFrame
            list of scipy.stats._distn_infrastructure.rv_frozen
                List of instance. Each instance is fitted method collection.
                For more information,
                https://docs.scipy.org/doc/scipy/reference/stats.html
            pandas.DataFrame
                Predicted probability distribution for each sample.
        """
        y_pred = self.estimator.predict(X.copy())

        mgr_attr = self.manager.get_attr()
        dist = mgr_attr["dist"]

        if dist == "poisson":
            pd_func = list()
            for mu in y_pred:
                pd_func.append(poisson(mu))

        elif dist == "nbinom":
            X["yhat_mean"] = y_pred
            X["yhat_mean_feature"] = X["yhat_mean"]
            c = self.nbinomc.predict(X)
            var = X["yhat_mean"] + c * X["yhat_mean"] * X["yhat_mean"]

            ps = np.minimum(np.where(var > 0, y_pred / var, 1 - 1e-8), 1 - 1e-8)
            ns = np.where(var > 0, y_pred * ps / (1 - ps), 1)

            pd_func = list()
            for n, p in np.stack((ns, ps)).T:
                pd_func.append(nbinom(n, p))

        if output == "func":
            return pd_func

        elif output == "pmf":
            min_xs = list()
            max_xs = list()
            if range is None:
                for dist in pd_func:
                    min_xs.append(dist.ppf(0.05))
                    max_xs.append(dist.ppf(0.95))
                min_x = min(min_xs)
                max_x = max(max_xs)
            else:
                min_x = range[0]
                max_x = range[1]
            x = np.linspace(start=min_x, stop=max_x, num=100).astype(int)
            pmfs = list()
            for dist in pd_func:
                pmfs.append(dist.pmf(x))

            return pd.DataFrame(pmfs, columns=x)


class QPDInteractionSearchModel(TornadoBase):
    """Class that controls learning using quantile regression in Tornado.

    This class uses a model with quantile regression and
    quantile-parameterized distributions (QPD). Learning for this class is
    divided into several steps. The first step is a multiple regression
    analysis with a single set of features (without interaction terms). Then,
    in the second step, a multiple regression analysis is performed with only
    two variables: the predictions of the multiple regression analysis and the
    interaction term, one in each iteration, to determine whether the
    interaction term is valid or not, based on whether the model improves or
    not. By repeating this process, the interaction terms are selected for
    model learning. Finally, the best model is generated by fitting each of
    the three quartile regression models with a single features and the
    selected interaction terms. This class inherits from :class:`TornadoBase`.

    Attributes
    ----------
    manager : One of the manager classes
        Class with TornadoManager as base class.

    quantile : float
        Lower quantile QPD symmetric-percentile triplet (SPT). Defaoult is 0.1.

    bound : str
        Different modes defined by supported target range, options are ``S``
        (semi-bound) and ``B`` (bound). Default is "S".

    lower : float
        lower bound of supported range (only active for bound and semi-bound
        modes). Default is 0.0.

    upper : float
        upper bound of supported range (only active for bound mode). Default
        is 1.0.

    loss : float
        Loss of learning.

    est_qpd : QPD_RegressorChain
        Pipeline with steps including binning and CB.
    """

    def __init__(
        self,
        TornadoManager,
        quantile=0.1,
        bound="S",
        lower=0.0,
        upper=1.0,
    ):
        super().__init__(TornadoManager)
        self.quantile = [quantile, 0.5, 1 - quantile]
        self.bound = bound
        self.lower = lower
        self.upper = upper
        self.loss = 0.0
        self.est_qpd = None

        if quantile >= 0.5:
            raise ValueError("quantile must be quantile < 0.5")

    def fit(
        self,
        target,
        train,
        validation,
        save_dir="./models",
        criterion="PINBALL",
        verbose=True,
    ) -> None:
        """Control the sequence of model learning using QPD.

        Prepare instances and datasets for training, then train the model.
        Fitting in this class contains several steps: the first is a
        single-round multiple regression analysis and the second is a
        recursive multiple regression analysis. And finally, the best model is
        generated by fitting with a three quartile regressor using the
        features selected in this process.

        Parameters
        ----------
        target : str
            Name of the target valiable.

        train : pandas.DataFrame
            Train set.

        validation : pandas.DataFrame
            Validation set.

        save_dir : str
            Path of the directory where the model will be stored. Default is
            "./models". Under this, a directory containing the model number
            (number of iterations) is created, in which information about the
            model is stored.

        criterion : str
            Best Model Evaluation Policy. 'COD' or 'PINBALL'. Default is
            'PINBALL'.

        verbose : bool
            Whether to display standard output or not. Default is True.
        """
        target = target.lower()
        evaluator = QuantileEvaluator()
        mgr_attr = self.manager.get_attr()
        training_round = [mgr_attr["first_round"], mgr_attr["second_round"]]
        logger = PriorPredForwardLogger(criterion, save_dir, training_round)

        self.manager.init(train, target)

        stage = "\n=== [{0}] {1} ===\n"
        status = "START"
        _logger.info(stage.format(status, mgr_attr["first_round"]))

        # prior prediction for quick interaction search
        base_model = self.tornado(
            target,
            validation,
            logger,
            evaluator,
            verbose,
            args={"quantile": self.quantile[0]},
        )

        X_train = mgr_attr["init_model_attr"]["X"].copy()
        X_valid = validation.drop(target, axis=1)
        pred_train = base_model.predict(X_train.copy())
        pred_valid = base_model.predict(X_valid.copy())

        col = "prior_pred"
        X_train[col] = pred_train
        validation[col] = pred_valid
        init_model_attr = mgr_attr["init_model_attr"]
        init_model_attr["X"] = X_train.copy()
        model_params = {k: v for k, v in mgr_attr["model_params"].items()}
        model_params["prior_prediction_column"] = col
        next_setting = {
            "mode": mgr_attr["second_round"],
            "experiment": 0,
            "X": X_train.copy(),
            "model_params": model_params,
            "init_model_attr": init_model_attr,
        }
        self.manager.set_attr(next_setting)
        logger.reset_count()
        evaluator.clear()

        status = "END"
        _logger.info(stage.format(status, mgr_attr["first_round"]))

        # quick interaction search with prior pred
        status = "START"
        _logger.info(stage.format(status, mgr_attr["second_round"]))

        _ = self.tornado(
            target,
            validation,
            logger,
            evaluator,
            verbose,
            args={"quantile": self.quantile[0]},
        )

        _logger.info(f"\nDetect {len(logger.valid_interactions)} interactions\n")
        status = "END"
        _logger.info(stage.format(status, mgr_attr["second_round"]))

        # model setting for QPD estimation
        base = [x for x in init_model_attr["feature_properties"].keys()]
        interaction = logger.get_attr()["valid_interactions"]
        features = base + interaction

        # best quantile models for QPD
        status = "START"
        _logger.info(stage.format(status, "QPD estimator training"))

        X_train = mgr_attr["init_model_attr"]["X"]
        model_params = {
            "feature_properties": init_model_attr["feature_properties"],
            "feature_groups": features,
        }
        est_quantiles = list()
        for quantile in self.quantile:
            model_params["quantile"] = quantile
            setting = {"X": X_train, "model_params": model_params}
            self.manager.set_attr(setting)
            self.manager.drop_unused_features()
            est = self.manager.build()
            est_quantiles.append(est)

        _logger.info("Trained 3 quantile estimators\n")
        status = "END"
        _logger.info(stage.format(status, "QPD estimator training"))

        X = copy.deepcopy(mgr_attr["init_model_attr"]["X"])
        y = copy.deepcopy(mgr_attr["init_model_attr"]["y"])
        self.est_qpd = QPD_RegressorChain(
            est_lowq=est_quantiles[0],
            est_median=est_quantiles[1],
            est_highq=est_quantiles[2],
            bound=self.bound,
            l=self.lower,
            u=self.upper,
        )
        _ = self.est_qpd.fit(X, y)

    def tornado(self, target, validation, logger, evaluator, verbose, args) -> Pipeline:
        """Recursive learning in Tornado.

        Recursive learning is performed by changing the explanatory variables
        in learning with QPD in Tornado. The process includes building,
        fitting, validating, and updating the model at each iteration. The
        process is repeated to search for the best model.

        Parameters
        ----------
        target : str
            Name of the target valiable.

        validation : pandas.DataFrame
            Validation set.

        logger : logger.Logger
            Instances to manage learning logs.

        evaluator : evaluator.Evaluator
            Instance to evaluate models.

        verbose : bool
            Whether to display standard output or not.

        Returns
        -------
        sklearn.pipeline.Pipeline
            Tornado estimator. Pipeline with steps including binning and CB.
        """
        while self.manager.manage():
            estimator = self.manager.build()

            X = copy.deepcopy(self.manager.X)
            y = copy.deepcopy(self.manager.y)
            _ = estimator.fit(X, y)

            y_valid = np.asarray(validation[target])
            X_valid = validation.loc[:, X.columns]
            y_pred = estimator.predict(X_valid)

            eval_history = evaluator.eval(
                y_valid,
                y_pred,
                args["quantile"],
                estimator,
                verbose=False,
            )
            mgr_attr = self.manager.get_attr()

            # log
            logger.log(estimator, eval_history, mgr_attr, verbose=verbose)
            self.manager.clear()

        return estimator

    def predict(self, X, quantile="median") -> np.ndarray:
        """Predict using the best model explored in Tornado.

        Parameters
        ---------
        X : pandas.DataFrame
            Dataset to predict.

        quantile : str
            Quartile point to be predicted. "lower", "median" or "upper".

        Returns
        -------
        numpy.ndarray
            Predicted values.
        """
        quantiles = self.est_qpd.predict(X)

        if quantile == "low":
            y_pred = quantiles[0]
        elif quantile == "median":
            y_pred = quantiles[1]
        elif quantile == "high":
            y_pred = quantiles[2]
        else:
            ValueError("quantile need to 'low' or 'median' or 'high'")

        return y_pred

    def predict_proba(
        self,
        X,
        output="pdf",
        range=None,
    ) -> Union[J_QPD_S, J_QPD_B, pd.DataFrame]:
        """Probability estimates using the best model explored in Tornado.

        Instances of the fitted method collection for a particular probability
        distribution or PPF (Percent Point Function) are returned depending on
        `output`.

        Parameters
        ---------
        X : pandas.DataFrame
            Dataset to predict.

        output : str
            Output type. 'pdf' or 'func'. Default is 'pdf'.

        range : list
            List containing the lower and upper bounds of the probability
            distribution. If None, the lower bound is the minimum of the 0.05
            quantile of all samples, and the upper bound is the maximum of the
            0.95 quantile of all samples. Default is None.
                For example: [100, 200]

        Returns
        -------
        Union[J_QPD_S, J_QPD_B] or pandas.DataFrame
            Union[J_QPD_S, J_QPD_B]
                List contains instances of :class:`J_QPD_S` or :class:`J_QPD_B`.
                Each instance is fitted to each prediction and contains several
                methods of the probability distribution. Length of the list is
                n_sumples.
            pandas.DataFrame
                Predicted PPF for every 0.05 for each sample. Shape of
                dataframe is n_samples x 19.
        """
        _, _, _, qpd = self.est_qpd.predict(X)

        if output == "func":
            return qpd
        elif output == "pdf":
            from findiff import FinDiff

            if range is None:
                min_x = min(qpd.ppf(0.05))
                max_x = max(qpd.ppf(0.95))
            else:
                min_x = range[0]
                max_x = range[1]
            x = np.linspace(start=min_x, stop=max_x, num=int(1e6))
            dx = x[1] - x[0]
            derivative = FinDiff(1, dx, 1)
            cdf_value = qpd.cdf(x)
            individual_pdf = derivative(cdf_value.T)

            return pd.DataFrame(individual_pdf, columns=x)
