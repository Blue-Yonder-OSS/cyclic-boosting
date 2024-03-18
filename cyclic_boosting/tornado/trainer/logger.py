"""Control of Tornado's training logs.

Standard output and output of files containing the best model, its settings,
evaluation results, graph plots, etc.
"""

import abc
import copy
import logging
import os
import pickle

import six

from cyclic_boosting import flags
from cyclic_boosting.plots import plot_analysis

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(fmt="%(message)s"))
handler.terminator = ""
_logger.addHandler(handler)


@six.add_metaclass(abc.ABCMeta)
class LoggerBase:
    """Base class for logging the progress of learning with Tornado.

    This class consolidates common functionalities that are shared among
    :class:`Logger` and :class:`BFForwardLogger`. It is responsible for
    storing logging data related to the model, iterations, features, feature
    properties, smoothers, and evaluation results.

    Parameters
    ----------
    save_dir : str
        The path to the directory where log data is saved.

    criterion : str
        Model evaluation criterion. "COD" or "PINBALL".

    Attributes
    ----------
    iter : int
        Iteration Count

    log_data : dict
        The logged data that holds information about iteration count, features,
        feature properties, smoothers, and evaluation results.

    model_dir : str
        The path to the directory where models are saved.
    """

    def __init__(self, criterion, save_dir):
        self.iter = 0
        self.save_dir = save_dir
        self.criterion = criterion
        self.log_data = dict()
        self.model_dir = None
        self.make_dir()

    def get_attr(self) -> dict:
        """Get class attributes.

        Exclude those starting with "__".

        Returns
        -------
        dict
            Dictionary with class attribute names as keys and class attribute
            as values.
        """
        class_vars = dict()
        for attr_name, value in self.__dict__.items():
            if not callable(value) and not attr_name.startswith("__"):
                class_vars[attr_name] = value

        return class_vars

    def set_attr(self, params: dict) -> None:
        """Set class attributes.

        Parameters
        ----------
        params : dict
            Parameters to be set. Dictionary with attribute names as keys and
            attribute as values.
        """
        for attr_name, value in params.items():
            self.__dict__[attr_name] = value

    def make_dir(self) -> None:
        """Create a directory if it does not exist at the `save_dir`."""
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)

    def make_model_dir(self) -> None:
        """Create a directory for saving the model.

        If the directory does not exist in `save_dir`.
        """
        dir_name = f"model_{self.log_data['iter']}"
        self.model_dir = os.path.join(self.save_dir, dir_name)
        if not os.path.isdir(self.model_dir):
            os.mkdir(self.model_dir)

    def save_model(self, est, name):
        """Save the model.

        Parameters
        ----------
        est : sklearn.pipeline.Pipeline
            The model to be saved.

        name : str
            The path to save the model.
        """
        pickle.dump(est, open(name, "wb"))

    def save_metrics(self, file_name):
        """Save the evaluation results of the model.

        Parameters
        ----------
        name : str
            The path to save the evaluation results.
        """
        with open(file_name, "w") as f:
            for name, value in self.log_data["metrics"].items():
                f.write(f"[{name}]: {value} \n")

    def save_setting(self, file_name):
        """Save the settings of the model.

        The settings include Feature property, Feature, Explicit Smoother, and
        Interaction term.

        Parameters
        ----------
        name : str
            The path to save the settings.
        """
        fp = self.log_data["feature_properties"]
        s = self.log_data["smoothers"]

        with open(file_name, "w") as f:
            # feature property
            f.write("=== Feature property ===\n")
            for feature, prop in fp.items():
                f.write(f"[{feature}]: {flags.flags_to_string(prop)} \n")
            f.write("\n")

            # feature
            f.write("=== Feature ===\n")
            for feature in self.log_data["features"]:
                f.write(f"{feature}\n")
            f.write("\n")

            # smoother
            f.write("=== Explicit Smoother ===\n")
            for feature, smoother in s.items():
                f.write(f"[{feature}]: {smoother} \n")
            f.write("\n")

            # interaction term
            f.write("=== Interaction term ===\n")
            for term in self.log_data["features"]:
                if isinstance(term, tuple):
                    f.write(f"{term}\n")

    def save_plot(self, est, name):
        """Save the visualization plot of the model.

        Parameters
        ----------
        est : sklearn.pipeline.Pipeline
            The model.

        name : str
            The path to save the plot.
        """
        plobs = [est[-1].observers[-1]]
        binner = est[-2]
        for p in plobs:
            plot_analysis(
                plot_observer=p,
                file_obj=name,
                use_tightlayout=False,
                binners=[binner],
            )

    def save(self, est):
        """Save information about the model to files.

        Parameters
        ----------
        est : sklearn.pipeline.Pipeline
            The model to be held.
        """
        self.make_model_dir()
        # metrics
        file_name = f"metrics_{self.log_data['iter']}.txt"
        self.save_metrics(os.path.join(self.model_dir, file_name))

        # setting
        file_name = f"setting_{self.log_data['iter']}.txt"
        self.save_setting(os.path.join(self.model_dir, file_name))

        # plot
        file_name = f'plot_{self.log_data["iter"]}'
        self.save_plot(est, os.path.join(self.model_dir, file_name))

        # model
        file_name = f'model_{self.log_data["iter"]}.pkl'
        self.save_model(est, os.path.join(self.model_dir, file_name))

    def reset_count(self) -> None:
        """Reset the iteration count."""
        self.iter = 0

    @abc.abstractmethod
    def log(self, est, eval_result, mng_attr, verbose=True) -> None:
        """Abstract method for logging the training of Tornado."""
        pass


class ForwardLogger(LoggerBase):
    """Logs the training for Tornado.

    Logs the training for regular Tornado or ForwardTrainer. Inherits from the
    base class LoggerBase.

    Parameters
    ----------
    save_dir : str
        The path to the directory to save the logs.

    criterion : str
        The evaluation criterion for the model. Should be "COD" or "PINBALL".

    round_names : list
        List of names of training rounds, including two round names.

    Attributes
    ----------
    iter : int
        Number of iterations.

    log_data : dict
        Holds information regarding iterations, features, feature
        characteristics, smoother, and evaluation results.

    model_dir : str
        Path to the directory where the model is saved.

    first_round : str
        Name of the first training round.
        Control variable for multiple recursive learning

    second_round : str
        Name of the second training round.
        Control variable for multiple recursive learning

    CODs : dict
        Model evaluation results (Coefficient of Determination (COD) and
        F-value).
    """

    def __init__(self, criterion, save_dir, round_names=["first", "second"]):
        super().__init__(criterion, save_dir)
        self.iter = 0
        self.save_dir = save_dir
        self.criterion = criterion
        self.first_round = round_names[0]
        self.second_round = round_names[1]
        self.log_data = dict()
        self.CODs = dict()
        self.model_dir = None
        self.make_dir()

    def output(self, eval_result, _):
        """Control standard output.

        Parameters
        ----------
        eval_result : dict
            Evaluation results of the model.

        mng_attr : dict
            Attributes in one of the manager classes
            Any manager class that inherits from
            :class:`cyclic_boosting.tornado.core.module.TornadoModuleBase`.
        """
        log = f"\n  ---- The best model was updated in iter {self.iter} ----\n"
        _logger.info(log)

        # log = f"    best_features{mng_attr.features}\n    "
        # _logger.info(log)

        for metrics in eval_result.keys():
            _logger.info(f"{metrics}: {eval_result[metrics]}, ")
        _logger.info("\n")

    def hold(self, est, eval_result, mng_attr, verbose=True, save=False):
        """Hold information of the best model.

        Parameters
        ----------
        est : sklearn.pipeline.Pipeline
            The model to be held.

        eval_result : dict
            Evaluation results of the model.

        mng_attr : dict
            Attributes in one of the manager classes
            Any manager class that inherits from
            :class:`cyclic_boosting.tornado.core.module.TornadoModuleBase`.

        verbose : bool
            Whether to display standard output or not. Default is True.

        save : bool
            Whether to save the model or not. Default is False.
        """
        feature_properties = dict()
        for feature, prop in mng_attr["feature_properties"].items():
            feature_properties[feature] = prop

        smoothers = dict()
        for feature, smoother in mng_attr["smoothers"].items():
            smoothers[feature] = smoother.__class__.__name__

        metrics = dict()
        for name, value in eval_result.items():
            metrics[name] = value

        features = [x for x in mng_attr["features"]]
        self.log_data = {
            "iter": self.iter,
            "features": features,
            "feature_properties": feature_properties,
            "smoothers": smoothers,
            "metrics": metrics,
        }

        if save:
            self.save_model(est, os.path.join(self.save_dir, "temp.pkl"))

        if verbose:
            self.output(eval_result, mng_attr)

    def validate(self, eval_result) -> bool:
        """Check if the model is the best or not.

        The model is compared with the coefficient of determination or Pinball
        loss, depending on the `criterion` variable of the model evaluation
        criterion.

        Parameters
        ----------
        eval_result : dict
            Evaluation results of the model.

        Returns
        -------
        bool
            Whether the model is best or not.
        """
        result = eval_result[self.criterion]
        bench_mark = self.bench_mark["metrics"][self.criterion]

        if self.criterion == "COD":
            is_detect = result > bench_mark
        elif self.criterion == "PINBALL":
            is_detect = result < bench_mark
        else:
            raise ValueError(f"{self.criterion} doesn't not exist")

        return is_detect

    def log_single(self, _, eval_result, mng_attr, last_iter) -> None:
        """Log a single regression analysis.

        This function saves the coefficient of determination (COD) and the F
        value in this class.

        Parameters
        ----------
        est : sklearn.pipeline.Pipeline
            The model to be held.

        eval_result : dict
            Evaluation results of the model.

        mng_attr : dict
            Attributes in one of the manager classes
            Any manager class that inherits from
            :class:`cyclic_boosting.tornado.core.module.TornadoModuleBase`.

        last_iter : bool
            Whether it is the last iteration or not
        """
        feature = mng_attr["features"][0]
        self.CODs[feature] = {
            "COD": eval_result["COD"],
            "F": eval_result["F"],
        }
        if last_iter:
            self.hold(None, eval_result, mng_attr, verbose=False)
            self.bench_mark = copy.deepcopy(self.log_data)

    def log_multiple(
        self,
        est,
        eval_result,
        mng_attr,
        first_iter,
        last_iter,
        verbose=True,
    ) -> None:
        """Log a multiple regression analysis.

        This function stores the best model information.

        Parameters
        ----------
        est : sklearn.pipeline.Pipeline
            The model to be held.

        eval_result : dict
            Evaluation results of the model.

        mng_attr : dict
            Attributes in one of the manager classes
            Any manager class that inherits from
            :class:`cyclic_boosting.tornado.core.module.TornadoModuleBase`.

        first_iter : bool
            Whether it is the first iteration or not

        last_iter : bool
            Whether it is the last iteration or not

        verbose : bool
            Whether to display standard output or not. Default is True.
        """
        # check
        if first_iter:
            is_best = True
        else:
            is_best = self.validate(eval_result)

        # update
        if is_best:
            self.hold(est, eval_result, mng_attr, verbose=verbose, save=True)
            self.bench_mark = copy.deepcopy(self.log_data)

        if last_iter:
            with open(os.path.join(self.save_dir, "temp.pkl"), "rb") as f:
                _est = pickle.load(f)
            self.save(_est)
            os.remove(os.path.join(self.save_dir, "temp.pkl"))
            _logger.info(
                "\n\n"
                "Now, you can make a forecasting analysis with the best model\n"
                "    using the pickle file in the ./models directory!\n"
                "For instructions, please refer to the file tornado.ipynb in\n"
                "    the examples/regression/tornado directory."
            )

    def log(self, est, eval_result, mng_attr, verbose=True) -> None:
        """Control logging.

        Perform logging for single regression analysis and for multiple
        regression analysis depending on the round of learning.

        Parameters
        ----------
        est : sklearn.pipeline.Pipeline
            The model to be held.

        eval_result : dict
            Evaluation results of the model.

        mng_attr : dict
            Attributes in one of the manager classes
            Any manager class that inherits from
            :class:`cyclic_boosting.tornado.core.module.TornadoModuleBase`.

        verbose : bool
            Whether to display standard output or not. Default is True.
        """
        iter_all = mng_attr["end"]
        self.iter = mng_attr["experiment"]
        _logger.info(f"\riter: {self.iter} / {iter_all} ")

        is_first_iter = self.iter == 1
        is_last_iter = iter_all <= self.iter
        if mng_attr["mode"] == self.first_round:
            self.log_single(
                est,
                eval_result,
                mng_attr,
                is_last_iter,
            )
        elif mng_attr["mode"] == self.second_round:
            self.log_multiple(
                est,
                eval_result,
                mng_attr,
                is_first_iter,
                is_last_iter,
                verbose,
            )


class PriorPredForwardLogger(LoggerBase):
    """Logs the training for Tornado.

    Logs the training for QPDForwardTrainer and inherits from the base class
    LoggerBase.

    Parameters
    ----------
    save_dir : str
        The path to the directory to save the logs.

    criterion : str
        The evaluation criterion for the model. Should be "COD" or "PINBALL".

    round_names : list
        List of names of training rounds, including two round names.

    Attributes
    ----------
    first_round : str
        Name of the first training round.
        Control variable for multiple recursive learning

    second_round : str
        Name of the second training round.
        Control variable for multiple recursive learning

    log_data : dict
        Holds information regarding iterations, features, feature
        characteristics, smoother, and evaluation results.

    bench_mark : dict
        Evaluation results of the best model at that point.

    valid_interactions : list
        List of valid interaction terms.
    """

    def __init__(self, criterion, save_dir, round_names=["first", "second"]):
        super().__init__(criterion, save_dir)
        self.first_round = round_names[0]
        self.second_round = round_names[1]
        self.log_data = dict()
        self.bench_mark = dict()
        self.valid_interactions = list()

    def output(self, eval_result, mng_attr):
        """Control standard output.

        Parameters
        ----------
        eval_result : dict
            Evaluation results of the model.

        mng_attr : dict
            Attributes in one of the manager classes
            Any manager class that inherits from
            :class:`cyclic_boosting.tornado.core.module.TornadoModuleBase`.
        """
        criterion = self.criterion
        metrics = eval_result[self.criterion]
        feature = mng_attr["features"][0]
        _logger.info(f"[DETECT] {criterion}: {metrics}, {feature}\n")

    def hold(self, est, eval_result, mng_attr, verbose=True, save=True):
        """Hold information of the best model.

        Parameters
        ----------
        est : sklearn.pipeline.Pipeline
            The model to be held.

        eval_result : dict
            Evaluation results of the model.

        mng_attr : dict
            Attributes in one of the manager classes
            Any manager class that inherits from
            :class:`cyclic_boosting.tornado.core.module.TornadoModuleBase`.

        verbose : bool
            Whether to display standard output or not. Default is True.

        save : bool
            Whether to save the model or not. Default is False.
        """
        feature_properties = dict()
        for feature, prop in mng_attr["feature_properties"].items():
            feature_properties[feature] = prop

        smoothers = dict()
        for f, smoother in mng_attr["smoothers"].items():
            smoothers[f] = smoother.__class__.__name__

        metrics = dict()
        for name, value in eval_result.items():
            metrics[name] = value

        features = [x for x in mng_attr["features"]]
        self.log_data = {
            "iter": self.iter,
            "features": features,
            "feature_properties": feature_properties,
            "smoothers": smoothers,
            "metrics": metrics,
        }

        if save:
            self.save_model(est, os.path.join(self.save_dir, "temp.pkl"))

        if verbose:
            self.output(eval_result, mng_attr)

    def validate(self, eval_result) -> bool:
        """Check if the model is the best or not.

        The model is compared with the coefficient of determination or Pinball
        loss, depending on the `criterion` variable of the model evaluation
        criterion.

        Parameters
        ----------
        eval_result : dict
            Evaluation results of the model.

        Returns
        -------
        bool
            Whether the model is best or not.
        """
        result = eval_result[self.criterion]
        bench_mark = self.bench_mark["metrics"][self.criterion]

        if self.criterion == "COD":
            is_detect = result > bench_mark
        elif self.criterion == "PINBALL":
            is_detect = result < bench_mark
        else:
            raise ValueError(f"{self.criterion} is not existed option")

        return is_detect

    def log_single(self, est, eval_result, mng_attr, last_iter) -> None:
        """Log a single regression analysis.

        Stores information of the model of the last iteration.
        Learning phase of the model using features excluding interaction terms
        in the QPDForwardTrainer.

        Parameters
        ----------
        est : sklearn.pipeline.Pipeline
            The model to be held.

        eval_result : dict
            Evaluation results of the model.

        mng_attr : dict
            Attributes in one of the manager classes
            Any manager class that inherits from
            :class:`cyclic_boosting.tornado.core.module.TornadoModuleBase`.

        last_iter : bool
            Whether it is the last iteration or not
        """
        if last_iter:
            self.hold(est, eval_result, mng_attr, save=False, verbose=False)
            self.bench_mark = copy.deepcopy(self.log_data)

    def log_multiple(
        self,
        est,
        eval_result,
        mng_attr,
        first_iter,
        _,
        verbose=True,
    ) -> None:
        """Log a multiple regression analysis.

        This function stores the best model information.
        Searching phase for interaction terms in QPDForwardTrainer.

        Parameters
        ----------
        est : sklearn.pipeline.Pipeline
            The model to be held.

        eval_result : dict
            Evaluation results of the model.

        mng_attr : dict
            Attributes in one of the manager classes
            Any manager class that inherits from
            :class:`cyclic_boosting.tornado.core.module.TornadoModuleBase`.

        first_iter : bool
            Whether it is the first iteration or not

        last_iter : bool
            Whether it is the last iteration or not

        verbose : bool
            Whether to display standard output or not. Default is True.
        """
        # check
        if first_iter:
            is_best = True
        else:
            is_best = self.validate(eval_result)

        # update
        if is_best:
            self.hold(est, eval_result, mng_attr, verbose=verbose, save=False)
            interaction = self.log_data["features"][0]
            self.valid_interactions.append(interaction)

    def log(self, est, eval_result, mng_attr, verbose=True) -> None:
        """Control logging.

        Perform logging for single regression analysis and for multiple
        regression analysis depending on the round of learning.

        Parameters
        ----------
        est : sklearn.pipeline.Pipeline
            The model to be held.

        eval_result : dict
            Evaluation results of the model.

        mng_attr : dict
            Attributes in one of the manager classes
            Any manager class that inherits from
            :class:`cyclic_boosting.tornado.core.module.TornadoModuleBase`.

        verbose : bool
            Whether to display standard output or not. Default is True.
        """
        iter_all = mng_attr["end"]
        self.iter = mng_attr["experiment"]
        _logger.info(f"\riter: {self.iter} / {iter_all} ")

        is_first_iter = self.iter == 1
        is_last_iter = iter_all <= self.iter
        if mng_attr["mode"] == self.first_round:
            self.log_single(
                est,
                eval_result,
                mng_attr,
                is_last_iter,
            )
        elif mng_attr["mode"] == self.second_round:
            self.log_multiple(
                est,
                eval_result,
                mng_attr,
                is_first_iter,
                is_last_iter,
                verbose,
            )
