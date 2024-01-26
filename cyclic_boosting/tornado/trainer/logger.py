import logging

import abc
import six
import os
import pickle
import copy

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
    def __init__(self, save_dir, policy):
        self.iter = 0
        self.save_dir = save_dir
        self.policy = policy
        self.log_data = dict()
        self.model_dir = None
        self.make_dir()

    def get_params(self) -> dict:
        class_vars = dict()
        for attr_name, value in self.__dict__.items():
            if not callable(value) and not attr_name.startswith("__"):
                class_vars[attr_name] = value

        return class_vars

    def set_params(self, params: dict) -> None:
        for attr_name, value in params.items():
            self.__dict__[attr_name] = value

    def make_dir(self) -> None:
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)

    def make_model_dir(self) -> None:
        dir_name = f"model_{self.log_data['iter']}"
        self.model_dir = os.path.join(self.save_dir, dir_name)
        if not os.path.isdir(self.model_dir):
            os.mkdir(self.model_dir)

    def save_model(self, est, name):
        pickle.dump(est, open(name, "wb"))

    def save_metrics(self, file_name):
        with open(file_name, "w") as f:
            for name, value in self.log_data["metrics"].items():
                f.write(f"[{name}]: {value} \n")

    def save_setting(self, file_name):
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
            for feature in self.log_data['features']:
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

    def reset_count(self) -> None:
        self.iter = 0

    @abc.abstractmethod
    def log(self, est, eval_result, mng_attr, verbose=True) -> None:
        pass


class Logger(LoggerBase):
    def __init__(self, save_dir, policy, round_names=["first", "second"]):
        super().__init__(save_dir, policy)
        self.iter = 0
        self.save_dir = save_dir
        self.policy = policy
        self.first_round = round_names[0]
        self.second_round = round_names[1]
        self.log_data = dict()
        self.CODs = dict()
        self.model_dir = None
        self.make_dir()

    def output(self, eval_result, _):
        log = f"\n  ---- The best model was updated in iter {self.iter} ----\n"
        _logger.info(log)

        # log = f"    best_features{mng_attr.features}\n    "
        # _logger.info(log)

        for metrics in eval_result.keys():
            _logger.info(f"{metrics}: {eval_result[metrics]}, ")
        _logger.info("\n")

    def hold(self, est, eval_result, mng_attr, verbose=True, save=False):
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
        result = eval_result[self.policy]
        bench_mark = self.bench_mark["metrics"][self.policy]

        if self.policy == "COD":
            is_detect = result > bench_mark
        elif self.policy == "PINBALL":
            is_detect = result < bench_mark
        else:
            raise ValueError(f"{self.policy} doesn't not exist")

        return is_detect

    def save(self, est):
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
        file_name = f'model_{self.log_data["iter"]}'
        self.save_model(est, os.path.join(self.model_dir, file_name))

    def log_single(self, _, eval_result, mng_attr, last_iter) -> None:
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
        # check
        if first_iter:
            is_best = True
        else:
            is_best = self.validate(eval_result)

        # update
        if is_best:
            self.hold(est, eval_result, mng_attr, verbose=True, save=True)
            # for next validation
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


class BFForwardLogger(LoggerBase):
    def __init__(self, save_dir, policy, round_names=["first", "second"]):
        super().__init__(save_dir, policy)
        self.first_round = round_names[0]
        self.second_round = round_names[1]
        self.log_data = dict()
        self.bench_mark = dict()
        self.valid_interactions = list()

    def output(self, eval_result, mng_attr):
        policy = self.policy
        metrics = eval_result[self.policy]
        feature = mng_attr["features"][0]
        _logger.info(f"[DETECT] {policy}: {metrics}, {feature}\n")

    def hold(self, est, eval_result, mng_attr, verbose=True, save=True):
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
        result = eval_result[self.policy]
        bench_mark = self.bench_mark["metrics"][self.policy]

        if self.policy == "COD":
            is_detect = result > bench_mark
        elif self.policy == "PINBALL":
            is_detect = result < bench_mark
        else:
            raise ValueError(f"{self.policy} is not existed option")

        return is_detect

    def log_single(self, est, eval_result, mng_attr, last_iter) -> None:
        if last_iter:
            self.hold(est, eval_result, mng_attr, save=False, verbose=False)
            self.bench_mark = copy.deepcopy(self.log_data)

    def log_multiple(self,
                     est,
                     eval_result,
                     mng_attr,
                     first_iter,
                     _,
                     verbose=True,
                     ) -> None:
        # check
        if first_iter:
            is_best = True
        else:
            is_best = self.validate(eval_result)

        # update
        if is_best:
            self.hold(est, eval_result, mng_attr, verbose, save=False)
            interaction = self.log_data["features"][0]
            self.valid_interactions.append(interaction)

    def log(self, est, eval_result, mng_attr, verbose=True) -> None:
        iter_all = mng_attr["end"]
        self.iter = mng_attr["experiment"]
        _logger.info(f"\riter: {self.iter} / {iter_all}\n")

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
