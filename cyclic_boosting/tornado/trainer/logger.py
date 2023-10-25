import logging

import os
import pickle
import numpy as np
import copy

from cyclic_boosting import flags
from cyclic_boosting.plots import plot_analysis


_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(fmt="%(message)s"))
handler.terminator = ''
_logger.addHandler(handler)


class Logger():
    def __init__(self, save_dir, policy):
        # super().__init__()
        self.iter = 0
        self.save_dir = save_dir
        self.model_dir = None
        self.policy = policy
        self.make_dir()
        self.CODs = {}
        self.sorted_CODs = {}
        self.best = {}

        # for vote
        self.counter = 0

    def make_dir(self) -> None:
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)

    def make_model_dir(self) -> None:
        self.model_dir = os.path.join(self.save_dir,
                                      f'model_{self.best["iter"]}')
        if not os.path.isdir(self.model_dir):
            os.mkdir(self.model_dir)

    def save_model(self, est, name):
        pickle.dump(est, open(name, 'wb'))

    def save_metrics(self, name):
        with open(name, 'w') as f:
            for k, v in self.best["metrics"].items():
                f.write(f"[{k}]: {v} \n")

    def save_setting(self, name):
        fp = self.best["feature_properties"]

        s = self.best["smoothers"]

        with open(name, 'w') as f:
            # feature property
            f.write("=== Feature property ===\n")
            for k, v in fp.items():
                f.write(f"[{k}]: {v} \n")
            f.write("\n")

            # feature
            f.write("=== Feature ===\n")
            f.write(f"{self.best['features']}\n")
            f.write("\n")

            # smoother
            f.write("=== Explicit Smoother ===\n")
            for k, v in s.items():
                f.write(f"[{k}]: {v} \n")
            f.write("\n")

            # interaction term
            f.write("=== Interaction term ===\n")
            for term in self.best["features"]:
                if type(term) is tuple:
                    f.write(f"{term}\n")

    def save_plot(self, est, name):
        plobs = [est[-1].observers[-1]]
        binner = est[-2]
        for i, p in enumerate(plobs):
            plot_analysis(
                plot_observer=p,
                file_obj=name,
                use_tightlayout=False,
                binners=[binner],
            )

    def save_best(self):
        with open(os.path.join(self.save_dir,
                               'temp.pkl'), 'rb') as f:
            est = pickle.load(f)
            self.make_model_dir()
            self.save_metrics(os.path.join(self.model_dir,
                                           f'metrics_{self.best["iter"]}.txt'))
            self.save_setting(os.path.join(self.model_dir,
                                           f'setting_{self.best["iter"]}.txt'))
            self.save_model(est,
                            os.path.join(self.model_dir,
                                         f'model_{self.best["iter"]}.pkl'))
            self.save_plot(est,
                           os.path.join(self.model_dir,
                                        f'plot_{self.best["iter"]}'))

    def output(self, evt, mng):
        _logger.info(f"  ---- The best model was updated in iter {self.iter} ----\n")
        _logger.info(f"    best_features{mng.features}\n    ")
        for keys in evt.result.keys():
            _logger.info(f"{keys}: {evt.result[keys][-1]}, ")
        _logger.info("\n")

    def update_best(self, est, evt, mng):
        best_feature_properties = {}
        for f, p in mng.feature_properties.items():
            best_feature_properties[f] = flags.flags_to_string(p)

        best_smoothers = {}
        for f, sm in mng.smoothers.items():
            best_smoothers[f] = sm.name

        best_metrics = {}
        for metrics, value in evt.result.items():
            best_metrics[metrics] = copy.copy(value[-1])
        self.best = {
            "iter": self.iter,
            "features": mng.features,
            "feature_properties": best_feature_properties,
            "smoothers": best_smoothers,
            "metrics": best_metrics,
            }

        self.save_model(est,
                        os.path.join(self.save_dir,
                                     'temp.pkl'))

        self.output(evt, mng)

    def is_best(self, evt) -> bool:
        if self.policy == "compute_COD":
            result = evt.result['COD']
            best = self.best["metrics"]["COD"]
            is_best = result[self.iter] > best
        else:
            best = {k: v for k, v in self.best["metrics"].items()
                    if k != "COD" and k != "F"}
            result = {k: v for k, v in evt.result.items()
                      if k != "COD" and k != "F"}
            if self.policy == "vote":
                cnt = 0
                for metrics, value in result.items():
                    before = best[metrics]
                    after = np.nanmean(value)
                    if abs(before) > abs(after):
                        cnt += 1
                is_best = self.counter < cnt
                if is_best:
                    self.counter = cnt
            elif self.policy == "vote_by_num":
                cnt = 0
                for metrics, value in result.items():
                    if value[self.iter] < best[metrics]:
                        cnt += 1
                is_best = cnt > len(evt.result.items().mapping) / 2

        return is_best

    def reset_count(self) -> None:
        self.iter = 0

    def log_single(self, est, evt, is_last_iter) -> None:
        self.CODs[est['CB'].feature_groups[0]] = {
            'COD': evt.result['COD'][self.iter],
            'F': evt.result['F'][self.iter]
            }
        if is_last_iter:
            self.sorted_CODs = sorted(self.CODs.items(),
                                      key=lambda x: x[1]['COD'], reverse=True)

    def log_multiple(self, est, evt, mng, is_first_iter, is_last_iter) -> None:
        if is_first_iter:
            is_best = True
        else:
            is_best = self.is_best(evt)
        if is_best:
            self.update_best(est, evt, mng)
        if not is_last_iter:
            if "VariableSelection" in str(type(mng)):
                next_features = copy.deepcopy(self.best["features"])
                next_features.append(mng.sorted_features[self.iter])
                mng.get_features(next_features)
            else:
                pass
        else:
            self.save_best()
            os.remove(os.path.join(self.save_dir,
                                   'temp.pkl'))
            _logger.info(
                "\n\n"
                "Now, you can make a forecasting analysis with the best model"
                "using the pickle file in the ./models directory!\n"
                "For instructions, please refer to the file tornado.ipynb in"
                "the examples/regression/tornado directory."
            )

    def log(self, est, evt, mng) -> None:
        _logger.info(f"\riter: {self.iter} / {mng.max_interaction-1}")
        is_first_iter = self.iter == 0
        is_last_iter = mng.max_interaction-1 <= self.iter
        if mng.type == "single":
            self.log_single(est, evt, is_last_iter)
        elif mng.type == "multiple":
            self.log_multiple(est, evt, mng, is_first_iter, is_last_iter)
        self.iter += 1
