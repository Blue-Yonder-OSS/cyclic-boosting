import logging

import abc
import copy
import six
import numpy as np
from sklearn.model_selection import train_test_split

from .evaluator import Evaluator
from .logger import Logger


_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(fmt="%(message)s"))
handler.terminator = ""
_logger.addHandler(handler)


@six.add_metaclass(abc.ABCMeta)
class TrainerBase:
    def __init__(self, DataModule, TornadoModule):
        self.data_deliveler = DataModule
        self.manager = TornadoModule

    @abc.abstractmethod
    def run(self, target, test_size=0.2, seed=0, save_dir="./models", log_policy="vote", verbose=True):
        pass

    def train(self, target, valid_data, logger, evaluator, verbose):
        while self.manager.manage():
            estimater = self.manager.build()
            # train
            X = copy.deepcopy(self.manager.X)
            y = copy.deepcopy(self.manager.y)
            _ = estimater.fit(X, y)

            # validation
            y_valid = np.asarray(valid_data[target])
            X_valid = valid_data.drop(target, axis=1)
            yhat = estimater.predict(X_valid)
            evaluator.eval_all(y_valid, yhat, estimater, verbose)

            # log
            logger.log(estimater, evaluator, self.manager)
            # self.manager.reset(train, target)


class Trainer(TrainerBase):
    def __init__(self, DataModule, TornadoModule):
        super().__init__(DataModule, TornadoModule)

    def run(self, target, test_size=0.2, seed=0, save_dir="./models", log_policy="COD", verbose=True):
        # initialize
        evaluator = Evaluator()
        logger = Logger(save_dir, log_policy)
        dataset = self.data_deliveler.generate()
        train, validation = train_test_split(dataset, test_size=test_size, random_state=seed)
        self.manager.init(train, target)

        # train
        self.train(target, validation, logger, evaluator, verbose)


class SqueezeTrainer(TrainerBase):
    def __init__(self, DataModule, TornadoModule):
        super().__init__(DataModule, TornadoModule)
        self.policy = ["COD", "vote_by_num", "vote"]

    def run(self, target, test_size=0.2, seed=0, save_dir="./models", log_policy="COD", verbose=True):
        # initialize
        evaluator = Evaluator()
        logger = Logger(save_dir, log_policy)
        dataset = self.data_deliveler.generate()
        train, validation = train_test_split(dataset, test_size=test_size, random_state=seed)
        self.manager.init(train, target)

        # sigle variable regression analysis for variable selection
        self.train(target, validation, logger, evaluator, verbose)

        truncated_features = {}
        # ここのthresholdは自由に決めることができるが基本的に２が良いらしい
        threshold = 2

        # valid interaction term selection
        base_explanatory_variables = []
        for feature, data in logger.sorted_CODs:
            # original variables
            # NOTE: is it better including original variables for truncating?
            if isinstance(feature, str):
                base_explanatory_variables.append(feature)
            # trancate
            if data["F"] > threshold:
                truncated_features[feature] = data

        # set updated setting
        interaction = [x for x in truncated_features.keys() if isinstance(x, tuple)]
        explanatory_variables = base_explanatory_variables + interaction
        self.manager.set_to_multiple(explanatory_variables)

        _logger.info("\nTRUNCATED\n")
        _logger.info(f"{truncated_features}\n\n")

        # clearing for next training
        logger.reset_count()
        evaluator.clear()

        # multiple variable regression (main training)
        self.train(target, validation, logger, evaluator, verbose)
