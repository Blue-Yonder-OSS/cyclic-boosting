import six
import abc
import copy
import numpy as np
from sklearn.model_selection import train_test_split

from .evaluator import EvaluatorBase
from .logger import Logger


@six.add_metaclass(abc.ABCMeta)
class TrainerBase():
    def __init__(self, DataModule, TornadoModule):
        self.data_deliveler = DataModule
        self.manager = TornadoModule

    @abc.abstractmethod
    def run(self,
            target,
            test_size=0.2,
            seed=0,
            save_dir='./models',
            log_policy='vote',
            metrix=None,
            verbose=True
            ):
        pass


class Trainer(TrainerBase):
    def __init__(self, DataModule, TornadoModule):
        super().__init__(DataModule, TornadoModule)

    def run(self,
            target,
            test_size=0.2,
            seed=0,
            save_dir='./models',
            log_policy='vote_by_num',
            metrix=None,
            verbose=True):
        dataset = self.data_deliveler.generate()
        train, validation = train_test_split(
            dataset,
            test_size=test_size,
            random_state=seed)

        evaluator = EvaluatorBase()
        logger = Logger(save_dir, log_policy)

        self.manager.init(train, target)
        while self.manager.manage():
            estimater = self.manager.build()
            # train
            X = copy.deepcopy(self.manager.X)
            y = copy.deepcopy(self.manager.y)
            _ = estimater.fit(X, y)

            # validation
            y_valid = np.asarray(validation[target])
            X_valid = validation.drop(target, axis=1)
            yhat = estimater.predict(X_valid)
            evaluator.eval_all(y_valid, yhat, verbose)

            # log
            logger.log(estimater, evaluator, self.manager)


class SqueezeTrainer(TrainerBase):
    def __init__(self, DataModule, TornadoModule):
        super().__init__(DataModule, TornadoModule)

    def run(self,
            target,
            test_size=0.2,
            seed=0,
            save_dir='./models',
            log_policy='compute_COD',
            metrix=None,
            verbose=True
            ):
        dataset = self.data_deliveler.generate()
        train, validation = train_test_split(
            dataset,
            test_size=test_size,
            random_state=seed)

        evaluator = EvaluatorBase()
        logger = Logger(save_dir, log_policy)

        self.manager.init(train, target)
        while self.manager.manage():
            estimater = self.manager.build()
            # train
            X = copy.deepcopy(self.manager.X)
            y = copy.deepcopy(self.manager.y)
            _ = estimater.fit(X, y)

            # validation
            y_valid = np.asarray(validation[target])
            X_valid = validation.drop(target, axis=1)
            yhat = estimater.predict(X_valid)
            evaluator.eval_all(y_valid, yhat, estimater, verbose)

            # log
            logger.log(estimater, evaluator, self.manager)

            # self.manager.reset(train, target)

        if log_policy == "compute_COD":
            truncated_features = {}
            #ここのthresholdは自由に決めることができるが基本的に２が良いらしい
            threshold = 2

            base = []
            for feature, data in logger.sorted_CODs:
                if isinstance(feature, str):
                    base.append(feature)

                if data["F"] > threshold:
                    truncated_features[feature] = data

            print("TRUNCATED")
            # base = [x for x in truncated_features.keys() if isinstance(x, str)]
            interaction = [x for x in truncated_features.keys() if isinstance(x, tuple)]

            # 寄与していると判定された交互作用項(2変数)の選別
            truncated_features = base + interaction
            self.manager.set_to_multiple(truncated_features)
            print(truncated_features)

            logger.reset_count()
            evaluator.clear()
            #ここで二回書いているのは後で関数にしたい
            while self.manager.manage():
                estimater = self.manager.build()
                # train
                X = copy.deepcopy(self.manager.X)
                y = copy.deepcopy(self.manager.y)
                _ = estimater.fit(X, y)

                # validation
                y_valid = np.asarray(validation[target])
                X_valid = validation.drop(target, axis=1)
                yhat = estimater.predict(X_valid)
                evaluator.eval_all(y_valid, yhat, estimater, verbose)
                # log
                logger.log(estimater, evaluator, self.manager)

                # self.manager.reset(train, target)
