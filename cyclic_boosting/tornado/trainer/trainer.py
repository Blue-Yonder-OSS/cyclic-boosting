import numpy as np
from sklearn.model_selection import train_test_split

from .evaluator import EvaluatorBase
from .logger import Logger


class Trainer():
    def __init__(self, DataModule, TornadoModule):
        # super().__init__()
        self.data_deliveler = DataModule
        self.manager = TornadoModule

    def run(self,
            target,
            test_size=0.2,
            seed=0,
            save_dir='./models',
            log_policy='compute_COD',
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
            _ = estimater.fit(self.manager.X, self.manager.y)

            # validation
            y_valid = np.asarray(validation[target])
            X_valid = validation.drop(target, axis=1)
            yhat = estimater.predict(X_valid)
            evaluator.eval_all(y_valid, yhat, estimater, verbose)

            # log
            logger.log(estimater, evaluator, self.manager)

            self.manager.reset(train, target)
        if log_policy == "compute_COD":
            truncated_features = {}
            #ここのthresholdは自由に決めることができるが基本的に２が良いらしい
            threshold = 2
            for feature, data in logger.sorted_CODs:
                if data["F"] > threshold:
                    truncated_features[feature] = data
            self.manager.set_to_multiple(truncated_features)
            logger.reset_count()
            evaluator.clear()
            #ここで二回書いているのは後で関数にしたい
            while self.manager.manage():
                estimater = self.manager.build()
                # train
                _ = estimater.fit(self.manager.X, self.manager.y)

                # validation
                y_valid = np.asarray(validation[target])
                X_valid = validation.drop(target, axis=1)
                yhat = estimater.predict(X_valid)
                evaluator.eval_all(y_valid, yhat, estimater, verbose)
                # log
                logger.log(estimater, evaluator, self.manager)

                self.manager.reset(train, target)
