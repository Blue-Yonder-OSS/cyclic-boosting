import numpy as np
import copy
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

