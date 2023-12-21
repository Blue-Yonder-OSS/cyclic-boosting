# Add some comments
#
#
#
import logging

import abc
import six
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
# from .preprocess import dayofweek, dayofyear, tolowerstr, todatetime, \
#                         encode_category, lag, rolling, expanding, \
#                         check_cardinality, check_dtype
from .preprocess import Preprocess


_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(fmt="%(message)s"))
handler.terminator = ''
_logger.addHandler(handler)


# @six.add_metaclass(abc.ABCMeta)
class TornadoDataModule():
    # some comments

    def __init__(self, path, save_dir=None, auto_preprocess=False, log_path=None, params={}) -> None:
        super().__init__()
        self.path_ds = path
        self.save_dir = save_dir
        self.auto_preprocess = auto_preprocess
        self.log_path = log_path
        self.params = params
        # self.dataset = None
        self.train = None
        self.valid = None
        self.target = None
        self.is_ts = True
        self.func = []
        if self.log_path:
            try:
                with open(self.log_path, 'rb') as p:
                    log = pickle.load(p)
                    self.preprocessors = log['preprocessors']
                    self.features = log['features']
                    print(self.preprocessors.keys())
                    print(self.features)
            except FileNotFoundError:
                self.preprocessors = {}
                self.features = []
        else:
            self.log_path = self.path_ds[:self.path_ds.rfind('.')] + '.pickle'

    def corr_based_removal(self):
        dataset = pd.concat([self.train.copy(), self.valid.copy()])
        dataset = dataset.drop(columns=["date"])
        corr_rl = 0.1
        corr_ul = 0.9
        corr = self.dataset.corr()
        features = corr.index
        for feature1 in features.drop(self.target):
            if abs(corr.loc[feature1, self.target]) < corr_rl:
                features = features.drop(feature1)
            else:
                for feature2 in features.drop(self.target):
                    if (abs(corr.loc[feature1, feature2]) > corr_ul) & (feature1 != feature2):
                        if corr.loc[feature1, self.target] < corr.loc[feature2, self.target]:
                            features = features.drop(feature1)
        dataset = dataset[features]
        self.train = self.train.loc[:, dataset.columns.tolist() + ['date']]
        self.valid = self.valid.loc[:, dataset.columns.tolist() + ['date']]

    def vif_based_removal(self) -> None:
        dataset = pd.concat([self.train.copy(), self.valid.copy()])
        dataset = dataset.drop(columns=["date", self.target])
        dataset = dataset.astype('float').dropna()
        c = 10
        vif_max = c
        while vif_max >= c:
            vif = pd.DataFrame()
            with np.errstate(divide='ignore'): 
                vif["VIF Factor"] = [variance_inflation_factor(dataset.values, i)
                                     for i in range(dataset.shape[1])]
            vif["features"] = dataset.columns
            vif_max_idx = vif["VIF Factor"].idxmax()
            vif_max = vif["VIF Factor"].max()
            if vif_max >= c:
                dataset.drop(columns=vif["features"][vif_max_idx], inplace=True)
                vif_max = vif["VIF Factor"].drop(vif_max_idx).max()
        self.train = self.train.loc[:, dataset.columns.tolist() + ['date', self.target]]
        self.valid = self.valid.loc[:, dataset.columns.tolist() + ['date', self.target]]

    def remove_features(self) -> None:
        if not self.features:
            self.vif_based_removal()
            self.features = self.train.columns.tolist()
        else:
            self.train = self.train.loc[:, self.features]
            self.valid = self.valid.loc[:, self.features]

    def generate(self, target, is_ts, test_size, seed) -> pd.DataFrame:
        self.target = target
        self.is_ts = is_ts
        # self.check_data(dataset)
        preprocess = Preprocess(self.params)
        dataset = preprocess.load_dataset(self.path_ds)
        n_features_original = len(dataset.columns) - 1
        _logger.info(f"\rn_features: {n_features_original} ->")
        if self.preprocessors:
            preprocess.set_preprocessors(self.preprocessors)
        else:
            preprocess.check_data(dataset, self.is_ts)
        self.train, self.valid = train_test_split(
            dataset,
            test_size=test_size,
            random_state=seed)
        self.train, self.valid = preprocess.apply(self.train, self.valid, self.target)
        n_features_preprocessed = len(self.train.columns) - 1
        _logger.info(f"\rn_features: {n_features_original} -> "
                     f"{n_features_preprocessed} ->")
        self.remove_features()
        n_features_selected = len(self.features) - 1
        _logger.info(f"\rn_features: {n_features_original} -> "
                     f"{n_features_preprocessed} -> {n_features_selected}\n")
        _logger.info(f"{self.features}\n")
        self.preprocessors = preprocess.get_preprocessors()
        with open(self.log_path, 'wb') as p:
            log = {'preprocessors': self.preprocessors,
                   'features': self.features}
            pickle.dump(log, p)

        return self.train, self.valid
