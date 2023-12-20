# Add some comments
#
#
#
import logging

import abc
import six
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
# from .preprocess import dayofweek, dayofyear, tolowerstr, todatetime, \
#                         encode_category, lag, rolling, expanding, \
#                         check_cardinality, check_dtype
from .preprocess import Preprocess


_logger = logging.getLogger(__name__)


# @six.add_metaclass(abc.ABCMeta)
class TornadoDataModule():
    # some comments

    def __init__(self, path, save_dir=None, auto_preprocess=False, preprocessors={}, params={}) -> None:
        super().__init__()
        self.path_ds = path
        self.save_dir = save_dir
        self.auto_preprocess = auto_preprocess
        self.preprocessors = preprocessors
        self.params = params
        # self.dataset = None
        self.train = None
        self.valid = None
        self.target = None
        self.is_ts = True
        self.func = []
        self.features = {}

    def get_preprocessors(self) -> dict:
        return self.preprocessors

    def set_preprocessors(self, preps) -> None:
        for prep, params in preps.items():
            self.preprocessors[prep] = params

    # def load_dataset(self) -> pd.DataFrame:
    #     # transfrom column name to lower
    #     if self.path_ds.endswith(".csv"):
    #         dataset = tolowerstr(pd.read_csv(self.path_ds))
    #     elif self.path_ds.endswith(".xlsx"):
    #         dataset = tolowerstr(pd.read_excel(self.path_ds))
    #     else:
    #         _logger.error("The file format is not supported.\n"
    #                       "Please use the csv or xlsx format.")
    #     return dataset

    # def check_data(self, dataset) -> None:
    #     # datasetに対してどんな前処理を施す必要があるかを調べて返す
    #     col_names = dataset.columns.to_list()

    #     self.preprocessors[None] = {x: {} for x in col_names}
    #     self.preprocessors[encode_category] = {}
    #     self.preprocessors[check_dtype] = {}
    #     self.preprocessors[check_cardinality] = {}

    #     if self.is_ts:
    #         if "date" in col_names:
    #             self.preprocessors[todatetime] = {}
    #             self.preprocessors[lag] = {}
    #             self.preprocessors[rolling] = {}
    #             self.preprocessors[expanding] = {}
    #         else:
    #             _logger.error("If this is a forecast of time-series data,\n"
    #                             " a 'date' column is required to identify\n"
    #                             " the datetime of the data.")

    #         if "dayofweek" not in col_names and "date" in col_names:
    #             self.preprocessors[dayofweek] = {}

    #         if "dayofyear" not in col_names and "date" in col_names:
    #             self.preprocessors[dayofyear] = {}

    # def preprocess(self) -> None:
    #     # 特徴量エンジニアリングを実行
    #     preprocessors = self.get_preprocessors()
    #     for prep, params in preprocessors.items():
    #         if prep in []:
    #             self.train, self.valid = prep(self.train, self.valid, self.target, any(params))

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

        # preprocessors = self.get_preprocessors()
        
        # remove_features = []
        # for prep, features in preprocessors.items():
        #     for feature in features:
        #         if prep in [None, todatetime, dayofweek, dayofyear]:
        #             col = feature
        #         else:
        #             col = f"{feature}_{prep.__name__}"
        #         if col not in self.train.columns:
        #             remove_features.append([prep, feature])
        # print("==== dropped features ====")
        # for prep, feature in remove_features:
        #     preprocessors[prep].pop(feature)
        #     print(f"{feature}_{prep}")
        # print("==========================")

        # self.set_preprocessors(preprocessors)

    def generate(self, target, is_ts, test_size, seed) -> pd.DataFrame:
        self.target = target
        self.is_ts = is_ts
        # self.check_data(dataset)
        preprocess = Preprocess(self.params)
        dataset = preprocess.load_dataset(self.path_ds)
        if self.preprocessors:
            preprocess.set_preprocessors(self.preprocessors)
        else:
            preprocess.check_data(dataset, self.is_ts)
        self.train, self.valid = train_test_split(
            dataset,
            test_size=test_size,
            random_state=seed)
        # self.preprocess()
        self.train, self.valid = preprocess.apply(self.train, self.valid, self.target)
        self.remove_features()
        print(self.features)

        return self.train, self.valid
