# Add some comments
#
#
#

import abc
import six
import pandas as pd
from .preprocess import dayofweek, dayofyear, tolowerstr, todatetime, \
                        encode_category


# @six.add_metaclass(abc.ABCMeta)
class TornadoDataModule():
    # some comments

    def __init__(self, path, save_dir=None) -> None:
        super().__init__()
        self.path_ds = path
        self.save_dir = save_dir
        self.dataset = None
        self.func = []

    def load_dataset(self) -> None:
        # transfrom column name to lower
        self.dataset = tolowerstr(pd.read_csv(self.path_ds))

    def check_data(self) -> None:
        # datasetに対してどんな前処理を施す必要があるかを調べて返す
        col_names = self.dataset.columns.to_list()

        self.func.append(encode_category)

        if "date" in col_names:
            self.func.append(todatetime)

        if "dayofweek" not in col_names and "date" in col_names:
            self.func.append(dayofweek)

        if "dayofyear" not in col_names and "date" in col_names:
            self.func.append(dayofyear)

    def preprocess(self) -> pd.DataFrame:
        # 特徴量エンジニアリングを実行
        dataset = self.dataset
        for func in self.func:
            dataset = func(dataset)

        return dataset

    def generate(self) -> pd.DataFrame:
        self.load_dataset()
        self.check_data()

        return self.preprocess()