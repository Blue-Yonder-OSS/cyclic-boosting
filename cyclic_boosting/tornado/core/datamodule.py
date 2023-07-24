# Add some comments
#
#
#

import abc
import six
import pandas as pd
from preprocess import dayofweek


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
        self.dataset = pd.read_csv(self.path_ds)

    def check_data(self) -> None:
        # datasetに対してどんな前処理を施す必要があるかを調べて返す
        col_names = self.dataset.columns.to_list()

        if "dayofweek" not in col_names:
            self.func.append(dayofweek)

    def preprocess(self) -> pd.DataFrame:
        # 特徴量エンジニアリングを実行
        dataset = self.dataset
        for f in self.func:
            dataset = self.f(dataset)

        return dataset

    def generate(self) -> pd.DataFrame:
        self.load_dataset()
        self.check_data()

        return self.preprocess()