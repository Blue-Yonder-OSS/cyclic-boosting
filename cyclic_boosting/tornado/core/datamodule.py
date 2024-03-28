"""Preparation (handling preprocessing) of data for the Tornado module."""

import logging
import os
import pickle
from itertools import combinations
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .preprocess import Preprocess

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(fmt="%(message)s"))
handler.terminator = ""
_logger.addHandler(handler)


class TornadoDataModule:
    """
    TornadoDataModule is a class that handles data preprocessing.

    This class accepts the file path of a dataset and returns preprocessed
    training and validation data.

    By default, it automatically applies multiple preprocessing steps to each
    feature, creating new feature columns. Subsequently, it reduces features
    based on the values of VIF (Variance Inflation Factor), a measure of
    multicollinearity.

    The history of preprocessing is output as a pickle file. If the path of
    this output file is provided as an argument, the same preprocessing can be
    replicated.

    Parameters
    ----------‵
    src: str or pandas.DataFrame
        Given the path to the data, the dataset as pandas.DataFrame is generated by
        loading from that location. Accepts either a csv or xlsx file.
        or received pandas.DataFrame directly.

    save_dir: str
        The path where the generated dataset should be saved. Defaults to None.
        Accepts either a csv or xlsx file.

    auto_preprocess: bool
        Whether to execute automatic preprocessing. Defaults to True.
        Set to False if using a dataset that has already been preprocessed.

    log_path: str
        The path to the preprocessing history file. Defaults to None.
        If None, the same path as the dataset will be set. If there is a
        pickle file at this path, it will be loaded and the preprocessing
        according to it will be replicated. If no file is present, automatic
        preprocessing will be performed and the history will be saved to this
        path.

    params: dict
        A dictionary of special options for various feature engineering
        procedures that are performed as preprocessing. Defaults to {}. The
        contents of the dictionary should have feature engineering process
        names as keys and options as values.

    Notes
    -----
    The `params` dictionary provided to the fifth argument should have feature
    engineering process names as keys and options as values.
    These options should be provided as dictionaries with option names as keys
    and the values for each option as values.
    By default, "minmax", "binning", and "clipping", which are not important
    considering the algorithmic characteristics of cyclic boosting, are not
    performed. You can perform these preprocessing optionally by adding them
    to the parameter `params`.

    Example:
        params =
            {"binning": {},
            "clipping": {"q_l": 0.10, "q_u": 0.90},
            "label_encoding": {"unknown_value": 0}}

    For details on what options are available for each technique, refer to the
    `Preprocess` class in preprocess.py.

    Attributes
    ----------
    train: pandas.DataFrame
        The data used for training.

    valid: pandas.DataFrame
        The data used for validation.

    target: str
        The name of the target variable.

    is_time_series: bool
        Indicates whether the data is time series data or not.

    preprocessors: dict
        The history of preprocessing.

    features: dict
        Unremoved features

    """

    def __init__(self, src, save_dir=None, auto_preprocess=True, log_path=None, params={}) -> None:
        super().__init__()
        self.src = src
        self.save_dir = save_dir
        self.auto_preprocess = auto_preprocess
        self.log_path = log_path
        self.params = params
        self.train = None
        self.valid = None
        self.target = None
        self.is_time_series = False
        self.set_log()

    def set_log(self) -> None:
        """Check and apply preprocessing log settings.

        If the path to the pickle file containing preprocessing logs,
        `log_path`, is not provided, `self.log_path` is set to the same
        directory as the dataset, and `self.preprocessors` and `self.features`
        are initialized. If `log_path` is provided and the file exists, its
        contents are assigned to `self.preprocessors` and `self.features`. If
        the file does not exist, `self.preprocessors` and `self.features` are
        initialized.
        """
        if self.log_path:
            try:
                with open(self.log_path, "rb") as p:
                    log = pickle.load(p)
                    self.preprocessors = log["preprocessors"]
                    self.features = log["features"]
            except FileNotFoundError:
                self.preprocessors = {}
                self.features = []
        else:
            if isinstance(self.src, str):
                self.log_path = self.src[: self.src.rfind(".")] + ".pkl"
            else:
                _logger.info(f"preprocessing log is created at {os.getcwd()}\n")
                self.log_path = os.path.join(os.getcwd(), "preprocessing.pkl")
            self.preprocessors = {}
            self.features = []

    def corr_based_removal(self) -> None:
        """Remove unnecessary features based on correlation coefficients.

        This function removes features based on two criteria: having a low
        absolute correlation coefficient with the target variable (default
        threshold: 0.1) and a high inter-feature absolute correlation
        coefficient (default threshold: 0.9). Among highly correlated pairs,
        the feature with the lower correlation to the target is discarded.
        """
        dataset = pd.concat([self.train.copy(), self.valid.copy()])
        try:
            dataset = dataset.drop(columns=["date"])
            has_date = True
        except KeyError:
            has_date = False

        corr_rl = 0.1
        corr_ul = 0.9
        corr = dataset.corr()
        features = corr.index

        for feature in features:
            if abs(corr.loc[feature, self.target]) < corr_rl:
                features = features.drop(feature)

        droped_features = []
        for feature1, feature2 in combinations(features.drop(self.target), 2):
            if abs(corr.loc[feature1, feature2]) > corr_ul:
                if not set([feature1, feature2]) & set(droped_features):
                    feature = corr.loc[[feature1, feature2], self.target].abs().idxmin()
                    features = features.drop(feature)
                    droped_features.append(feature)
        dataset = dataset[features]

        if has_date:
            self.train = self.train.loc[:, dataset.columns.tolist() + ["date"]]
            self.valid = self.valid.loc[:, dataset.columns.tolist() + ["date"]]

    def vif_based_removal(self) -> None:
        """Remove unnecessary features based on VIF.

        This function calculates the Variance Inflation Factor (VIF) for all
        features and removes the feature with the highest VIF
        value. This process is repeated until the maximum VIF among the
        features drops below the threshold of 10, suggesting a reduction of
        multicollinearity in the dataset.
        """
        dataset = pd.concat([self.train.copy(), self.valid.copy()])
        try:
            dataset = dataset.drop(columns=["date"])
            has_date = True
        except KeyError:
            has_date = False
        dataset = dataset.drop(columns=[self.target])
        dataset = dataset.astype("float").dropna()

        c = 10
        vif_max = c
        while vif_max >= c:
            vif = pd.DataFrame()
            with np.errstate(divide="ignore"):
                vif["VIF"] = [variance_inflation_factor(dataset.values, i) for i in range(dataset.shape[1])]
            vif["features"] = dataset.columns
            vif_max_idx = vif["VIF"].idxmax()
            vif_max = vif["VIF"].max()
            if vif_max >= c:
                dataset.drop(columns=vif["features"][vif_max_idx], inplace=True)
                vif_max = vif["VIF"].drop(vif_max_idx).max()

        if has_date:
            self.train = self.train.loc[:, dataset.columns.tolist() + ["date", self.target]]
            self.valid = self.valid.loc[:, dataset.columns.tolist() + ["date", self.target]]
        else:
            self.train = self.train.loc[:, dataset.columns.tolist() + [self.target]]
            self.valid = self.valid.loc[:, dataset.columns.tolist() + [self.target]]

    def remove_features(self) -> None:
        """Remove unnecessary features.

        By default, features are removed based on Variance Inflation Factor.
        However, the method can be changed to one based on Correlation
        Coefficient by editing the function call section.
        """
        if not self.features:
            self.vif_based_removal()
            self.features = self.train.columns.tolist()
        else:
            self.train = self.train.loc[:, self.features]
            self.valid = self.valid.loc[:, self.features]

    def generate_trainset(
        self,
        target,
        test_size=0.2,
        seed=0,
        is_time_series=False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate a dataset for prediction.

        This function loads a dataset from a specified path, applies
        appropriate preprocessing steps, and allows the dropping of non-
        essential variables. The preprocessing log is output as a pickle
        file, which can be referenced later.

        Parameters
        ----------
        target : str
            The name of the target variable.

        test_size : float
            The proportion of the data to allocate as test data. Default is 0.2.

        seed : int
            The random seed used for splitting the data into training and
            validation sets. Default is 0.

        is_time_series : bool
            Whether the data is a time series dataset or not.

        Returns
        -------
        tuple
            (pandas.DataFrame, pandas.DataFrame)
            The training and validation datasets as a tuple of pandas
            DataFrames.
        """
        self.target = target
        self.is_time_series = is_time_series

        preprocess = Preprocess(self.params)
        dataset = preprocess.load_dataset(self.src)

        if self.auto_preprocess:
            _logger.info("[START] Auto feature engineering \n")

            n_features_original = len(dataset.columns) - 1

            preprocess.check_data(dataset, self.is_time_series)
            if self.is_time_series:
                self.train, self.valid = train_test_split(dataset, test_size=test_size, shuffle=False)
            else:
                self.train, self.valid = train_test_split(dataset, test_size=test_size, random_state=seed)

            self.train, self.valid = preprocess.apply(self.train, self.valid, self.target)

            n_features_preprocessed = len(self.train.columns) - 1
            self.remove_features()

            n_features_selected = len(self.features) - 1
            self.preprocessors = preprocess.get_preprocessors()
            with open(self.log_path, "wb") as p:
                log = {"preprocessors": self.preprocessors, "features": self.features}
                pickle.dump(log, p)

            _logger.info(
                f"  Original     -> {n_features_original} features\n"
                f"  Preprocessed -> {n_features_preprocessed} features\n"
                f"  Selected     -> {n_features_selected} features\n"
            )
            _logger.info("[END] Auto feature engineering \n\n")

        else:
            self.train, self.valid = train_test_split(dataset, test_size=test_size, random_state=seed)

            if self.is_time_series:
                self.preprocessors["todatetime"] = {}
                preprocess.set_preprocessors(self.preprocessors)
                self.train, self.valid = preprocess.apply(self.train, self.valid, self.target)

        return self.train, self.valid

    def generate_testset(
        self,
        X,
    ) -> pd.DataFrame:
        """Generate a dataset for prediction.

        This function apply some preprocessing if `generete_trainset` method run before.
        if it is not, the dataset rename columns to lower case is just applied.
        (`Tornado` needs dataset with lower case columns.)

        Parameters
        ----------
        X : pandas.DataFrame
            Raw testset

        Returns
        -------
        pandas.DataFrame
            The testset
        """
        preprocess = Preprocess(self.params)
        X = preprocess.tolowerstr(X)

        if self.preprocessors:
            preprocess.set_preprocessors(self.preprocessors)
            self.train, _ = preprocess.apply(X, X, self.target)
            self.remove_features()

        else:
            if self.is_time_series:
                self.preprocessors["todatetime"] = {}
                preprocess.set_preprocessors(self.preprocessors)
                self.train, _ = preprocess.apply(X, X, self.target)
        X = self.train
        if self.target in X.columns:
            X = X.drop(self.target, axis=1)

        return X