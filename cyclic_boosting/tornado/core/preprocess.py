"""
Data preprocessing and feature engineering techniques.

Processing necessary for data loading, checking, and other preprocessing
tasks, as well as various feature engineering methods.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import (
    KBinsDiscretizer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    QuantileTransformer,
    StandardScaler,
    TargetEncoder,
)
import warnings

_logger = logging.getLogger(__name__)


class Preprocess:
    """
    Preprocess is a class for performing data preprocessing.

    It contains a series of procedures and various feature engineering methods
    as functions that are necessary for handling preprocessing inside a
    datamodule.

    This class takes preprocessing history and options, and performs
    preprocessing accordingly, returning the preprocessed training and
    validation data. If no preprocessing history or options are given, it
    determines and applies preprocessing methods that can be applied to each
    feature of the dataset.

    Some methods generate new features while others transform original
    features. As a result, the number of features usually increases compared
    to before processing.

    By default, "minmax", "binning", and "clipping", which are not important
    considering the methodological characteristics of cyclic boosting, are not
    performed. You can perform these preprocessing optionally by adding them
    to the parameter `opt`.

    Parameters
    ----------
    opt: str
        A dictionary for setting special options for various feature
        engineering procedures that are performed as preprocessing. Defaults
        to {}. The dictionary should have feature engineering process names as
        keys and dictionaries specifying options as values.
        These option-specifying dictionaries should have option names as keys
        and the values to set for the options as values.
        (Refer to the 'params' argument and Notes of the TornadoDataModule
         class in datamodule)

    Notes
    -----
    For encoding categorical variables, label encoding is selected by default.
    It can be changed by editing the 'encode_category' part of the pickle file
    provided to the check_data function or the TornadoDataModule class in
    datamodule.

    Example:
        To use target encoding, edit the check_data function:
            self.preprocessors["encode_category"] = {"label_encoding": {}}
            ->
            self.preprocessors["encode_category"] = {"target_encoding": {}}

    Attributes
    ----------
    preprocessors: dict
        The history of preprocessing.

    train: pandas.DataFrame
        The data used for training.

    valid: pandas.DataFrame
        The data used for validation.

    train_raw: pandas.DataFrame
        The raw training data before preprocessing.

    valid_raw: pandas.DataFrame
        The raw validation data before preprocessing.

    """

    def __init__(self, opt) -> None:
        self.preprocessors = {}
        self.opt = opt

    def get_preprocessors(self) -> dict:
        """Get `self.preprocessors`, the history of preprocessing."""
        return self.preprocessors

    def set_preprocessors(self, preps) -> None:
        """Set `self.preprocessors`, the history of preprocessing.

        Parameters
        ----------
        preps : dict
            The history of preprocessing.
            Keys are names of preprocessing methods, and values are the
            variables used in those specific methods.
        """
        for prep, config in preps.items():
            self.preprocessors[prep] = config

    def get_opt(self, func_name) -> dict:
        """Get `self.opt`, the options for various preprocessing methods.

        Parameters
        ----------
        func_name : str
            Name of the function.
        """
        try:
            opt = self.opt[func_name]
        except KeyError:
            opt = {}

        return opt

    def load_dataset(self, src) -> pd.DataFrame:
        """Load a dataset from a csv or xlsx file.

        Column names in the loaded dataset will be renamed to lowercase.

        Parameters
        ----------
        src : str
            Dataset path of csv or xlsx, or pandas.DataFrame

        Returns
        -------
        pandas.DataFrame
            Dataset
        """
        if isinstance(src, str):
            if src.endswith(".csv"):
                df = pd.read_csv(src)
            elif src.endswith(".xlsx"):
                df = pd.read_excel(src)
            else:
                _logger.error("The file format is not supported.\n" "Please use the csv or xlsx format.")
        elif isinstance(src, pd.DataFrame):
            df = src
        else:
            ValueError("please set path of csv or xlsx, or pandas.DataFrame")
        dataset = self.tolowerstr(df)

        return dataset

    def check_data(self, dataset, is_time_series) -> None:
        """Set up the necessary preprocessing for the dataset.

        Parameters
        ----------
        dataset : pandas.DataFrame
            Dataset

        is_time_series : bool
            Whether the data is a time series dataset or not.
        """
        if self.preprocessors:
            return
        col_names = dataset.columns.to_list()

        self.preprocessors["encode_category"] = {"label_encoding": {}}
        self.preprocessors["check_dtype"] = {}
        self.preprocessors["check_cardinality"] = {}
        self.preprocessors["check_data_leakage"] = {}

        if "date" in col_names:
            self.preprocessors["todatetime"] = {}
            if "dayofweek" not in col_names:
                self.preprocessors["dayofweek"] = {}

            if "dayofyear" not in col_names:
                self.preprocessors["dayofyear"] = {}

            if is_time_series:
                self.preprocessors["lag"] = {}
                self.preprocessors["rolling"] = {}
                self.preprocessors["expanding"] = {}

        elif is_time_series:
            _logger.error(
                "If this is a forecast of time-series data,\n"
                " a 'date' column is required to identify\n"
                " the datetime of the data."
            )

        # self.preprocessors["standardization"] = {}
        # self.preprocessors["logarithmic"] = {}
        # self.preprocessors["rank"] = {}
        # self.preprocessors["rankgauss"] = {}

    def apply(self, train, valid, target) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply feature engineering to the dataset.

        Parameters
        ----------
        train: pandas.DataFrame
            Training data

        valid: pandas.DataFrame
            Validation data

        target: str
            Name of the target variable.

        Returns
        -------
        tuple
            (pandas.DataFrame, pandas.DataFrame)
            Training data and validation data after applying preprocessing

        Notes
        -----
        This function utilizes three pairs of training and validation data
        with distinct roles:

        - `self.train`, `self.valid`:
            Datasets that store features modified by preprocessing.

        - `self.train_raw`, `self.valid_raw`:
            The original datasets that are normally unmodified, except when
            the `todatetime` or `encode_category` functions are applied, which
            then alter the datasets.

        - `train`, `valid`:
            Temporary datasets for holding interim results between
            preprocessing steps.
        """
        self.train_raw = train.copy()
        self.valid_raw = valid.copy()
        self.train = train.copy()
        self.valid = valid.copy()

        for prep in self.opt.keys():
            if prep not in self.get_preprocessors().keys():
                self.set_preprocessors({prep: {}})
        preprocessors = self.get_preprocessors().copy()

        for prep, config in preprocessors.items():
            train, valid = eval(f"self.{prep}")(self.train_raw.copy(), self.valid_raw.copy(), target, config)
            train[self.train.columns] = self.train
            valid[self.valid.columns] = self.valid
            self.train = train.copy()
            self.valid = valid.copy()

        return self.train, self.valid

    def todatetime(self, train, valid, *args) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Convert the data type of the "date" column to datetime.

        Parameters
        ----------
        train: pandas.DataFrame
            Training data

        valid: pandas.DataFrame
            Validation data

        Returns
        -------
        tuple
            (pandas.DataFrame, pandas.DataFrame)
            Training and validation data with "date" column of type datetime.
        """
        train["date"] = pd.to_datetime(train["date"])
        valid["date"] = pd.to_datetime(valid["date"])

        self.set_preprocessors({"todatetime": {}})
        self.train.drop("date", axis=1, inplace=True)
        self.valid.drop("date", axis=1, inplace=True)
        self.train_raw = train
        self.valid_raw = valid

        return train, valid

    def tolowerstr(self, dataset) -> pd.DataFrame:
        """Take a dataset and return it with lowercase column names.

        Parameters
        ----------
        dataset: pandas.DataFrame
            dataset

        Returns
        -------
        pandas.DataFrame
            dataset with column names in lowercase.
        """
        renames = {before: before.lower() for before in dataset.columns}
        dataset = dataset.rename(columns=renames)

        return dataset

    def dayofweek(self, train, valid, *args) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate day-of-week feature from date feature.

        Parameters
        ----------
        train: pandas.DataFrame
            Training data

        valid: pandas.DataFrame
            Validation data

        Returns
        -------
        tuple
            (pandas.DataFrame, pandas.DataFrame)
            Training and validation data with "dayofweek" column.
        """
        train["dayofweek"] = train["date"].dt.dayofweek
        valid["dayofweek"] = valid["date"].dt.dayofweek
        train["dayofweek"] = train["dayofweek"].astype("int64")
        valid["dayofweek"] = valid["dayofweek"].astype("int64")
        self.set_preprocessors({"dayofweek": {}})

        return train, valid

    def dayofyear(self, train, valid, *args) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate day-of-year feature from date feature.

        Parameters
        ----------
        train: pandas.DataFrame
            Training data

        valid: pandas.DataFrame
            Validation data

        Returns
        -------
        tuple
            (pandas.DataFrame, pandas.DataFrame)
            Training and validation data with "dayofyear" column.
        """
        train["dayofyear"] = train["date"].dt.dayofyear
        valid["dayofyear"] = valid["date"].dt.dayofyear
        train["dayofyear"] = train["dayofyear"].astype("int64")
        valid["dayofyear"] = valid["dayofyear"].astype("int64")
        self.set_preprocessors({"dayofyear": {}})

        return train, valid

    def create_time_based_average_data(self, dataset, target) -> pd.DataFrame:
        """Take a dataset and create and return time-based average data.

        Parameters
        ----------
        dataset: pandas.DataFrame
            dataset

        target: str
            Name of the target variable.

        Returns
        -------
        pandas.DataFrame
            Data averaged by time
        """
        time_based_average_data = dataset.groupby("date")[target].mean()
        time_based_average_data = pd.DataFrame(time_based_average_data).sort_index()

        return time_based_average_data

    def check_corr(self, time_based_average_data) -> Tuple[int, int]:
        """Return lag size with the largest autocorrelation coefficients.

        Take time-based average data, check the autocorrelation coefficient
        and partial autocorrelation coefficient, and return the lag sizes at
        which these coefficients are the largest.

        Parameters
        ----------
        time_based_average_data: pandas.DataFrame
            Data averaged by time

        Returns
        -------
        tuple
            (int, int)
            Lag size with maximum autocorrelation coefficient and lag size
            with maximum partial autocorrelation coefficient.
        """
        nlags = int(len(time_based_average_data) / 2 - 1)
        nlags = min(nlags, 500)
        autocorrelation = sm.tsa.stattools.acf(time_based_average_data, nlags=nlags)
        partial_autocorrelation = sm.tsa.stattools.pacf(time_based_average_data, method="ywm", nlags=nlags)
        idx_autocorrelation = np.argmax(autocorrelation[1:]) + 1
        idx_partial_autocorrelation = np.argmax(partial_autocorrelation[1:]) + 1

        return idx_autocorrelation, idx_partial_autocorrelation

    def lag(self, train, valid, target, config) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate lag feature from date feature.

        Parameters
        ----------
        train: pandas.DataFrame
            Training data

        valid: pandas.DataFrame
            Validation data

        target: str
            Name of the target variable.

        config: dict
            Configuration for the function (manually or as history).

        Returns
        -------
        tuple
            (pandas.DataFrame, pandas.DataFrame)
            Training and validation data with "lag" features.

        Notes
        -----
        By default, this function generates lag features at the lag size where
        the partial autocorrelation coefficient is maximized. However, users
        have the option to produce lag features at a specific lag size by
        setting the `lag_size` parameter in the lag function. This allows for
        fine-tuning of the lag features based on domain knowledge or specific
        analytical needs.

        You can specify the `lag_size` as follows.
            >>> from cyclic_boosting.tornado.core.preprocess import Preprocess
            >>> opt = {"lag": {"lag_size": 5}}
            >>> prep = Preprocess(opt)
            >>> lag_size = prep.get_opt("lag")["lag_size"]
            >>> lag_size
            5
        """

        def calc_lag(data, lag_size) -> pd.Series:
            if train.shape[0] < lag_size:
                raise RuntimeError(
                    f"The provided data shape is {data.shape}. More than {lag_size} records are required."
                )
            lags = time_based_average_data.shift(lag_size)
            return lags

        dataset = pd.concat([train, valid])
        time_based_average_data = self.create_time_based_average_data(dataset, target)
        if config:
            lag_size = self.get_preprocessors()["lag"]["lag_size"]
            lags = calc_lag(time_based_average_data, lag_size)
        else:
            opt = self.get_opt("lag")
            opt.setdefault("lag_size", self.check_corr(time_based_average_data)[1])
            lag_size = opt["lag_size"]
            _logger.info(f"lag_size = {lag_size}")
            self.set_preprocessors({"lag": {"lag_size": lag_size}})
            lags = calc_lag(time_based_average_data, lag_size)

        train["lag"] = train["date"].map(lags[target])
        valid["lag"] = valid["date"].map(lags[target])

        return train, valid

    def rolling(self, train, valid, target, config) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate rolling feature from date feature.

        Parameters
        ----------
        train: pandas.DataFrame
            Training data

        valid: pandas.DataFrame
            Validation data

        target: str
            Name of the target variable.

        config: dict
            Configuration for the function (manually or as history).

        Returns
        -------
        tuple
            (pandas.DataFrame, pandas.DataFrame)
            Training and validation data with "rolling" features.

        Notes
        -----
        By default, the rolling feature is created with a window that starts
        from a lag of one and extends to the lag size where the partial
        autocorrelation coefficient is maximized. However, the rolling
        function offers optional parameters `lag_size` and `best_lag`, which
        enable the creation of rolling features with a user-defined specific
        window size. This option provides the flexibility to customize the
        window size for rolling features based on particular requirements or
        preferences.

        You can specify the `lag_size` and `best_lag` as follows.
            >>> from cyclic_boosting.tornado.core.preprocess import Preprocess
            >>> opt = {"rolling": {"lag_size": 5, "best_lag": 10}}
            >>> prep = Preprocess(opt)
            >>> lag_size = prep.get_opt("rolling")["lag_size"]
            >>> lag_size
            5
            >>> best_lag = prep.get_opt("rolling")["best_lag"]
            >>> best_lag
            10
        """

        def calc_rolling(data, lag_start, lag_end) -> pd.Series:
            window_width = lag_start - lag_end
            if data.shape[0] < lag_end:
                raise RuntimeError(
                    f"The provided data shape is {data.shape}. More than {lag_end} records are required."
                )
            lags = time_based_average_data.shift(lag_end)
            rollings = lags.rolling(window_width).mean()
            return rollings

        dataset = pd.concat([train, valid])
        time_based_average_data = self.create_time_based_average_data(dataset, target)

        if config:
            best_lag, lag_size = self.get_preprocessors()["rolling"].values()
            if best_lag > lag_size:
                rollings = calc_rolling(time_based_average_data, best_lag, lag_size)
            else:
                return train, valid

        else:
            opt = self.get_opt("rolling")
            opt.setdefault("lag_size", 1)
            opt.setdefault("best_lag", self.check_corr(time_based_average_data)[0])

            lag_size = opt["lag_size"]
            best_lag = opt["best_lag"]
            _logger.info(f"best_lag = {best_lag}")
            self.set_preprocessors({"rolling": {"best_lag": best_lag, "lag_size": lag_size}})

            if best_lag > lag_size:
                rollings = calc_rolling(time_based_average_data, best_lag, lag_size)
            else:
                return train, valid

        train["rolling"] = train["date"].map(rollings[target])
        valid["rolling"] = valid["date"].map(rollings[target])

        return train, valid

    def expanding(self, train, valid, target, config) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate rolling feature from date feature.

        Parameters
        ----------
        train: pandas.DataFrame
            Training data

        valid: pandas.DataFrame
            Validation data

        target: str
            Name of the target variable.

        config: dict
            Configuration for the function (manually or as history).

        Returns
        -------
        tuple
            (pandas.DataFrame, pandas.DataFrame)
            Training and validation data with "expanding" features.

        Notes
        -----
        By default, this function generates the expanding features up to one
        data point before. However, by specifying lag_size as an option for
        the expanding function, it is possible to generate the expanding
        features up to the specified number of data points before.

        You can specify the `lag_size` as follows.
            >>> from cyclic_boosting.tornado.core.preprocess import Preprocess
            >>> opt = {"expanding": {"lag_size": 5}}
            >>> prep = Preprocess(opt)
            >>> lag_size = prep.get_opt("expanding")["lag_size"]
            >>> lag_size
            5
        """

        def calc_expanding(data, lag_end) -> pd.Series:
            if data.shape[0] < lag_end:
                raise RuntimeError(
                    f"The provided data shape is {data.shape}. More than {lag_end} records are required."
                )
            lags = time_based_average_data.shift(lag_end)
            expandings = lags.expanding().mean()
            return expandings

        dataset = pd.concat([train, valid])
        time_based_average_data = self.create_time_based_average_data(dataset, target)

        if config:
            lag_size = self.get_preprocessors()["expanding"]["lag_size"]
            expandings = calc_expanding(time_based_average_data, lag_size)

        else:
            opt = self.get_opt("expanding")
            opt.setdefault("lag_size", 1)

            lag_size = opt["lag_size"]
            self.set_preprocessors({"expanding": {"lag_size": lag_size}})
            expandings = calc_expanding(time_based_average_data, lag_size)

        train["expanding"] = train["date"].map(expandings[target])
        valid["expanding"] = valid["date"].map(expandings[target])

        return train, valid

    def check_cardinality(self, train, valid, target, config) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Warn for features with high cardinality.

        Users can set specific cardinality thresholds as an option.

        Parameters
        ----------
        train: pandas.DataFrame
            Training data

        valid: pandas.DataFrame
            Validation data

        target: str
            Name of the target variable.

        config: dict
            Configuration for the function (manually or as history).

        Returns
        -------
        tuple
            (pandas.DataFrame, pandas.DataFrame)
            Training and validation data
        """
        if not config:
            opt = self.get_opt("check_cardinality")
            opt.setdefault("cardinality_th", 0.8)

            cardinality_th = opt["cardinality_th"]
            dataset = pd.concat([train, valid])
            category_df = dataset.drop(columns=target).select_dtypes("int")

            high_cardinality_cols = []
            for col in category_df.columns:
                unique_values = category_df[col].nunique()
                if unique_values / len(category_df) > cardinality_th:
                    high_cardinality_cols.append(col)

            if len(high_cardinality_cols) > 0:
                _logger.warning(
                    f"The cardinality of the {high_cardinality_cols} column is very high.\n"
                    "    By using methods such as hierarchical grouping,\n"
                    "    the cardinality can be reduced,\n"
                    "    leading to an improvement in inference accuracy."
                )

            self.set_preprocessors({"check_cardinality": {}})

        return train, valid

    def check_dtype(self, train, valid, target, config) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Warn for float type features with no decimals.

        Parameters
        ----------
        train: pandas.DataFrame
            Training data

        valid: pandas.DataFrame
            Validation data

        target: str
            Name of the target variable.

        config: dict
            Configuration for the function (manually or as history).

        Returns
        -------
        tuple
            (pandas.DataFrame, pandas.DataFrame)
            Training and validation data
        """
        if not config:
            dataset = pd.concat([train, valid])
            float_dataset = dataset.drop(columns=target).select_dtypes("float")

            float_integer_cols = []
            for col in float_dataset.columns:
                if float_dataset[col].apply(lambda x: x.is_integer()).all():
                    float_integer_cols.append(col)

            if len(float_integer_cols) > 0:
                _logger.warning(
                    f"Please check the columns {float_integer_cols}.\n"
                    "    Ensure that categorical variables are of 'int' type\n"
                    "    and continuous variables are of 'float' type."
                )

            self.set_preprocessors({"check_dtype": {}})

        return train, valid

    def check_data_leakage(self, train, valid, target, config) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Warn for possible target leakage.

        Warn of possible target leakage when VIF (Variance Inflation Factor)
        of the objective variable is infinite.

        Parameters
        ----------
        train: pandas.DataFrame
            Training data

        valid: pandas.DataFrame
            Validation data

        target: str
            Name of the target variable.

        config: dict
            Configuration for the function (manually or as history).

        Returns
        -------
        tuple
            (pandas.DataFrame, pandas.DataFrame)
            Training and validation data
        """
        if not config:
            dataset = pd.concat([train, valid])
            dataset = dataset.select_dtypes(include=["int", "float"])
            dataset = dataset.dropna(axis=1)  # ignore columns included float with Nan

            with warnings.catch_warnings(record=True) as w:
                vif = pd.DataFrame(index=dataset.columns)
                vif["VIF Factor"] = [variance_inflation_factor(dataset.values, i) for i in range(dataset.shape[1])]
                is_target_leaked = vif.loc[target, "VIF Factor"] == np.inf

                if is_target_leaked and len(w) > 0:
                    _logger.warning(
                        "Variance Inflation Factor (VIF) of the objective variable is infinite.\n"
                        "This means that there is a very high association (multi-collinearity)\n"
                        "between the explanatory variables and the objective variable.\n"
                        "Confirmation is recommended due to the possibility of target leakage."
                    )

            self.set_preprocessors({"check_data_leakage": {}})

        return train, valid

    def standardization(self, train, valid, target, config) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate features with standardization applied.

        Users can specify several arguments of the :class:`StandardScaler` as
        options.

        Parameters
        ----------
        train: pandas.DataFrame
            Training data

        valid: pandas.DataFrame
            Validation data

        target: str
            Name of the target variable.

        config: dict
            Configuration for the function (manually or as history).

        Returns
        -------
        tuple
            (pandas.DataFrame, pandas.DataFrame)
            Training and validation data with standardized features.
        """
        float_train = train.drop(columns=target).select_dtypes("float")

        if len(float_train.columns) > 0:
            opt = self.get_opt("standardization")
            scaler = StandardScaler(**opt)

            if config:
                attr = self.get_preprocessors()["standardization"]["attr"]
                setattr(scaler, "__dict__", attr)
            else:
                scaler.fit(train[float_train.columns])
                attr = getattr(scaler, "__dict__")
                self.set_preprocessors({"standardization": {"attr": attr}})

            columns = float_train.columns + "_standardization"
            train[columns] = scaler.transform(train[float_train.columns])
            valid[columns] = scaler.transform(valid[float_train.columns])

        return train, valid

    def minmax(self, train, valid, target, config) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate features with min-max normalization applied.

        Users can specify several arguments of the :class:`MinMaxScaler` as
        options.

        Parameters
        ----------
        train: pandas.DataFrame
            Training data

        valid: pandas.DataFrame
            Validation data

        target: str
            Name of the target variable.

        config: dict
            Configuration for the function (manually or as history).

        Returns
        -------
        tuple
            (pandas.DataFrame, pandas.DataFrame)
            Training and validation data with min-max scaled features.
        """
        float_train = train.drop(columns=target).select_dtypes("float")

        if len(float_train.columns) > 0:
            opt = self.get_opt("minmax")
            scaler = MinMaxScaler(**opt)

            if config:
                attr = self.get_preprocessors()["minmax"]["attr"]
                setattr(scaler, "__dict__", attr)
            else:
                scaler.fit(train[float_train.columns])
                attr = getattr(scaler, "__dict__")
                self.set_preprocessors({"minmax": {"attr": attr}})

            columns = float_train.columns + "_minmax"
            train[columns] = scaler.transform(train[float_train.columns])
            valid[columns] = scaler.transform(valid[float_train.columns])

        return train, valid

    def logarithmic(self, train, valid, target, config) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate features with log-transformation applied.

        Users can specify several arguments of the :class:`PowerTransformer` as
        options.

        Parameters
        ----------
        train: pandas.DataFrame
            Training data

        valid: pandas.DataFrame
            Validation data

        target: str
            Name of the target variable.

        config: dict
            Configuration for the function (manually or as history).

        Returns
        -------
        tuple
            (pandas.DataFrame, pandas.DataFrame)
            Training and validation data with log-transformed features.
        """
        float_train = train.drop(columns=target).select_dtypes("float")

        if len(float_train.columns) > 0:
            opt = self.get_opt("logarithmic")
            pt = PowerTransformer(**opt)

            if config:
                attr = self.get_preprocessors()["logarithmic"]["attr"]
                setattr(pt, "__dict__", attr)
            else:
                pt.fit(train[float_train.columns])
                attr = getattr(pt, "__dict__")
                self.set_preprocessors({"logarithmic": {"attr": attr}})

            columns = float_train.columns + "_logarithmic"
            train[columns] = pt.transform(train[float_train.columns])
            valid[columns] = pt.transform(valid[float_train.columns])

        return train, valid

    def clipping(self, train, valid, target, config) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate features with clipping applied.

        Users can set an upper and lower quartile limit for clipping data as
        an option (default is 0.01 for the lower limit and 0.99 for the upper
        limit).

        Parameters
        ----------
        train: pandas.DataFrame
            Training data

        valid: pandas.DataFrame
            Validation data

        target: str
            Name of the target variable.

        config: dict
            Configuration for the function (manually or as history).

        Returns
        -------
        tuple
            (pandas.DataFrame, pandas.DataFrame)
            Training and validation data including features clipped at the
            upper and lower limits.
        """
        float_train = train.drop(columns=target).select_dtypes("float")

        if len(float_train.columns) > 0:
            if config:
                p = self.get_preprocessors()["clipping"]
                p_l = p["p_l"]
                p_u = p["p_u"]
            else:
                opt = self.get_opt("clipping")
                opt.setdefault("q_l", 0.01)
                opt.setdefault("q_u", 0.99)
                p_l = train[float_train.columns].quantile(opt["q_l"])
                p_u = train[float_train.columns].quantile(opt["q_u"])
                self.set_preprocessors({"clipping": {"p_l": p_l, "p_u": p_u}})

            columns = float_train.columns + "_clipping"
            train[columns] = train[float_train.columns].clip(p_l, p_u, axis=1)
            valid[columns] = valid[float_train.columns].clip(p_l, p_u, axis=1)

        return train, valid

    def binning(self, train, valid, target, config) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate features with binning applied.

        Users can specify several arguments of the :class:`KBinsDiscretizer` as
        options.

        Parameters
        ----------
        train: pandas.DataFrame
            Training data

        valid: pandas.DataFrame
            Validation data

        target: str
            Name of the target variable.

        config: dict
            Configuration for the function (manually or as history).

        Returns
        -------
        tuple
            (pandas.DataFrame, pandas.DataFrame)
            Training and validation data with binned features.
        """
        float_train = train.drop(columns=target).select_dtypes("float")

        if len(float_train.columns) > 0:
            opt = self.get_opt("binning")
            opt.setdefault("n_bins", int(np.log2(float_train.shape[0]) + 1))
            opt.setdefault("encode", "ordinal")
            opt.setdefault("strategy", "uniform")
            opt.setdefault("random_state", 0)
            opt.setdefault("subsample", 200000)
            binner = KBinsDiscretizer(**opt)

            if config:
                attr = self.get_preprocessors()["binning"]["attr"]
                setattr(binner, "__dict__", attr)
            else:
                binner.fit(train[float_train.columns])
                attr = getattr(binner, "__dict__")
                self.set_preprocessors({"binning": {"attr": attr}})

            columns = float_train.columns + "_binning"
            train[columns] = binner.transform(train[float_train.columns])
            valid[columns] = binner.transform(valid[float_train.columns])

        return train, valid

    def rank(self, train, valid, target, config) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate features by converting to rank values.

        Users can specify several arguments of the :class:`QuantileTransformer`
        as options.

        Parameters
        ----------
        train: pandas.DataFrame
            Training data

        valid: pandas.DataFrame
            Validation data

        target: str
            Name of the target variable.

        config: dict
            Configuration for the function (manually or as history).

        Returns
        -------
        tuple
            (pandas.DataFrame, pandas.DataFrame)
            Training and validation data with features converted to ranks.
        """
        float_train = train.drop(columns=target).select_dtypes("float")

        if len(float_train.columns) > 0:
            opt = self.get_opt("rank")
            opt.setdefault("n_quantiles", min(1000, float_train.shape[0]))
            opt.setdefault("output_distribution", "uniform")
            opt.setdefault("random_state", 0)
            qt = QuantileTransformer(**opt)

            if config:
                attr = self.get_preprocessors()["rank"]["attr"]
                setattr(qt, "__dict__", attr)
            else:
                qt.fit(train[float_train.columns])
                attr = getattr(qt, "__dict__")
                self.set_preprocessors({"rank": {"attr": attr}})

            columns = float_train.columns + "_rank"
            train[columns] = qt.transform(train[float_train.columns])
            valid[columns] = qt.transform(valid[float_train.columns])

        return train, valid

    def rankgauss(self, train, valid, target, config) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate features transformed by Rank Gauss normalization.

        Users can specify several arguments of the :class:`QuantileTransformer`
        as options.

        Parameters
        ----------
        train: pandas.DataFrame
            Training data

        valid: pandas.DataFrame
            Validation data

        target: str
            Name of the target variable.

        config: dict
            Configuration for the function (manually or as history).

        Returns
        -------
        tuple
            (pandas.DataFrame, pandas.DataFrame)
            Training and validation data with features transformed by
            Rank Gauss normalization.
        """
        float_train = train.drop(columns=target).select_dtypes("float")

        if len(float_train.columns) > 0:
            opt = self.get_opt("rankgauss")
            opt.setdefault("n_quantiles", min(1000, float_train.shape[0]))
            opt.setdefault("output_distribution", "normal")
            opt.setdefault("random_state", 0)
            qt = QuantileTransformer(**opt)

            if config:
                attr = self.get_preprocessors()["rankgauss"]["attr"]
                setattr(qt, "__dict__", attr)
            else:
                qt.fit(train[float_train.columns])
                attr = getattr(qt, "__dict__")
                self.set_preprocessors({"rankgauss": {"attr": attr}})

            columns = float_train.columns + "_rankgauss"
            train[columns] = qt.transform(train[float_train.columns])
            valid[columns] = qt.transform(valid[float_train.columns])

        return train, valid

    def onehot_encoding(self, train, valid, target, config) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Encode categorical features with one-hot encoding.

        Users can specify several arguments of the :class:`OneHotEncoder` as
        options.

        Parameters
        ----------
        train: pandas.DataFrame
            Training data

        valid: pandas.DataFrame
            Validation data

        target: str
            Name of the target variable.

        config: dict
            Configuration for the function (manually or as history).

        Returns
        -------
        tuple
            (pandas.DataFrame, pandas.DataFrame)
            Training and validation data with categorical features encoded
            using one-hot encoding.
        """
        dataset = pd.concat([train, valid])
        object_dataset = dataset.drop(columns=["date", target], errors="ignore").select_dtypes("object")

        if len(object_dataset.columns) > 0:
            opt = self.get_opt("onehot_encoding")
            opt.setdefault("handle_unknown", "ignore")
            opt.setdefault("sparse_output", False)
            opt.setdefault("dtype", np.int64)
            ohe = OneHotEncoder(**opt)

            if config:
                attr = self.get_preprocessors()["encode_category"]["onehot_encoding"]["attr"]
                setattr(ohe, "__dict__", attr)
            else:
                ohe.fit(dataset[object_dataset.columns])
                attr = getattr(ohe, "__dict__")
                self.set_preprocessors({"encode_category": {"onehot_encoding": {"attr": attr}}})

            columns = []
            for i, c in enumerate(object_dataset.columns):
                columns += [f"{c}_{v}" for v in ohe.categories_[i]]

            train_ohe = pd.DataFrame(ohe.transform(train[object_dataset.columns]), index=train.index, columns=columns)
            valid_ohe = pd.DataFrame(ohe.transform(valid[object_dataset.columns]), index=valid.index, columns=columns)

            train = pd.concat([train.drop(object_dataset.columns, axis=1), train_ohe], axis=1)
            valid = pd.concat([valid.drop(object_dataset.columns, axis=1), valid_ohe], axis=1)

            self.train.drop(object_dataset.columns, axis=1, inplace=True)
            self.valid.drop(object_dataset.columns, axis=1, inplace=True)

        return train, valid

    def label_encoding(self, train, valid, target, config) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Encode categorical features with label encoding.

        Users can specify several arguments of the :class:`OrdinalEncoder` as
        options.

        Parameters
        ----------
        train: pandas.DataFrame
            Training data

        valid: pandas.DataFrame
            Validation data

        target: str
            Name of the target variable.

        config: dict
            Configuration for the function (manually or as history).

        Returns
        -------
        tuple
            (pandas.DataFrame, pandas.DataFrame)
            Training and validation data with categorical features encoded
            using label encoding.
        """
        dataset = pd.concat([train, valid])
        object_dataset = dataset.drop(columns=["date", target], errors="ignore").select_dtypes("object")

        if len(object_dataset.columns) > 0:
            opt = self.get_opt("label_encoding")
            opt.setdefault("handle_unknown", "use_encoded_value")
            opt.setdefault("unknown_value", -1)
            opt.setdefault("encoded_missing_value", -2)
            opt.setdefault("dtype", np.int64)
            oe = OrdinalEncoder(**opt)

            if config:
                attr = self.get_preprocessors()["encode_category"]["label_encoding"]["attr"]
                setattr(oe, "__dict__", attr)
            else:
                oe.fit(dataset[object_dataset.columns])
                attr = getattr(oe, "__dict__")
                self.set_preprocessors({"encode_category": {"label_encoding": {"attr": attr}}})

            train[object_dataset.columns + "_label_encoding"] = oe.transform(train[object_dataset.columns])
            valid[object_dataset.columns + "_label_encoding"] = oe.transform(valid[object_dataset.columns])

            train.drop(object_dataset.columns, axis=1, inplace=True)
            valid.drop(object_dataset.columns, axis=1, inplace=True)

            self.train.drop(object_dataset.columns, axis=1, inplace=True)
            self.valid.drop(object_dataset.columns, axis=1, inplace=True)

        return train, valid

    def feature_hashing(self, train, valid, target, config) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Encode categorical features with feature hashing.

        Users can specify several arguments of the :class:`FeatureHasher` as
        options.

        Parameters
        ----------
        train: pandas.DataFrame
            Training data

        valid: pandas.DataFrame
            Validation data

        target: str
            Name of the target variable.

        config: dict
            Configuration for the function (manually or as history).

        Returns
        -------
        tuple
            (pandas.DataFrame, pandas.DataFrame)
            Training and validation data with categorical features encoded
            using feature hashing.
        """
        object_train = train.drop(columns=["date", target], errors="ignore").select_dtypes("object")
        object_valid = valid.drop(columns=["date", target], errors="ignore").select_dtypes("object")

        if len(object_train.columns) > 0:
            opt = self.get_opt("feature_hashing")
            if config:
                opt.setdefault(
                    "n_features", self.get_preprocessors()["encode_category"]["feature_hashing"]["n_features"]
                )
            else:
                opt.setdefault("n_features", 10)
            opt.setdefault("input_type", "string")
            opt.setdefault("dtype", np.int64)
            n_features = opt["n_features"]

            for col in object_train.columns:
                fh = FeatureHasher(**opt)
                hash_train = fh.transform(object_train[col].astype(str).values[:, np.newaxis].tolist())
                hash_valid = fh.transform(object_valid[col].astype(str).values[:, np.newaxis].tolist())
                hash_train = pd.DataFrame(
                    hash_train.todense(), index=train.index, columns=[f"{col}_{i}" for i in range(n_features)]
                )
                hash_valid = pd.DataFrame(
                    hash_valid.todense(), index=valid.index, columns=[f"{col}_{i}" for i in range(n_features)]
                )
                train = pd.concat([train, hash_train], axis=1)
                valid = pd.concat([valid, hash_valid], axis=1)
            self.set_preprocessors({"encode_category": {"feature_hashing": {"n_features": n_features}}})

            train.drop(object_train.columns, axis=1, inplace=True)
            valid.drop(object_valid.columns, axis=1, inplace=True)

            self.train.drop(object_train.columns, axis=1, inplace=True)
            self.valid.drop(object_valid.columns, axis=1, inplace=True)

        return train, valid

    def frequency_encoding(self, train, valid, target, config) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Encode categorical features with frequency encoding.

        Parameters
        ----------
        train: pandas.DataFrame
            Training data

        valid: pandas.DataFrame
            Validation data

        target: str
            Name of the target variable.

        config: dict
            Configuration for the function (manually or as history).

        Returns
        -------
        tuple
            (pandas.DataFrame, pandas.DataFrame)
            Training and validation data with categorical features encoded
            using frequency encoding.
        """
        object_train = train.drop(columns=["date", target], errors="ignore").select_dtypes("object")

        if len(object_train.columns) > 0:
            if config:
                freqs = self.get_preprocessors()["encode_category"]["frequency_encoding"]["freqs"]
            else:
                freqs = {}
                for col in object_train.columns:
                    freqs[col] = train[col].value_counts()
                self.set_preprocessors({"encode_category": {"frequency_encoding": {"freqs": freqs}}})
            for col in object_train.columns:
                train[col] = train[col].map(freqs[col])
                valid[col] = valid[col].map(freqs[col])
                train.rename(columns={col: col + "_frequency_encoding"}, inplace=True)
                valid.rename(columns={col: col + "_frequency_encoding"}, inplace=True)

            self.train.drop(object_train.columns, axis=1, inplace=True)
            self.valid.drop(object_train.columns, axis=1, inplace=True)

        return train, valid

    def target_encoding(self, train, valid, target, config) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Encode categorical features with target encoding.

        Users can specify several arguments of the :class:`TargetEncoder` as
        options.

        Parameters
        ----------
        train: pandas.DataFrame
            Training data

        valid: pandas.DataFrame
            Validation data

        target: str
            Name of the target variable.

        config: dict
            Configuration for the function (manually or as history).

        Returns
        -------
        tuple
            (pandas.DataFrame, pandas.DataFrame)
            Training and validation data with categorical features encoded
            using target encoding.
        """
        object_train = train.drop(columns=["date", target], errors="ignore").select_dtypes("object")

        if len(object_train.columns) > 0:

            if config:
                attrs = self.get_preprocessors()["encode_category"]["target_encoding"]["attrs"]
                for col in object_train.columns:
                    te = TargetEncoder()
                    attr = attrs[col]
                    setattr(te, "__dict__", attr)
                    X_train = np.array([train[col].values]).T
                    X_valid = np.array([valid[col].values]).T
                    train[col] = te.transform(X_train)
                    valid[col] = te.transform(X_valid)
                    train.rename(columns={col: col + "_target_encoding"}, inplace=True)
                    valid.rename(columns={col: col + "_target_encoding"}, inplace=True)

            else:
                opt = self.get_opt("target_encoding")
                opt.setdefault("random_state", 0)
                attrs = {}
                for col in object_train.columns:
                    te = TargetEncoder(**opt)
                    X_train = np.array([train[col].values]).T
                    X_valid = np.array([valid[col].values]).T
                    y = train[target].values
                    train[col] = te.fit_transform(X_train, y)
                    valid[col] = te.transform(X_valid)
                    attr = getattr(te, "__dict__")
                    attrs[col] = attr
                    train.rename(columns={col: col + "_target_encoding"}, inplace=True)
                    valid.rename(columns={col: col + "_target_encoding"}, inplace=True)
                self.set_preprocessors({"encode_category": {"target_encoding": {"attrs": attrs}}})

            self.train.drop(columns=object_train.columns, inplace=True)
            self.valid.drop(columns=object_train.columns, inplace=True)

        return train, valid

    def encode_category(self, train, valid, target, *args) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Encode categorical features.

        Parameters
        ----------
        train: pandas.DataFrame
            Training data

        valid: pandas.DataFrame
            Validation data

        target: str
            Name of the target variable.

        Returns
        -------
        tuple
            (pandas.DataFrame, pandas.DataFrame)
            Training and validation data with encoded categorical features.
        """
        dataset = pd.concat([train, valid])
        object_df = dataset.drop(columns=["date", target], errors="ignore").select_dtypes("object")

        category = []
        for col in object_df.columns:
            subset = object_df[col].dropna()
            subset = subset.values
            is_str = [1 for data in subset if isinstance(data, str)]
            if len(subset) == len(is_str):
                category.append(col)
            else:
                raise RuntimeError("The dataset has differenct dtype in same col")

        if len(category) > 0:
            encoders = self.get_preprocessors().copy()["encode_category"]
            if len(encoders) != 1:
                raise RuntimeError("Single encoding method should be used for categorical variables.")
            for enc, config in encoders.items():
                train, valid = eval(f"self.{enc}")(train, valid, target, config)

            self.train_raw = train
            self.valid_raw = valid

        return train, valid
