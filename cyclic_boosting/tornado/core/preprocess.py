# Add some comments
#
#
#
import logging

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, \
                                  PowerTransformer, QuantileTransformer, \
                                  OrdinalEncoder, KBinsDiscretizer, \
                                  OneHotEncoder, TargetEncoder
from sklearn.feature_extraction import FeatureHasher
import statsmodels.api as sm

_logger = logging.getLogger(__name__)


class Preprocess():
    def __init__(self, opt):
        self.preprocessors = {}
        self.opt = opt

    def get_preprocessors(self) -> dict:
        return self.preprocessors

    def set_preprocessors(self, preps) -> None:
        for prep, params in preps.items():
            self.preprocessors[prep] = params

    def get_opt(self, func_name) -> dict:
        try:
            opt = self.opt[func_name]
        except KeyError:
            opt = {}

        return opt

    def load_dataset(self, path_ds) -> pd.DataFrame:
        # transfrom column name to lower
        if path_ds.endswith(".csv"):
            dataset = self.tolowerstr(pd.read_csv(path_ds))
        elif path_ds.endswith(".xlsx"):
            dataset = self.tolowerstr(pd.read_excel(path_ds))
        else:
            _logger.error("The file format is not supported.\n"
                          "Please use the csv or xlsx format.")
        return dataset

    def check_data(self, dataset, is_ts) -> None:
        # datasetに対してどんな前処理を施す必要があるかを調べて返す
        if self.preprocessors:
            return
        col_names = dataset.columns.to_list()

        self.preprocessors['encode_category'] = {}
        self.preprocessors['check_dtype'] = {}
        self.preprocessors['check_cardinality'] = {}

        if is_ts:
            if "date" in col_names:
                self.preprocessors['todatetime'] = {}
                self.preprocessors['lag'] = {}
                self.preprocessors['rolling'] = {}
                self.preprocessors['expanding'] = {}
            else:
                _logger.error("If this is a forecast of time-series data,\n"
                              " a 'date' column is required to identify\n"
                              " the datetime of the data.")

            if "dayofweek" not in col_names and "date" in col_names:
                self.preprocessors['dayofweek'] = {}

            if "dayofyear" not in col_names and "date" in col_names:
                self.preprocessors['dayofyear'] = {}

        self.preprocessors['standarlization'] = {}
        self.preprocessors['minmax'] = {}
        self.preprocessors['logarithmic'] = {}
        self.preprocessors['clipping'] = {}
        self.preprocessors['binning'] = {}
        self.preprocessors['rank'] = {}
        self.preprocessors['rankgauss'] = {}

    def apply(self, train, valid, target) -> None:
        # 特徴量エンジニアリングを実行
        self.train_raw = train.copy()
        self.valid_raw = valid.copy()
        self.train = train.copy()
        self.valid = valid.copy()
        preprocessors = self.get_preprocessors().copy()
        for prep, params in preprocessors.items():
            train, valid = eval(f'self.{prep}')(self.train_raw.copy(),
                                                self.valid_raw.copy(),
                                                target,
                                                any(params))
            train[self.train.columns] = self.train
            valid[self.valid.columns] = self.valid
            self.train = train.copy()
            self.valid = valid.copy()

        return self.train, self.valid

    def todatetime(self, train, valid, target, params_exist) -> pd.DataFrame:
        train["date"] = pd.to_datetime(train["date"])
        valid["date"] = pd.to_datetime(valid["date"])

        self.set_preprocessors({'todatetime': {}})
        self.train.drop("date", axis=1, inplace=True)
        self.valid.drop("date", axis=1, inplace=True)
        self.train_raw = train
        self.valid_raw = valid

        return train, valid

    def tolowerstr(self, dataset) -> pd.DataFrame:
        renames = {before: before.lower() for before in dataset.columns}
        dataset = dataset.rename(columns=renames)

        return dataset

    def dayofweek(self, train, valid, target, params_exist) -> pd.DataFrame:
        train["dayofweek"] = train["date"].dt.dayofweek
        valid["dayofweek"] = valid["date"].dt.dayofweek
        train["dayofweek"] = train["dayofweek"].astype('int64')
        valid["dayofweek"] = valid["dayofweek"].astype('int64')
        self.set_preprocessors({'dayofweek': {}})

        return train, valid

    def dayofyear(self, train, valid, target, params_exist) -> pd.DataFrame:
        train["dayofyear"] = train["date"].dt.dayofyear
        valid["dayofyear"] = valid["date"].dt.dayofyear
        train["dayofyear"] = train["dayofyear"].astype('int64')
        valid["dayofyear"] = valid["dayofyear"].astype('int64')
        self.set_preprocessors({'dayofyear': {}})

        return train, valid

    def create_daily_data(self, dataset, target) -> pd.DataFrame:
        daily_data = dataset.groupby("date")[target].mean()
        daily_data = pd.DataFrame(daily_data).sort_index()

        return daily_data

    def check_corr(self, daily_data) -> pd.DataFrame:
        nlags = int(len(daily_data)/2-1)
        nlags = min(nlags, 500)
        acf = sm.tsa.stattools.acf(daily_data, nlags=nlags)
        pacf = sm.tsa.stattools.pacf(daily_data, method='ywm', nlags=nlags)
        argmax_acf = np.argmax(acf[1:])+1
        argmax_pacf = np.argmax(pacf[1:])+1

        return argmax_acf, argmax_pacf

    def lag(self, train, valid, target, params_exist) -> pd.DataFrame:
        if params_exist:
            lags = self.get_preprocessors()['lag']['lags']
        else:
            dataset = pd.concat([train, valid])
            daily_data = self.create_daily_data(dataset, target)
            opt = self.get_opt('lag')
            opt.setdefault('lag_size', self.check_corr(daily_data)[1])
            lag_size = opt['lag_size']
            _logger.info(f"lag_size = {lag_size}")
            lags = daily_data.shift(lag_size)
            self.set_preprocessors({'lag': {'lags': lags}})
        train["lag"] = train["date"].map(lags[target])
        valid["lag"] = valid["date"].map(lags[target])

        return train, valid

    def rolling(self, train, valid, target, params_exist) -> pd.DataFrame:
        if params_exist:
            rollings = self.get_preprocessors()['rolling']['rollings']
        else:
            dataset = pd.concat([train, valid])
            daily_data = self.create_daily_data(dataset, target)
            opt = self.get_opt('rolling')
            opt.setdefault('lag_size', 1)
            opt.setdefault('best_lag', self.check_corr(daily_data)[0])
            lag_size = opt['lag_size']
            best_lag = opt['best_lag']
            _logger.info(f"best_lag = {best_lag}")
            if best_lag > lag_size:
                lags = daily_data.shift(lag_size)
                rollings = lags.rolling(best_lag - lag_size).mean()
            else:
                return train, valid
            self.set_preprocessors({'rolling': {'rollings': rollings}})
        train["rolling"] = train["date"].map(rollings[target])
        valid["rolling"] = valid["date"].map(rollings[target])

        return train, valid

    def expanding(self, train, valid, target, params_exist) -> pd.DataFrame:
        if params_exist:
            expandings = self.get_preprocessors()['expanding']['expandings']
        else:
            dataset = pd.concat([train, valid])
            daily_data = self.create_daily_data(dataset, target)
            opt = self.get_opt('expanding')
            opt.setdefault('lag_size', 1)
            lag_size = opt['lag_size']
            lags = daily_data.shift(lag_size)
            expandings = lags.expanding().mean()
            self.set_preprocessors({'expanding': {'expandings': expandings}})
        train["expanding"] = train["date"].map(expandings[target])
        valid["expanding"] = valid["date"].map(expandings[target])

        return train, valid

    def check_cardinality(self, train, valid, target, params_exist) -> pd.DataFrame:
        if not params_exist:
            opt = self.get_opt('check_cardinality')
            opt.setdefault('cardinality_th', 0.8)
            cardinality_th = opt['cardinality_th']
            dataset = pd.concat([train, valid])
            category_df = dataset.drop(columns=target).select_dtypes('int')
            high_cardinality_cols = []
            for col in category_df.columns:
                unique_values = category_df[col].nunique()
                if unique_values / len(category_df) > cardinality_th:
                    high_cardinality_cols.append(col)
            if len(high_cardinality_cols) > 0:
                _logger.warn(f"The cardinality of the {high_cardinality_cols} "
                             "column is very high.\n    By using methods such "
                             "as hierarchical grouping,\n    the cardinality "
                             "can be reduced, leading to an improvement\n    "
                             "in inference accuracy.")
            self.set_preprocessors({'check_cardinality': {}})

        return train, valid

    def check_dtype(self, train, valid, target, params_exist) -> pd.DataFrame:
        if not params_exist:
            dataset = pd.concat([train, valid])
            float_datset = dataset.drop(columns=target).select_dtypes('float')
            float_integer_cols = []
            for col in float_datset.columns:
                if float_datset[col].apply(lambda x: x.is_integer()).all():
                    float_integer_cols.append(col)
            if len(float_integer_cols) > 0:
                _logger.warn(f"Please check the columns {float_integer_cols}."
                             "\n    Ensure that categorical variables are of "
                             "'int' type\n    and continuous variables are of "
                             "'float' type.")
            self.set_preprocessors({'check_dtype': {}})

        return train, valid

    def standarlization(self, train, valid, target, params_exist) -> pd.DataFrame:
        float_train = train.drop(columns=target).select_dtypes('float')
        if len(float_train.columns) > 0:
            opt = self.get_opt('standarlization')
            scaler = StandardScaler(**opt)
            if params_exist:
                attr = self.get_preprocessors()['standarlization']['attr']
                setattr(scaler, '__dict__', attr)
            else:
                scaler.fit(train[float_train.columns])
                attr = getattr(scaler, '__dict__')
                self.set_preprocessors({'standarlization': {'attr': attr}})
            columns = float_train.columns + '_standarlization'
            train[columns] = scaler.transform(train[float_train.columns])
            valid[columns] = scaler.transform(valid[float_train.columns])

        return train, valid

    def minmax(self, train, valid, target, params_exist) -> pd.DataFrame:
        float_train = train.drop(columns=target).select_dtypes('float')
        if len(float_train.columns) > 0:
            opt = self.get_opt('minmax')
            scaler = MinMaxScaler(**opt)
            if params_exist:
                attr = self.get_preprocessors()['minmax']['attr']
                setattr(scaler, '__dict__', attr)
            else:
                scaler.fit(train[float_train.columns])
                attr = getattr(scaler, '__dict__')
                self.set_preprocessors({'minmax': {'attr': attr}})
            columns = float_train.columns + '_minmax'
            train[columns] = scaler.transform(train[float_train.columns])
            valid[columns] = scaler.transform(valid[float_train.columns])

        return train, valid

    def logarithmic(self, train, valid, target, params_exist) -> pd.DataFrame:
        float_train = train.drop(columns=target).select_dtypes('float')
        if len(float_train.columns) > 0:
            opt = self.get_opt('logarithmic')
            pt = PowerTransformer(**opt)
            if params_exist:
                attr = self.get_preprocessors()['logarithmic']['attr']
                setattr(pt, '__dict__', attr)
            else:
                pt.fit(train[float_train.columns])
                attr = getattr(pt, '__dict__')
                self.set_preprocessors({'logarithmic': {'attr': attr}})
            columns = float_train.columns + '_logarithmic'
            train[columns] = pt.transform(train[float_train.columns])
            valid[columns] = pt.transform(valid[float_train.columns])

        return train, valid

    def clipping(self, train, valid, target, params_exist) -> pd.DataFrame:
        float_train = train.drop(columns=target).select_dtypes('float')
        if len(float_train.columns) > 0:
            if params_exist:
                p = self.get_preprocessors()['clipping']
                p_l = p['p_l']
                p_u = p['p_u']
            else:
                opt = self.get_opt('clipping')
                opt.setdefault('q_l', 0.01)
                opt.setdefault('q_u', 0.99)
                p_l = train[float_train.columns].quantile(opt['q_l'])
                p_u = train[float_train.columns].quantile(opt['q_u'])
                self.set_preprocessors({'clipping': {'p_l': p_l, 'p_u': p_u}})
            columns = float_train.columns + '_clipping'
            train[columns] = train[float_train.columns].clip(p_l, p_u, axis=1)
            valid[columns] = valid[float_train.columns].clip(p_l, p_u, axis=1)

        return train, valid

    def binning(self, train, valid, target, params_exist) -> pd.DataFrame:
        float_train = train.drop(columns=target).select_dtypes('float')
        if len(float_train.columns) > 0:
            opt = self.get_opt('binning')
            opt.setdefault('n_bins', int(np.log2(float_train.shape[0]) + 1))
            opt.setdefault('encode', 'ordinal')
            opt.setdefault('strategy', 'uniform')
            opt.setdefault('random_state', 0)
            binner = KBinsDiscretizer(**opt)
            if params_exist:
                attr = self.get_preprocessors()['binning']['attr']
                setattr(binner, '__dict__', attr)
            else:
                binner.fit(train[float_train.columns])
                attr = getattr(binner, '__dict__')
                self.set_preprocessors({'binning': {'attr': attr}})
            columns = float_train.columns + '_binning'
            train[columns] = binner.transform(train[float_train.columns])
            valid[columns] = binner.transform(valid[float_train.columns])

        return train, valid

    def rank(self, train, valid, target, params_exist) -> pd.DataFrame:
        float_train = train.drop(columns=target).select_dtypes('float')
        if len(float_train.columns) > 0:
            opt = self.get_opt('rank')
            # opt.setdefault('n_quantiles', float_train.shape[0])
            opt.setdefault('output_distribution', 'uniform')
            opt.setdefault('random_state', 0)
            qt = QuantileTransformer(**opt)
            if params_exist:
                attr = self.get_preprocessors()['rank']['attr']
                setattr(qt, '__dict__', attr)
            else:
                qt.fit(train[float_train.columns])
                attr = getattr(qt, '__dict__')
                self.set_preprocessors({'rank': {'attr': attr}})
            columns = float_train.columns + '_rank'
            train[columns] = qt.transform(train[float_train.columns])
            valid[columns] = qt.transform(valid[float_train.columns])

        return train, valid

    def rankgauss(self, train, valid, target, params_exist) -> pd.DataFrame:
        float_train = train.drop(columns=target).select_dtypes('float')
        if len(float_train.columns) > 0:
            opt = self.get_opt('rankgauss')
            # opt.setdefault('n_quantiles', float_train.shape[0])
            opt.setdefault('output_distribution', 'normal')
            opt.setdefault('random_state', 0)
            qt = QuantileTransformer(**opt)
            if params_exist:
                attr = self.get_preprocessors()['rankgauss']['attr']
                setattr(qt, '__dict__', attr)
            else:
                qt.fit(train[float_train.columns])
                attr = getattr(qt, '__dict__')
                self.set_preprocessors({'rankgauss': {'attr': attr}})
            columns = float_train.columns + '_rankgauss'
            train[columns] = qt.transform(train[float_train.columns])
            valid[columns] = qt.transform(valid[float_train.columns])

        return train, valid

    def onehot_encording(self, train, valid, target, params_exist) -> pd.DataFrame:
        dataset = pd.concat([train, valid])
        object_dataset = dataset.drop(columns=["date", target]).select_dtypes('object')
        if len(object_dataset.columns) > 0:
            opt = self.get_opt('onehot_encording')
            opt.setdefault('handle_unknown', 'ignore')
            opt.setdefault('sparse_output', False)
            ohe = OneHotEncoder(**opt)
            if params_exist:
                attr = self.get_preprocessors()['onehot_encording']['attr']
                setattr(ohe, '__dict__', attr)
            else:
                ohe.fit(dataset[object_dataset.columns])
                attr = getattr(ohe, '__dict__')
                self.set_preprocessors({'onehot_encording': {'attr': attr}})
            columns = []
            for i, c in enumerate(object_dataset.columns):
                columns += [f'{c}_{v}' for v in ohe.categories_[i]]
            train_ohe = pd.DataFrame(ohe.transform(train[object_dataset.columns]), columns=columns)
            valid_ohe = pd.DataFrame(ohe.transform(valid[object_dataset.columns]), columns=columns)
            train = pd.concat([train.drop[object_dataset.columns], train_ohe], axis=1)
            valid = pd.concat([valid.drop[object_dataset.columns], valid_ohe], axis=1)
            self.train.drop(object_dataset.columns, axis=1, inplace=True)
            self.valid.drop(object_dataset.columns, axis=1, inplace=True)

        return train, valid

    def label_encording(self, train, valid, target, params_exist) -> pd.DataFrame:
        dataset = pd.concat([train, valid])
        object_dataset = dataset.drop(columns=["date", target]).select_dtypes('object')
        if len(object_dataset.columns) > 0:
            opt = self.get_opt('label_encording')
            opt.setdefault('handle_unknown', 'use_encoded_value')
            opt.setdefault('unknown_value', -1)
            opt.setdefault('encoded_missing_value', -2)
            opt.setdefault('dtype', np.int64)
            oe = OrdinalEncoder(**opt)
            if params_exist:
                attr = self.get_preprocessors()['label_encording']['attr']
                setattr(oe, '__dict__', attr)
            else:
                oe.fit(dataset[object_dataset.columns])
                attr = getattr(oe, '__dict__')
                self.set_preprocessors({'label_encording': {'attr': attr}})
            train[object_dataset.columns + 'label_encording'] = oe.transform(train[object_dataset.columns])
            valid[object_dataset.columns + 'label_encording'] = oe.transform(valid[object_dataset.columns])
            train.drop(object_dataset.columns, axis=1, inplace=True)
            valid.drop(object_dataset.columns, axis=1, inplace=True)
            self.train.drop(object_dataset.columns, axis=1, inplace=True)
            self.valid.drop(object_dataset.columns, axis=1, inplace=True)

        return train, valid

    def feature_hashing(self, train, valid, target, params_exist) -> pd.DataFrame:
        object_train = train.drop(columns=["date", target]).select_dtypes('object')
        object_valid = valid.drop(columns=["date", target]).select_dtypes('object')
        if len(object_train.columns) > 0:
            opt = self.get_opt('feature_hashing')
            if params_exist:
                opt.setdefault('n_features', self.get_preprocessors()['feature_hashing']['n_features'])
            else:
                opt.setdefault('n_features', 10)
            opt.setdefault('input_type', 'string')
            n_features = opt['n_features']
            for col in object_train.columns:
                fh = FeatureHasher(**opt)
                hash_train = fh.transform(object_train[col].astype(str).values)
                hash_valid = fh.transform(object_valid[col].astype(str).values)
                hash_train = pd.DataFrame(hash_train.todense(), columns=[f'{col}_{i}' for i in range(n_features)])
                hash_valid = pd.DataFrame(hash_valid.todense(), columns=[f'{col}_{i}' for i in range(n_features)])
                train = pd.concat([train, hash_train], axis=1)
                valid = pd.concat([valid, hash_valid], axis=1)
            self.set_preprocessors({'feature_hashing': {'n_features': n_features}})
            train.drop(object_train.columns, axis=1, inplace=True)
            valid.drop(object_valid.columns, axis=1, inplace=True)
            self.train.drop(object_train.columns, axis=1, inplace=True)
            self.valid.drop(object_valid.columns, axis=1, inplace=True)

        return train, valid

    def freqency_encording(self, train, valid, target, params_exist) -> pd.DataFrame:
        object_train = train.drop(columns=["date", target]).select_dtypes('object')
        if len(object_train.columns) > 0:
            opt = self.get_opt('freqency_encording')
            if params_exist:
                opt.setdefault('freqs', self.get_preprocessors()['freqency_encording']['freqs'])
            else:
                freqs = {}
                for col in object_train.columns:
                    freqs[col] = train[col].value_counts()
                opt.setdefault('freqs', freqs)
                self.set_preprocessors({'freqency_encording': {'freqs': freqs}})
            freqs = opt['freqs']
            for col in object_train.columns:
                train[col] = train[col].map(freqs[col])
                valid[col] = valid[col].map(freqs[col])
                train.rename(columns={col: col + '_freqency_encording'}, inplace=True)
                valid.rename(columns={col: col + '_freqency_encording'}, inplace=True)
            self.train.drop(object_train.columns, axis=1, inplace=True)
            self.valid.drop(object_train.columns, axis=1, inplace=True)

        return train, valid

    def target_encording(self, train, valid, target, params_exist) -> pd.DataFrame:
        object_train = train.drop(columns=["date", target]).select_dtypes('object')
        if len(object_train.columns) > 0:
            if params_exist:
                attrs = self.get_preprocessors()['freqency_encording']['attrs']
                for col in object_train.columns:
                    te = TargetEncoder()
                    attr = attrs[col]
                    setattr(te, '__dict__', attr)
                    X_train = np.array([train[col].values]).T
                    X_valid = np.array([valid[col].values]).T
                    train[col] = te.transform(X_train)
                    valid[col] = te.transform(X_valid)
                    train.rename(columns={col: col + '_target_encording'}, inplace=True)
                    valid.rename(columns={col: col + '_target_encording'}, inplace=True)
            else:
                opt = self.get_opt('target_encording')
                attrs = {}
                for col in object_train.columns:
                    te = TargetEncoder(**opt)
                    X_train = np.array([train[col].values]).T
                    X_valid = np.array([valid[col].values]).T
                    y = train[target].values
                    train[col] = te.fit_transform(X_train, y)
                    valid[col] = te.transform(X_valid)
                    attr = getattr(te, '__dict__')
                    attrs[col] = attr
                    train.rename(columns={col: col + '_target_encording'}, inplace=True)
                    valid.rename(columns={col: col + '_target_encording'}, inplace=True)
                self.set_preprocessors({'target_encording': {'attrs': attrs}})
            self.train.drop(columns=object_train.columns, inplace=True)
            self.valid.drop(columns=object_train.columns, inplace=True)

        return train, valid

    def encode_category(self, train, valid, target, params_exist) -> pd.DataFrame:
        dataset = pd.concat([train, valid])
        object_df = dataset.drop(columns=["date", target]).select_dtypes('object')
        category = []
        for col in object_df.columns:
            subset = object_df[col].dropna()
            subset = subset.values
            is_str = [1 for data in subset if isinstance(data, str)]
            if len(subset) == len(is_str):
                category.append(col)
            else:
                raise RuntimeError('The dataset has differenct '
                                   'dtype in same col')

        if len(category) > 0:
            # NOTE: check unknown_value and encoded_missing_value's behaivier
            # NOTE: and check CB's missing feature processing
            # it might be better than this process
            train, valid = self.target_encording(train,
                                                 valid,
                                                 target,
                                                 params_exist)
            self.train_raw = train
            self.valid_raw = valid

        return train, valid
