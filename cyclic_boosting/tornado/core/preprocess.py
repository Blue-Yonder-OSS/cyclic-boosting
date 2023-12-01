# Add some comments
#
#
#

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder


def todatetime(dataset) -> pd.DataFrame:
    dataset["date"] = pd.to_datetime(dataset["date"])

    return dataset


def tolowerstr(dataset) -> pd.DataFrame:
    renames = {before: before.lower() for before in dataset.columns}
    dataset = dataset.rename(columns=renames)

    return dataset


def dayofweek(dataset) -> pd.DataFrame:
    dataset["dayofweek"] = dataset["date"].dt.dayofweek
    dataset["dayofweek"] = dataset["dayofweek"].astype('int64')

    return dataset


def dayofyear(dataset) -> pd.DataFrame:
    dataset["dayofyear"] = dataset["date"].dt.dayofyear
    dataset["dayofyear"] = dataset["dayofyear"].astype('int64')

    return dataset


def encode_category(dataset) -> pd.DataFrame:
    object_df = dataset.drop(columns="date").select_dtypes('object')
    category = []
    for col in object_df.columns:
        subset = object_df[col].dropna()
        subset = subset.values
        is_str = [1 for data in subset if isinstance(data, str)]
        if len(subset) == len(is_str):
            category.append(col)
        else:
            raise RuntimeError('The dataset has differenct dtype in same col')

    if len(category) > 0:
        # NOTE: check unknown_value and encoded_missing_value's behaivier
        # NOTE: and check CB's missing feature processing
        # it might be better than this process
        enc = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-2,
            dtype=np.int64
        )

        # transform
        dataset[category] = enc.fit_transform(dataset[category])

    return dataset
