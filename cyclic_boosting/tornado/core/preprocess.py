# Add some comments
#
#
#

import abc
import six
import pandas as pd


def dayofweek(dataset) -> pd.DataFrame:
    dataset["dayofweek"] = dataset["date"].dayofweek

    return dataset