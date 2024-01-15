import pandas as pd
import numpy as np

import warnings


# 無視したい警告を指定する
warnings.filterwarnings("ignore", message="RuntimeWarning")

path = "./data.csv"
df = pd.read_csv(path)


def drop_LAMBDA(df):
    df = df.drop(columns="LAMBDA")
    return df


def convert_datatype(df, col):
    if df[col].dtype == np.float64:
        df = df.astype({col: np.int64})
    elif df[col].dtype == np.int64:
        df = df.astype({col: np.float64})
    return df


df_test = df.copy()
df_test = drop_LAMBDA(df_test)
df_test = convert_datatype(df_test, col="SCHOOL_HOLIDAY")
df_test.drop(["PG_ID_1", "PG_ID_2", "PG_ID_3"], inplace=True, axis=1)
df_test.to_csv("./data_test.csv", index=False)

from cyclic_boosting.tornado import Generator, Manager, Tornado

data_deliverler = Generator.TornadoDataModule("data_test.csv")
manager = Manager.ForwardSelectionModule(dist="poisson")
tornado_model = Tornado.ForwardTrainer(data_deliverler, manager)
# manager = Manager.BFForwardSelectionModule(dist="qpd")
# tornado_model = Tornado.QPDForwardTrainer(data_deliverler, manager)
# tornado_model.fit(target="sales", log_policy="PINBALL", verbose=False)
tornado_model.fit(target="sales", log_policy="COD", verbose=False)
yhat = tornado_model.predict(df_test)
# yhat = tornado_model.predict(df_test, output="ppf", ix=0.5)
print(yhat)


# class hoge:
#     def __init__(self) -> None:
#         self.hoge = "hogehoge"

#     def get_params(self) -> dict:
#         class_vars = {}
#         for k, v in self.__dict__.items():
#             if not callable(v) and not k.startswith("__"):
#                 class_vars[k] = v
#         return class_vars


# h = hoge()
# print(h.get_params())
