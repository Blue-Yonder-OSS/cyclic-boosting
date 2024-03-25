from cyclic_boosting.tornado.core import datamodule, manager
from cyclic_boosting.tornado.trainer import tornado
import shutil


def test_interaction_search_model(prepare_data):
    X, y = prepare_data
    df, target = X.iloc[:200].copy(), y[:200].copy()
    df.loc[:, "sales"] = target

    data_deliverer = datamodule.TornadoDataModule(df)
    manager_ = manager.TornadoManager(max_iter=5)
    predictor = tornado.InteractionSearchModel(data_deliverer, manager_)
    predictor.fit(target="sales", criterion="COD", verbose=False)
    _ = predictor.predict(df)
    _ = predictor.predict_proba(df, output="proba")
    shutil.rmtree("./models")


def test_forward_selection_model(prepare_data):
    X, y = prepare_data
    df, target = X.iloc[:200].copy(), y[:200].copy()
    df.loc[:, "sales"] = target

    data_deliverer = datamodule.TornadoDataModule(df)
    manager_ = manager.ForwardSelectionManager(max_iter=5)
    predictor = tornado.ForwardSelectionModel(data_deliverer, manager_)
    predictor.fit(target="sales", criterion="COD", verbose=False)
    _ = predictor.predict(df)
    _ = predictor.predict_proba(df, output="proba")
    shutil.rmtree("./models")


def test_prior_pred_interaction_search_model(prepare_data):
    X, y = prepare_data
    df, target = X.iloc[:200].copy(), y[:200].copy()
    df.loc[:, "sales"] = target

    data_deliverer = datamodule.TornadoDataModule(df)
    manager_ = manager.PriorPredForwardSelectionManager(max_iter=5)
    predictor = tornado.PriorPredInteractionSearchModel(data_deliverer, manager_)
    predictor.fit(target="sales", criterion="COD", verbose=False)
    _ = predictor.predict(df)
    _ = predictor.predict_proba(df, output="proba")
    shutil.rmtree("./models")


def test_qpd_interaction_search_model(prepare_data):
    X, y = prepare_data
    df, target = X.iloc[:2000].copy(), y[:2000].copy()  # need a dataset included enough data size to train
    df.loc[:, "sales"] = target

    data_deliverer = datamodule.TornadoDataModule(df)
    manager_ = manager.PriorPredForwardSelectionManager(
        max_iter=10,
        dist="qpd",
    )
    predictor = tornado.QPDInteractionSearchModel(data_deliverer, manager_, bound="S", lower=0)
    predictor.fit(target="sales", criterion="PINBALL", verbose=False)
    _ = predictor.predict(df)
    _ = predictor.predict_proba(df, output="proba")
    shutil.rmtree("./models")
