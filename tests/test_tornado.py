from cyclic_boosting.tornado.core import datamodule, manager
from cyclic_boosting.tornado.trainer import tornado
from sklearn.model_selection import train_test_split
import shutil


def test_interaction_search_model(prepare_data):
    X, y = prepare_data
    df, target = X.iloc[:500].copy(), y[:500].copy()
    df.loc[:, "sales"] = target
    df_train, df_test = train_test_split(df, test_size=0.1, shuffle=False)

    data_deliverer = datamodule.TornadoDataModule(df_train)
    train, validation = data_deliverer.generate_trainset(test_size=0.2, seed=0, target="sales", is_time_series=True)
    test = data_deliverer.generate_testset(df_test)
    manager_ = manager.TornadoManager(max_iter=5)
    predictor = tornado.InteractionSearchModel(manager_)
    predictor.fit("sales", train, validation, criterion="COD", verbose=False)
    _ = predictor.predict(test)
    _ = predictor.predict_proba(test, output="proba")
    shutil.rmtree("./models")


def test_forward_selection_model(prepare_data):
    X, y = prepare_data
    df, target = X.iloc[:500].copy(), y[:500].copy()
    df.loc[:, "sales"] = target
    df_train, df_test = train_test_split(df, test_size=0.2, shuffle=False)

    data_deliverer = datamodule.TornadoDataModule(df_train)
    train, validation = data_deliverer.generate_trainset(test_size=0.2, seed=0, target="sales", is_time_series=True)
    test = data_deliverer.generate_testset(df_test)

    manager_ = manager.ForwardSelectionManager(max_iter=5)
    predictor = tornado.ForwardSelectionModel(manager_)
    predictor.fit("sales", train, validation, criterion="COD", verbose=False)
    _ = predictor.predict(test)
    _ = predictor.predict_proba(test, output="proba")
    shutil.rmtree("./models")


def test_prior_pred_interaction_search_model(prepare_data):
    X, y = prepare_data
    df, target = X.iloc[:500].copy(), y[:500].copy()
    df.loc[:, "sales"] = target
    df_train, df_test = train_test_split(df, test_size=0.2, shuffle=False)

    data_deliverer = datamodule.TornadoDataModule(df_train)
    train, validation = data_deliverer.generate_trainset(test_size=0.2, seed=0, target="sales", is_time_series=True)
    test = data_deliverer.generate_testset(df_test)

    manager_ = manager.PriorPredForwardSelectionManager(max_iter=5)
    predictor = tornado.PriorPredInteractionSearchModel(manager_)
    predictor.fit("sales", train, validation, criterion="COD", verbose=False)
    _ = predictor.predict(test)
    _ = predictor.predict_proba(test, output="proba")
    shutil.rmtree("./models")


def test_qpd_interaction_search_model(prepare_data):
    X, y = prepare_data
    df, target = X.iloc[:3000].copy(), y[:3000].copy()  # need a dataset included enough data size to train
    df.loc[:, "sales"] = target
    df_train, df_test = train_test_split(df, test_size=0.2, shuffle=False)

    data_deliverer = datamodule.TornadoDataModule(df_train)
    train, validation = data_deliverer.generate_trainset(test_size=0.2, seed=0, target="sales", is_time_series=True)
    test = data_deliverer.generate_testset(df_test)

    manager_ = manager.PriorPredForwardSelectionManager(
        max_iter=10,
        dist="qpd",
    )
    predictor = tornado.QPDInteractionSearchModel(manager_, bound="S", lower=0)
    predictor.fit("sales", train, validation, criterion="PINBALL", verbose=False)
    _ = predictor.predict(test)
    _ = predictor.predict_proba(test, output="proba")
    shutil.rmtree("./models")
