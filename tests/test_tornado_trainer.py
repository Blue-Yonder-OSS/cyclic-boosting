from cyclic_boosting.tornado.core import datamodule, module
from cyclic_boosting.tornado.trainer import trainer


def test_tornado(prepare_data):
    X, y = prepare_data
    df, target = X.iloc[:200].copy(), y[:200].copy()
    df.loc[:, "sales"] = target
    df.to_csv("temp.csv", index=False)

    data_deliverer = datamodule.TornadoDataModule("temp.csv")
    manager = module.TornadoModule(max_iter=5)
    predictor = trainer.Tornado(data_deliverer, manager)
    predictor.fit(target="sales", log_policy="COD", verbose=False)
    _ = predictor.predict(df)
    _ = predictor.predict_proba(df, output="proba")


def test_forward_trainer(prepare_data):
    X, y = prepare_data
    df, target = X.iloc[:200].copy(), y[:200].copy()
    df.loc[:, "sales"] = target
    df.to_csv("temp.csv", index=False)

    data_deliverer = datamodule.TornadoDataModule("temp.csv")
    manager = module.ForwardSelectionModule(max_iter=5)
    predictor = trainer.ForwardTrainer(data_deliverer, manager)
    predictor.fit(target="sales", log_policy="COD", verbose=False)
    _ = predictor.predict(df)
    _ = predictor.predict_proba(df, output="proba")


def test_qpd_forward_trainer(prepare_data):
    X, y = prepare_data
    df, target = X.iloc[:200].copy(), y[:200].copy()
    df.loc[:, "sales"] = target
    df.to_csv("temp.csv", index=False)

    data_deliverer = datamodule.TornadoDataModule("temp.csv")
    manager = module.PriorPredForwardSelectionModule(max_iter=10, model="additive")
    predictor = trainer.QPDForwardTrainer(data_deliverer, manager)
    predictor.fit(target="sales", log_policy="PINBALL", verbose=False)
    _ = predictor.predict(df)
    _ = predictor.predict_proba(df, output="proba")
