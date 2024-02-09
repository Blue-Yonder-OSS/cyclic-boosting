from cyclic_boosting.tornado.core import datamodule, module
from cyclic_boosting.tornado.trainer import trainer
import shutil


def test_tornado(prepare_data):
    X, y = prepare_data
    df, target = X.iloc[:200].copy(), y[:200].copy()
    df.loc[:, "sales"] = target

    data_deliverer = datamodule.TornadoDataModule(df)
    manager = module.TornadoModule(max_iter=5)
    predictor = trainer.Tornado(data_deliverer, manager)
    predictor.fit(target="sales", criterion="COD", verbose=False)
    _ = predictor.predict(df)
    _ = predictor.predict_proba(df, output="proba")
    shutil.rmtree('./models')


def test_forward_trainer(prepare_data):
    X, y = prepare_data
    df, target = X.iloc[:200].copy(), y[:200].copy()
    df.loc[:, "sales"] = target

    data_deliverer = datamodule.TornadoDataModule(df)
    manager = module.ForwardSelectionModule(max_iter=5)
    predictor = trainer.ForwardTrainer(data_deliverer, manager)
    predictor.fit(target="sales", criterion="COD", verbose=False)
    _ = predictor.predict(df)
    _ = predictor.predict_proba(df, output="proba")
    shutil.rmtree('./models')


def test_qpd_forward_trainer(prepare_data):
    X, y = prepare_data
    df, target = X.iloc[:200].copy(), y[:200].copy()
    df.loc[:, "sales"] = target

    data_deliverer = datamodule.TornadoDataModule(df)
    manager = module.PriorPredForwardSelectionModule(max_iter=10, model="additive")
    predictor = trainer.QPDForwardTrainer(data_deliverer, manager)
    predictor.fit(target="sales", criterion="PINBALL", verbose=False)
    _ = predictor.predict(df)
    _ = predictor.predict_proba(df, output="proba")
    shutil.rmtree('./models')
