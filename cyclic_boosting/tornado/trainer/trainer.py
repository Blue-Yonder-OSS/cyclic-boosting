import pandas as pd


class Trainer():
    def __init__(DataModule, TornadoModule):
        self.data_deliveler = DataModule
        self.manager = TornadoModule


    def fit(self):
        dataset = self.data_deliveler.generate()
        self.manager.manage()
        estimater = self.manager.build()
        estimater.fit()