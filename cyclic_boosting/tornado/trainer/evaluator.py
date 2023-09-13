import sys
import numpy as np


class EvaluatorBase():
    def __init__(self) -> None:
        self.result = {}

    def mean_deviation(self, y, yhat) -> float:
        return np.nanmean(y - yhat)

    def mean_absolute_deviation(self, y, yhat) -> float:
        return np.nanmean(np.abs(np.nanmean(y) - yhat))

    def mean_square_error(self, y, yhat) -> float:
        return np.nanmean(np.square(y - yhat))

    def mean_absolute_error(self, y, yhat) -> float:
        return np.nanmean(np.abs(y - yhat))

    def median_absolute_error(self, y, yhat) -> float:
        return np.nanmedian(np.abs(y - yhat))

    def weighted_absolute_percentage_error(self, y, yhat) -> float:
        return np.nansum(np.abs(y - yhat) * y) / np.nansum(y)

    def symmetric_mean_absolute_percentage_error(self, y, yhat) -> float:
        return 100. * np.nanmean(np.abs(y - yhat) / ((np.abs(y) + np.abs(yhat)) / 2.))

    def mean_y(self, y, _) -> float:
        return np.nanmean(y)
    
    def coefficient_of_determination(self, y, yhat) -> float:
        #決定係数
        return 1 - (np.nansum((y - yhat)**2) / np.nansum((y - np.nanmean(y))**2))
    
    def F(self, y, yhat, est):
        n = len(y)
        k = len(est['CB'].feature_groups)
        residual_variance = np.nansum((y - yhat)**2) / (n - k - 1)
        regression_variance = np.nansum((yhat - np.nanmean(y))**2) / k
        return regression_variance / residual_variance


    def eval_all(self, y, yhat, est,verbose=True, digit=5):

        funcs = {
            "MD": self.mean_deviation,
            "MAD": self.mean_absolute_deviation,
            "MSE": self.mean_square_error,
            "MAE": self.mean_absolute_error,
            "MedAE": self.median_absolute_error,
            "WMAPE": self.weighted_absolute_percentage_error,
            "SMAPE": self.symmetric_mean_absolute_percentage_error,
            "COD": self.coefficient_of_determination,
            "F": self.F,
            # "MEAN-Y": self.mean_y,
        }

        # eval
        for metrics, func in funcs.items():
            if metrics not in self.result.keys():
                self.result[metrics] = []

            if not metrics == 'F':
                result = func(y, yhat)
            else:
                result = func(y, yhat, est)
            self.result[metrics].append(result)

            if verbose:
                print(f"[{metrics}]: {round(result, digit)}")

    def mean_metrics(self, digit=5):
        for metrics, values in self.result.items():
            print(f"[{metrics}]: {round(np.mean(values), digit)}")

    def clear(self):
        self.result = {}
