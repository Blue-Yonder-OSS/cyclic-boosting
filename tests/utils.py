import numpy as np

from cyclic_boosting.binning import BinNumberTransformer
from cyclic_boosting.plots import plot_analysis

from typing import Optional


def plot_CB(filename: str, plobs: list, binner: Optional[BinNumberTransformer] = None):
    for i, p in enumerate(plobs):
        plot_analysis(plot_observer=p, file_obj=filename + "_{}".format(i), use_tightlayout=False, binners=[binner])


def costs_mad(prediction: np.ndarray, y: np.ndarray, weights):
    return np.nanmean(np.abs(y - prediction))


def costs_mse(prediction: np.ndarray, y: np.ndarray, weights):
    return np.nanmean(np.square(y - prediction))
