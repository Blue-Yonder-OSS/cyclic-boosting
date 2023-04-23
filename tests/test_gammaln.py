from cyclic_boosting.nbinom import gammaln, GAMMALN_CONSTANT
from math import log, exp
from scipy import special
import numpy as np


def test_gammaln():
    data = np.linspace(1e-04, 2 - 1e-02, 100)
    tol = (GAMMALN_CONSTANT + 1) * np.exp(-GAMMALN_CONSTANT)
    for x in data:
        assert abs(gammaln(x) - special.gammaln(x)) < tol, f"for x={x}"

    data = np.linspace(10,1000, 10000)
    tol = 1e-01
    for x in data:
        assert abs(gammaln(x) - special.gammaln(x)) < tol, f"for x={x}"