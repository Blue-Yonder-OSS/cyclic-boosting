from cyclic_boosting.nbinom import gammaln
from math import log, exp


def test_gammaln():
    assert abs(gammaln(5.0) - log(24.0)) < 1e-6
    assert abs(gammaln(4.0) - log(6.0)) < 1e-6
    assert abs(gammaln(1.0) - log(1.0)) < 1e-6
    assert abs(gammaln(2.0) - log(1.0)) < 1e-6
