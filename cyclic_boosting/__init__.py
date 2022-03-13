"""The ``cyclic_boosting`` subpackage is the home to the cyclic-boosting family
of machine learning algorithms.

If you are looking for conceptional explanations of the cyclic boosting
algorithm, you can find them in two flavours in our documentation, there is a
`statistical/mathematical introduction <concepts/cyclicboosting>`_  and a
`higher-level introduction <concepts/cyclic_boosting_family>`_


This is the API reference of the cyclic boosting algorithm, you'll find the
API docs for:

Convenience estimators (also do binning)

- :class:`~.CBGammaPoissonDensity`
- :class:`~.CBClassifier`

Core estimators (no Binning, etc.)

- :class:`~.CBWidthRegressor`
- :class:`~.CBCoreLocationRegressor`
- :class:`~.CBPoissonRegressor`
- :class:`~.CBFixedVarianceRegressor`
- :class:`~.CBCoreClassifier`
"""

from __future__ import division, print_function

from cyclic_boosting.base import CyclicBoostingBase
from cyclic_boosting.classification import CBCoreClassifier
from cyclic_boosting.GBSregression import CBCoreGBSRegressor
from cyclic_boosting.location import CBCoreLocationRegressor

from cyclic_boosting.regression import (  # noqa # isort:skip
    CBFixedVarianceRegressor,
    CBPoissonRegressor,
)
from cyclic_boosting.price import CBExponential  # noqa #isort:skip

__all__ = [
    "CBFixedVarianceRegressor",
    "CBClassifier",
    "CBCoreClassifier",
    "CBCoreLocationRegressor",
    "CBPoissonRegressor",
    "CyclicBoostingBase",
    "CBExponential",
    "CBCoreGBSRegressor",
]

__version__ = "1.2.0"
