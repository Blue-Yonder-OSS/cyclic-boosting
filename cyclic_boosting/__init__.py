"""This package contains the Cyclic Boosting family of machine learning
algorithms.

If you are looking for conceptional explanations of the Cyclic Boosting
algorithm, you might have a look at the two papers
https://arxiv.org/abs/2002.03425 and https://arxiv.org/abs/2009.07052.

API reference of the different Cyclic Boosting methods:

Multiplicative Regression

- :class:`~.CBFixedVarianceRegressor`
- :class:`~.CBPoissonRegressor`
- :class:`~.CBExponential`

Additive Regression

- :class:`~.CBCoreLocationRegressor`
- :class:`~.CBCoreLinearPoissonRegressor`

PDF Prediction

- :class:`~.CBNBinomC`

Classification

- :class:`~.CBCoreClassifier`

Background Subtraction

- :class:`~.CBCoreGBSRegressor`
"""

from __future__ import division, print_function

from cyclic_boosting.base import CyclicBoostingBase
from cyclic_boosting.regression import CBFixedVarianceRegressor,\
    CBPoissonRegressor
from cyclic_boosting.price import CBExponential
from cyclic_boosting.location import CBCoreLocationRegressor,\
    CBCoreLinearPoissonRegressor
from cyclic_boosting.nbinom import CBNBinomC
from cyclic_boosting.classification import CBCoreClassifier
from cyclic_boosting.GBSregression import CBCoreGBSRegressor

__all__ = [
    "CyclicBoostingBase",
    "CBFixedVarianceRegressor",
    "CBPoissonRegressor",
    "CBExponential",
    "CBCoreLocationRegressor",
    "CBCoreLinearPoissonRegressor",
    "CBNBinomC",
    "CBCoreClassifier",
    "CBCoreGBSRegressor",
]

__version__ = "1.0"
