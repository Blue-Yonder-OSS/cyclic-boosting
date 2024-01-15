"""This package contains the Cyclic Boosting family of machine learning
algorithms.

If you are looking for conceptional explanations of the Cyclic Boosting
algorithm, you might have a look at the two papers
https://arxiv.org/abs/2002.03425 and https://arxiv.org/abs/2009.07052.

API reference of the different Cyclic Boosting methods:

Multiplicative Regression

- :class:`~.CBPoissonRegressor`
- :class:`~.CBNBinomRegressor`
- :class:`~.CBExponential`
- :class:`~.CBMultiplicativeQuantileRegressor`
- :class:`~.CBMultiplicativeGenericCRegressor`

Additive Regression

- :class:`~.CBLocationRegressor`
- :class:`~.CBLocPoissonRegressor`
- :class:`~.CBAdditiveQuantileRegressor`
- :class:`~.CBAdditiveGenericCRegressor`

PDF Prediction

- :class:`~.CBNBinomC`

Classification

- :class:`~.CBClassifier`
- :class:`~.CBGenericClassifier`

Background Subtraction

- :class:`~.CBGBSRegressor`
"""

from __future__ import division, print_function


from cyclic_boosting.base import CyclicBoostingBase
from cyclic_boosting.regression import CBNBinomRegressor, CBPoissonRegressor
from cyclic_boosting.price import CBExponential
from cyclic_boosting.location import CBLocationRegressor, CBLocPoissonRegressor
from cyclic_boosting.nbinom import CBNBinomC
from cyclic_boosting.classification import CBClassifier
from cyclic_boosting.GBSregression import CBGBSRegressor
from cyclic_boosting.generic_loss import (
    CBMultiplicativeQuantileRegressor,
    CBAdditiveQuantileRegressor,
    CBMultiplicativeGenericCRegressor,
    CBAdditiveGenericCRegressor,
    CBGenericClassifier,
)
from cyclic_boosting.pipelines import (
    pipeline_CBPoissonRegressor,
    pipeline_CBNBinomRegressor,
    pipeline_CBClassifier,
    pipeline_CBLocationRegressor,
    pipeline_CBExponential,
    pipeline_CBLocPoissonRegressor,
    pipeline_CBNBinomC,
    pipeline_CBGBSRegressor,
    pipeline_CBMultiplicativeQuantileRegressor,
    pipeline_CBAdditiveQuantileRegressor,
    pipeline_CBMultiplicativeGenericCRegressor,
    pipeline_CBAdditiveGenericCRegressor,
    pipeline_CBGenericClassifier,
)

__all__ = [
    "CyclicBoostingBase",
    "CBPoissonRegressor",
    "CBNBinomRegressor",
    "CBExponential",
    "CBLocationRegressor",
    "CBLocPoissonRegressor",
    "CBNBinomC",
    "CBClassifier",
    "CBGBSRegressor",
    "CBMultiplicativeQuantileRegressor",
    "CBAdditiveQuantileRegressor",
    "CBMultiplicativeGenericCRegressor",
    "CBAdditiveGenericCRegressor",
    "CBGenericClassifier",
    "pipeline_CBPoissonRegressor",
    "pipeline_CBNBinomRegressor",
    "pipeline_CBClassifier",
    "pipeline_CBLocationRegressor",
    "pipeline_CBExponential",
    "pipeline_CBLocPoissonRegressor",
    "pipeline_CBNBinomC",
    "pipeline_CBGBSRegressor",
    "pipeline_CBMultiplicativeQuantileRegressor",
    "pipeline_CBAdditiveQuantileRegressor",
    "pipeline_CBMultiplicativeGenericCRegressor",
    "pipeline_CBAdditiveGenericCRegressor",
    "pipeline_CBGenericClassifier",
]

__version__ = "1.0"
