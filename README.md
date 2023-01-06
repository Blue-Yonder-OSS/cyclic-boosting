cyclic-boosting
===============

This package contains the implementation of the Machine Learning algorithm Cyclic Boosting, which is described in [Cyclic Boosting - an explainable supervised machine learning algorithm](https://arxiv.org/abs/2002.03425) and [Demand Forecasting of Individual Probability Density Functions with Machine Learning](https://arxiv.org/abs/2009.07052).

Documentation
-------------

The documentation of this package can be found [here](https://cyclic-boosting.readthedocs.io/en/latest/).

Usage
-----

It can be used in a [scikit-learn](https://scikit-learn.org/stable/)-like fashion. You need to combine a binner (e.g., [BinNumberTransformer](https://github.com/Blue-Yonder-OSS/cyclic-boosting/blob/main/cyclic_boosting/binning/bin_number_transformer.py)) with an estimator (find all estimators in the [init](https://github.com/Blue-Yonder-OSS/cyclic-boosting/blob/main/cyclic_boosting/__init__.py)). A usage example can be found in the [integration tests](https://github.com/Blue-Yonder-OSS/cyclic-boosting/blob/main/tests/test_integration.py). A more detailed example, including additional helper functionality, can be found [here](https://github.com/Blue-Yonder-OSS/cyclic-boosting-example).
