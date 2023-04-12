# cyclic-boosting

This package contains the implementation of the machine learning algorithm Cyclic Boosting, which is described in [Cyclic Boosting - an explainable supervised machine learning algorithm](https://arxiv.org/abs/2002.03425) and [Demand Forecasting of Individual Probability Density Functions with Machine Learning](https://arxiv.org/abs/2009.07052).

## Documentation

The documentation can be found [here](https://cyclic-boosting.readthedocs.io/en/latest/).

## Quickstart

```
pip install cyclic-boosting
```

```python
from cyclic_boosting.pipelines import pipeline_CBPoissonRegressor
CB_est = pipeline_CBPoissonRegressor()
CB_est.fit(X_train, y)
yhat = CB_est.predict(X_test)
```

## Usage

It can be used in a [scikit-learn](https://scikit-learn.org/stable/)-like fashion, combining a binning method (e.g., [BinNumberTransformer](https://github.com/Blue-Yonder-OSS/cyclic-boosting/blob/main/cyclic_boosting/binning/bin_number_transformer.py)) with a Cyclic Boosting estimator (find all estimators in the [init](https://github.com/Blue-Yonder-OSS/cyclic-boosting/blob/main/cyclic_boosting/__init__.py)). Usage examples can be found in the [integration tests](https://github.com/Blue-Yonder-OSS/cyclic-boosting/blob/main/tests/test_integration.py).
