# Install and Quickstart

User Installation
-----------------

```
pip install cyclic-boosting
```

Quickstart
----------

```python
from cyclic_boosting.pipelines import pipeline_CBPoissonRegressor
CB_est = pipeline_CBPoissonRegressor()
CB_est.fit(X_train, y)
yhat = CB_est.predict(X_test)
```

Development and Tests
---------------------

```
git clone https://github.com/Blue-Yonder-OSS/cyclic-boosting.git
pip install -e .
pip install -r requirements-dev.txt
pytest tests
```