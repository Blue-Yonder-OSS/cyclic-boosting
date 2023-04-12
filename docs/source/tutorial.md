# Tutorial

Cyclic Boosting can be used in a
[scikit-learn](https://scikit-learn.org/stable/)-like fashion.

Several examples can be found in the
[integration tests](https://github.com/Blue-Yonder-OSS/cyclic-boosting/blob/main/tests/test_integration.py).

A more detailed example, including additional helper functionality, can be
found [here](https://github.com/Blue-Yonder-OSS/cyclic-boosting-example).

For the simplest, default case, just do:
```python
from cyclic_boosting.pipelines import pipeline_CBPoissonRegressor

CB_est = pipeline_CBPoissonRegressor()
CB_est.fit(X_train, y)

yhat = CB_est.predict(X_test)
```

## Analysis Plots

To additionally create analysis plots of the training:
```python
from cyclic_boosting.pipelines import pipeline_CBPoissonRegressor
from cyclic_boosting import observers
from cyclic_boosting.plots import plot_analysis

def plot_CB(filename, plobs, binner):
    for i, p in enumerate(plobs):
        plot_analysis(
            plot_observer=p,
            file_obj=filename + "_{}".format(i),
            use_tightlayout=False,
            binners=[binner]
        )

plobs = [observers.PlottingObserver(iteration=-1)]
CB_est = pipeline_CBPoissonRegressor(observers=plobs)

CB_est.fit(X_train, y)
plot_CB('analysis_CB_iterlast', [CB_est[-1].observers[-1]], CB_est[-2])

yhat = CB_est.predict(X_test)
```

## Set Feature Properties
By setting feature properties/flags (all available ones can be found
[here](https://cyclic-boosting.readthedocs.io/en/latest/cyclic_boosting.html#module-cyclic_boosting.flags)),
you can also specify the treatment of individual features, e.g., as continuous
or categorical (including treatment of missing values):
```python
from cyclic_boosting.pipelines import pipeline_CBPoissonRegressor
from cyclic_boosting import flags

fp = {}
fp['feature1'] = flags.IS_UNORDERED
fp['feature1'] = flags.IS_CONTINUOUS | flags.HAS_MISSING | flags.MISSING_NOT_LEARNED

CB_est = pipeline_CBPoissonRegressor(feature_properties=fp)

CB_est.fit(X_train, y)

yhat = CB_est.predict(X_test)
```

## Set Features
You can also specify which columns to use as features, including interaction
terms (default is all available columns as individual features only):
```python
from cyclic_boosting.pipelines import pipeline_CBPoissonRegressor

features = [
    'feature1',
    'feature2',
    ('feature1', 'feature2')
]

CB_est = pipeline_CBPoissonRegressor(feature_groups=features)

CB_est.fit(X_train, y)

yhat = CB_est.predict(X_test)
```

## Manual Binning
Behind the scenes, Cyclic Boosting works by combining a binning method (e.g.,
[BinNumberTransformer](https://github.com/Blue-Yonder-OSS/cyclic-boosting/blob/main/cyclic_boosting/binning/bin_number_transformer.py))
with a Cyclic Boosting estimator (find all estimators
[here](https://github.com/Blue-Yonder-OSS/cyclic-boosting/blob/main/cyclic_boosting/__init__.py)).

If you want to use a different binning, e.g., a different number of bins
(default is 100), you can combine binners and estimators manually:
```python
from sklearn.pipeline import Pipeline
from cyclic_boosting import binning, CBPoissonRegressor

binner = binning.BinNumberTransformer(n_bins=50)
est = CBPoissonRegressor()
CB_est = Pipeline([("binning", binner), ("CB", est)])

CB_est.fit(X_train, y)

yhat = CB_est.predict(X_test)
```