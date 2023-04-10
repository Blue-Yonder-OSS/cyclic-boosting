# Install and Quickstart

## User Installation

```
pip install cyclic-boosting
```

## Quickstart

```python
from cyclic_boosting.pipelines import pipeline_CBPoissonRegressor
CB_est = pipeline_CBPoissonRegressor()
CB_est.fit(X_train, y)
yhat = CB_est.predict(X_test)
```

## Development and Tests

For developing, please run `poetry install`. This installs all package and
development dependencies. Run `poetry install --with docs` to install the
dependencies to build the docs. Installing with `--with jupyter` adds a
[jupyter lab](https://jupyter.org/)
to the virtual environment.

Don't forget to either activate the env with `poetry shell` or prepend your
commands (e.g., `poetry run black .`) The command `poetry env info` provides you
with information about the environment including the path to the python
interpreter which might be required to set up your IDE.

Example:

```shell
git clone https://github.com/Blue-Yonder-OSS/cyclic-boosting.git
cd cyclic-boosting
poetry install --with jupyter,docs
# either
poetry run pytest tests
# or
poetry shell
pytest tests
```
