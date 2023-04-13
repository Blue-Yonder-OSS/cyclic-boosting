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

## Linting and Formatting

We use [`black`](https://github.com/psf/black) for file formatting and
[`ruff`](https://github.com/charliermarsh/ruff) for linting. Both tools execute
really fast. Every time a commit is pushed to the repository, all files are
checked with these tools. It is therefore recommend that you check all files
beforehand. A configuration for [`pre-commit`](https://pre-commit.com/) is
included in this repository. To activate automatic checking each time you
commit, please add a commit hook to your local git repository:

```sh
poetry run pre-commit install
```

You cannot commit then if not all files comply (but you can circumvent
this limitation by using the `--no-verify` or `-n` switch when committing).

Alternatively, you can run the following each time before pushing:

```sh
poetry run pre-commit run --all-files 
```
