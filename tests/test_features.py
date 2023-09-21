from cyclic_boosting import CBPoissonRegressor
import numpy as np
import pytest
import pandas as pd
from typing import Dict, Tuple

from cyclic_boosting.regression import CBBaseRegressor


@pytest.fixture
def expected_feature_importances() -> Dict[str, float]:
    return {
        "dayofweek": 0.08183693583617015,
        "L_ID": 0.14191802307396523,
        "PG_ID_3": 0.12016395453139928,
        "P_ID": 0.23511743026016937,
        "PROMOTION_TYPE": 0.10313172776022547,
        "price_ratio": 0.030753319720274865,
        "dayofyear": 0.09212591146822456,
        "P_ID_L_ID": 0.19495269734957096,
    }


@pytest.fixture
def expected_feature_contributions() -> Dict[str, float]:
    return {
        "dayofweek": 1.0033225561393633,
        "L_ID": 0.9966915915554274,
        "PG_ID_3": 0.9962981313257777,
        "P_ID": 0.9581821452147931,
        "PROMOTION_TYPE": 0.9896018791652068,
        "price_ratio": 1.0,
        "dayofyear": 1.0506461325899688,
        "P_ID L_ID": 0.9140640045438535,
    }


@pytest.fixture(scope="session")
def estimator_data(prepare_data, features, feature_properties) -> Tuple[CBBaseRegressor, pd.DataFrame]:
    X, y = prepare_data
    est = CBPoissonRegressor(
        feature_groups=features,
        feature_properties=feature_properties,
    )
    est.fit(X, y)
    return est, X


def test_feature_importance(estimator_data, expected_feature_importances):
    estimator, _ = estimator_data
    norm_feature_importances = estimator.get_feature_importances()

    for feature_name, feature_importance in norm_feature_importances.items():
        assert feature_name in expected_feature_importances.keys()
        np.testing.assert_almost_equal(feature_importance, expected_feature_importances[feature_name], 4)
    np.testing.assert_almost_equal(sum(norm_feature_importances.values()), 1.0, 3)


def test_feature_contributions(estimator_data, expected_feature_contributions):
    estimator, X = estimator_data
    feature_contributions = estimator.get_feature_contributions(X)

    for feature_name, feature_contribution in feature_contributions.items():
        assert feature_name in expected_feature_contributions.keys()
        np.testing.assert_almost_equal(feature_contribution.mean(), expected_feature_contributions[feature_name], 3)
