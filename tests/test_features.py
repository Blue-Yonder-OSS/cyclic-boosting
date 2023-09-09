from cyclic_boosting import CBPoissonRegressor
import numpy as np
import pytest
from typing import List


@pytest.fixture
def expected_feature_importances() -> List[float]:
    return [
        0.3001108076903618,
        0.011901588088793208,
        0.004428680307559594,
        0.006827823766347157,
        0.0069619726813944,
        0.025250911051351754,
        0.027094558535123603,
        0.015477424141162243,
        0.005558373302600022,
        0.016814438906556692,
        0.16315611923004458,
        0.41641730229870494,
    ]


def test_poisson_regressor_feature_importance(get_inputs, expected_feature_importances):
    X, y, feature_prop, feature_groups = get_inputs
    est = CBPoissonRegressor(
        feature_groups=feature_groups,
        feature_properties=feature_prop,
    )
    est.fit(X, y)
    norm_feature_importances = est.get_feature_importances()

    assert [ele for ele in norm_feature_importances.values()] == expected_feature_importances
    np.testing.assert_almost_equal(sum(norm_feature_importances.values()), 1.0, 3)
