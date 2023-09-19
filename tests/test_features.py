from cyclic_boosting import CBPoissonRegressor
import numpy as np
import pytest
from typing import List


@pytest.fixture
def expected_feature_importances() -> List[float]:
    return [
        0.08183693583617015,
        0.14191802307396523,
        0.12016395453139928,
        0.23511743026016937,
        0.10313172776022547,
        0.030753319720274865,
        0.09212591146822456,
        0.19495269734957096,
    ]


def test_poisson_regressor_feature_importance(prepare_data, features, feature_properties, expected_feature_importances):
    X, y = prepare_data
    est = CBPoissonRegressor(
        feature_groups=features,
        feature_properties=feature_properties,
    )
    est.fit(X, y)
    norm_feature_importances = est.get_feature_importances()

    assert [ele[0].feature_group for ele in norm_feature_importances.keys()] == [
        ("dayofweek",),
        ("L_ID",),
        ("PG_ID_3",),
        ("P_ID",),
        ("PROMOTION_TYPE",),
        ("price_ratio",),
        ("dayofyear",),
        ("P_ID", "L_ID"),
    ]

    for ind, f_imp in enumerate(norm_feature_importances.values()):
        np.testing.assert_almost_equal(f_imp, expected_feature_importances[ind], 4)
    np.testing.assert_almost_equal(sum(norm_feature_importances.values()), 1.0, 3)
