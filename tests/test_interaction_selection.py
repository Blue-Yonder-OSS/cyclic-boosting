import pandas as pd
from pandas.testing import assert_frame_equal

from cyclic_boosting.interaction_selection import create_interactions, build_binned_interaction_features
from cyclic_boosting import flags


def test_create_interactions():
    inputs = ["A", "B", "C", "D", "E"]
    interactions = create_interactions(inputs, 3)
    expected = [
        ("A", "B"),
        ("A", "C"),
        ("A", "D"),
        ("A", "E"),
        ("B", "C"),
        ("B", "D"),
        ("B", "E"),
        ("C", "D"),
        ("C", "E"),
        ("D", "E"),
        ("A", "B", "C"),
        ("A", "B", "D"),
        ("A", "B", "E"),
        ("A", "C", "D"),
        ("A", "C", "E"),
        ("A", "D", "E"),
        ("B", "C", "D"),
        ("B", "C", "E"),
        ("B", "D", "E"),
        ("C", "D", "E"),
    ]
    assert interactions == expected


def test_build_binned_interaction_features():
    X = pd.DataFrame({"A": [3, 2, 1], "B": [4, 5, 6], "C": [9, 8, 7]})
    interaction_terms = [("A", "B"), ("B", "C")]
    feature_properties = {"A": flags.IS_UNORDERED, "B": flags.IS_UNORDERED, "C": flags.IS_UNORDERED}
    interaction_features = build_binned_interaction_features(X, interaction_terms, feature_properties)
    expected = pd.DataFrame({"('A', 'B')": [6, 4, 2], "('B', 'C')": [2, 4, 6]})
    interaction_features.rename(columns=lambda x: str(x), inplace=True)
    assert_frame_equal(interaction_features, expected)
