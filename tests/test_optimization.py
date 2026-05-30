import numpy as np
import pandas as pd

from src.optimization import clustered_minimum_variance_weights, equal_weight, minimum_variance_weights


def test_equal_weight_sums_to_one():
    weights = equal_weight(["A", "B", "C"])
    assert np.isclose(weights.sum(), 1)
    assert len(weights) == 3


def test_minimum_variance_weights_long_only_sum_to_one():
    covariance = pd.DataFrame(
        [[0.10, 0.02], [0.02, 0.20]],
        index=["A", "B"],
        columns=["A", "B"],
    )
    weights = minimum_variance_weights(covariance)
    assert np.isclose(weights.sum(), 1)
    assert (weights >= 0).all()


def test_clustered_minimum_variance_allocates_all_assets():
    covariance = pd.DataFrame(
        [[0.10, 0.01, 0.00], [0.01, 0.20, 0.02], [0.00, 0.02, 0.30]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    clusters = pd.Series([0, 0, 1], index=["A", "B", "C"])
    weights = clustered_minimum_variance_weights(covariance, clusters)
    assert np.isclose(weights.sum(), 1)
    assert set(weights.index) == {"A", "B", "C"}
