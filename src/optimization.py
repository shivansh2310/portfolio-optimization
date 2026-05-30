from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def equal_weight(assets: list[str] | pd.Index) -> pd.Series:
    assets = list(assets)
    return pd.Series(1 / len(assets), index=assets, name="weight")


def minimum_variance_weights(
    covariance: pd.DataFrame,
    long_only: bool = True,
    max_weight: float | None = None,
) -> pd.Series:
    assets = covariance.columns
    n_assets = len(assets)
    initial = np.repeat(1 / n_assets, n_assets)
    bounds = [(0, max_weight or 1) for _ in assets] if long_only else [(-1, 1) for _ in assets]
    constraints = ({"type": "eq", "fun": lambda weights: np.sum(weights) - 1},)

    result = minimize(
        lambda weights: float(weights.T @ covariance.values @ weights),
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    weights = pd.Series(result.x, index=assets, name="weight")
    weights[np.abs(weights) < 1e-10] = 0
    return weights / weights.sum()


def mean_variance_weights(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_aversion: float = 5.0,
    long_only: bool = True,
    max_weight: float | None = None,
) -> pd.Series:
    assets = covariance.columns
    mu = expected_returns.reindex(assets).fillna(0).values
    n_assets = len(assets)
    initial = np.repeat(1 / n_assets, n_assets)
    bounds = [(0, max_weight or 1) for _ in assets] if long_only else [(-1, 1) for _ in assets]
    constraints = ({"type": "eq", "fun": lambda weights: np.sum(weights) - 1},)

    def objective(weights: np.ndarray) -> float:
        variance = weights.T @ covariance.values @ weights
        expected = weights @ mu
        return float(0.5 * risk_aversion * variance - expected)

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")
    return pd.Series(result.x, index=assets, name="weight")


def clustered_minimum_variance_weights(
    covariance: pd.DataFrame,
    clusters: pd.Series,
    long_only: bool = True,
) -> pd.Series:
    """Allocate equally across clusters, then minimum variance within each cluster."""
    weights = pd.Series(0.0, index=covariance.columns, name="weight")
    cluster_ids = sorted(clusters.unique())
    cluster_budget = 1 / len(cluster_ids)
    for cluster_id in cluster_ids:
        members = clusters[clusters == cluster_id].index.intersection(covariance.columns)
        if len(members) == 1:
            weights.loc[members[0]] = cluster_budget
            continue
        sub_weights = minimum_variance_weights(covariance.loc[members, members], long_only=long_only)
        weights.loc[members] = sub_weights * cluster_budget
    return weights / weights.sum()
