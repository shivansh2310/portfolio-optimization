from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering, KMeans


def correlation_distance(correlation: pd.DataFrame) -> pd.DataFrame:
    distance = np.sqrt((1 - correlation.clip(-1, 1)) / 2)
    return pd.DataFrame(distance, index=correlation.index, columns=correlation.columns)


def hierarchical_clusters(returns: pd.DataFrame, n_clusters: int = 3) -> pd.Series:
    corr = returns.corr()
    dist = correlation_distance(corr)
    model = AgglomerativeClustering(
        n_clusters=min(n_clusters, returns.shape[1]),
        metric="precomputed",
        linkage="average",
    )
    labels = model.fit_predict(dist)
    return pd.Series(labels, index=returns.columns, name="cluster")


def kmeans_clusters(returns: pd.DataFrame, n_clusters: int = 3, random_state: int = 42) -> pd.Series:
    features = returns.T.fillna(0)
    model = KMeans(n_clusters=min(n_clusters, returns.shape[1]), n_init="auto", random_state=random_state)
    labels = model.fit_predict(features)
    return pd.Series(labels, index=returns.columns, name="cluster")


def hierarchical_linkage(returns: pd.DataFrame, method: str = "average") -> np.ndarray:
    dist = correlation_distance(returns.corr())
    condensed = squareform(dist.values, checks=False)
    return linkage(condensed, method=method)


def cluster_representatives(returns: pd.DataFrame, clusters: pd.Series) -> list[str]:
    """Pick the lowest volatility asset from each cluster."""
    vol = returns.std()
    representatives: list[str] = []
    for _, members in clusters.groupby(clusters):
        representatives.append(vol.loc[members.index].idxmin())
    return representatives
