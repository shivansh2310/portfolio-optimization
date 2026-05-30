from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf, OAS


def _as_dataframe(matrix: np.ndarray, index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(matrix, index=index, columns=index)


def ledoit_wolf_covariance(returns: pd.DataFrame, annualize: bool = True) -> pd.DataFrame:
    estimator = LedoitWolf().fit(returns.dropna().values)
    cov = estimator.covariance_
    if annualize:
        cov = cov * 252
    return _as_dataframe(cov, returns.columns)


def oas_covariance(returns: pd.DataFrame, annualize: bool = True) -> pd.DataFrame:
    estimator = OAS().fit(returns.dropna().values)
    cov = estimator.covariance_
    if annualize:
        cov = cov * 252
    return _as_dataframe(cov, returns.columns)


def crem_denoised_covariance(
    returns: pd.DataFrame,
    q: float | None = None,
    annualize: bool = True,
) -> pd.DataFrame:
    """Denoise covariance via constant residual eigenvalue method.

    Eigenvalues below the Marchenko-Pastur upper noise edge are replaced by
    their average while signal eigenvalues are preserved.
    """
    clean = returns.dropna()
    if clean.shape[0] <= clean.shape[1]:
        raise ValueError("CREM needs more observations than assets.")

    cov = clean.cov().values
    corr = clean.corr().values
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    q = q or clean.shape[0] / clean.shape[1]
    lambda_plus = (1 + np.sqrt(1 / q)) ** 2
    signal_count = int(np.sum(eigenvalues > lambda_plus))

    adjusted = eigenvalues.copy()
    if signal_count < len(eigenvalues):
        adjusted[signal_count:] = adjusted[signal_count:].mean()

    denoised_corr = eigenvectors @ np.diag(adjusted) @ eigenvectors.T
    denoised_corr = _cov_to_corr(denoised_corr)
    std = np.sqrt(np.diag(cov))
    denoised_cov = denoised_corr * np.outer(std, std)
    if annualize:
        denoised_cov = denoised_cov * 252
    return _as_dataframe(denoised_cov, returns.columns)


def _cov_to_corr(matrix: np.ndarray) -> np.ndarray:
    diag = np.sqrt(np.diag(matrix))
    corr = matrix / np.outer(diag, diag)
    corr[corr < -1] = -1
    corr[corr > 1] = 1
    np.fill_diagonal(corr, 1)
    return corr


def nearest_positive_semidefinite(covariance: pd.DataFrame, epsilon: float = 1e-8) -> pd.DataFrame:
    values, vectors = np.linalg.eigh(covariance.values)
    values = np.maximum(values, epsilon)
    repaired = vectors @ np.diag(values) @ vectors.T
    return _as_dataframe(repaired, covariance.index)
