from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def volatility_regimes(
    market_returns: pd.Series,
    window: int = 21,
    n_regimes: int = 2,
) -> pd.DataFrame:
    """Label regimes by rolling volatility quantiles.

    This is a deterministic baseline that keeps the pipeline usable even when
    a Markov model fails to converge on a small sample.
    """
    vol = market_returns.rolling(window).std() * np.sqrt(252)
    labels = pd.qcut(vol.rank(method="first"), q=n_regimes, labels=False, duplicates="drop")
    labels = labels.astype("float").ffill().fillna(0).astype(int)
    probability = pd.Series(1.0, index=market_returns.index, name="regime_probability")
    return pd.DataFrame({"regime": labels, "regime_probability": probability})


def markov_regime_switching(
    market_returns: pd.Series,
    k_regimes: int = 2,
    switching_variance: bool = True,
) -> pd.DataFrame:
    """Fit a Markov switching model and return most likely regimes.

    Falls back to volatility regimes if statsmodels cannot converge.
    """
    try:
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

        clean = market_returns.dropna() * 100
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = MarkovRegression(clean, k_regimes=k_regimes, trend="c", switching_variance=switching_variance)
            result = model.fit(disp=False, maxiter=200)

        probabilities = result.smoothed_marginal_probabilities
        regime = probabilities.idxmax(axis=1).astype(int)
        regime_probability = probabilities.max(axis=1)
        output = pd.DataFrame({"regime": regime, "regime_probability": regime_probability})
        return output.reindex(market_returns.index).ffill().dropna()
    except Exception:
        return volatility_regimes(market_returns, n_regimes=k_regimes)


def regime_summary(returns: pd.Series, regimes: pd.Series) -> pd.DataFrame:
    joined = pd.concat([returns.rename("return"), regimes.rename("regime")], axis=1).dropna()
    return joined.groupby("regime")["return"].agg(["count", "mean", "std", "min", "max"])
