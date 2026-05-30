from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def annualized_return(returns: pd.Series) -> float:
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    cumulative = (1 + returns).prod()
    years = len(returns) / TRADING_DAYS
    return cumulative ** (1 / years) - 1


def annualized_volatility(returns: pd.Series) -> float:
    return returns.dropna().std() * np.sqrt(TRADING_DAYS)


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    excess = returns.dropna() - risk_free_rate / TRADING_DAYS
    vol = excess.std()
    if vol == 0 or np.isnan(vol):
        return np.nan
    return excess.mean() / vol * np.sqrt(TRADING_DAYS)


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    excess = returns.dropna() - risk_free_rate / TRADING_DAYS
    downside = excess[excess < 0].std()
    if downside == 0 or np.isnan(downside):
        return np.nan
    return excess.mean() / downside * np.sqrt(TRADING_DAYS)


def max_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns.dropna()).cumprod()
    drawdown = equity / equity.cummax() - 1
    return drawdown.min()


def calmar_ratio(returns: pd.Series) -> float:
    drawdown = abs(max_drawdown(returns))
    if drawdown == 0 or np.isnan(drawdown):
        return np.nan
    return annualized_return(returns) / drawdown


def expected_shortfall(returns: pd.Series, alpha: float = 0.05) -> float:
    clean = returns.dropna()
    if clean.empty:
        return np.nan
    cutoff = clean.quantile(alpha)
    return clean[clean <= cutoff].mean()


def turnover(weights: pd.DataFrame) -> pd.Series:
    return weights.diff().abs().sum(axis=1).fillna(0)


def performance_summary(returns: pd.Series, weights: pd.DataFrame | None = None) -> pd.Series:
    summary = pd.Series(
        {
            "mean_return": returns.mean(),
            "std_dev": returns.std(),
            "skewness": returns.skew(),
            "kurtosis": returns.kurtosis(),
            "annualized_return": annualized_return(returns),
            "annualized_volatility": annualized_volatility(returns),
            "sharpe_ratio": sharpe_ratio(returns),
            "sortino_ratio": sortino_ratio(returns),
            "max_drawdown": max_drawdown(returns),
            "calmar_ratio": calmar_ratio(returns),
            "expected_shortfall": expected_shortfall(returns),
        }
    )
    if weights is not None:
        summary["average_turnover"] = turnover(weights).mean()
    return summary
