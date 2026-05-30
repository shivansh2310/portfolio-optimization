from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change(fill_method=None).dropna(how="all")


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna(how="all")


def rolling_returns(returns: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    return (1 + returns).rolling(window).apply(np.prod, raw=True) - 1


def rolling_volatility(returns: pd.DataFrame, window: int = 21, annualize: bool = True) -> pd.DataFrame:
    vol = returns.rolling(window).std()
    return vol * np.sqrt(TRADING_DAYS) if annualize else vol


def realized_volatility(returns: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    return np.sqrt((returns**2).rolling(window).sum() * TRADING_DAYS / window)


def drawdowns(prices_or_equity: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    cumulative_max = prices_or_equity.cummax()
    return prices_or_equity / cumulative_max - 1


def moving_averages(prices: pd.DataFrame, windows: tuple[int, ...] = (20, 50, 200)) -> dict[int, pd.DataFrame]:
    return {window: prices.rolling(window).mean() for window in windows}


def momentum(prices: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    return prices.pct_change(window, fill_method=None)


def covariance_matrix(returns: pd.DataFrame, annualize: bool = True) -> pd.DataFrame:
    cov = returns.cov()
    return cov * TRADING_DAYS if annualize else cov


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.corr()


def rolling_correlation(returns: pd.DataFrame, window: int = 63) -> dict[pd.Timestamp, pd.DataFrame]:
    return {
        date: returns.loc[:date].tail(window).corr()
        for date in returns.index[window - 1 :]
    }
