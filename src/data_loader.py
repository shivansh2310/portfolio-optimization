from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


DEFAULT_TICKERS = ["AAPL", "NVDA", "TSLA", "XOM", "JPM", "LLY", "REGN"]


def download_prices(
    tickers: list[str] | None = None,
    start: str = "2015-01-01",
    end: str | None = None,
    auto_adjust: bool = False,
    cache_path: str | Path | None = None,
) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance.

    Returns a dataframe with a MultiIndex column shaped as (field, ticker).
    """
    tickers = tickers or DEFAULT_TICKERS
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        group_by="column",
        progress=False,
        threads=True,
    )
    if data.empty:
        raise ValueError("No market data was returned by yfinance.")

    data = data.sort_index()
    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(cache_path)
    return data


def load_cached_prices(path: str | Path) -> pd.DataFrame:
    """Load a CSV previously written by download_prices."""
    return pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)


def get_adjusted_close(data: pd.DataFrame) -> pd.DataFrame:
    """Extract adjusted close prices from Yahoo-style data."""
    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.get_level_values(0):
            prices = data["Adj Close"]
        elif "Close" in data.columns.get_level_values(0):
            prices = data["Close"]
        else:
            raise KeyError("Expected 'Adj Close' or 'Close' in price data.")
    else:
        prices = data
    return prices.dropna(axis=1, how="all").ffill().dropna(how="all")
