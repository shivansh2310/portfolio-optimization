from __future__ import annotations

import pandas as pd


def clean_prices(prices: pd.DataFrame, min_history: float = 0.9) -> pd.DataFrame:
    """Forward-fill prices and drop assets with too much missing history."""
    if not 0 < min_history <= 1:
        raise ValueError("min_history must be in (0, 1].")
    enough_data = prices.notna().mean() >= min_history
    cleaned = prices.loc[:, enough_data].ffill().dropna()
    if cleaned.empty:
        raise ValueError("No price data remains after cleaning.")
    return cleaned


def align_returns_and_features(*frames: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    """Return frames aligned to the shared non-null date index."""
    common_index = frames[0].dropna().index
    for frame in frames[1:]:
        common_index = common_index.intersection(frame.dropna().index)
    return tuple(frame.loc[common_index] for frame in frames)
