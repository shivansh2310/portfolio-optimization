from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd


WeightFunction = Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class WalkForwardConfig:
    train_window: int = 252
    test_window: int = 21
    step_size: int = 21


def walk_forward_backtest(
    returns: pd.DataFrame,
    weight_function: WeightFunction,
    config: WalkForwardConfig | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Run train/test rolling portfolio backtest.

    weight_function receives the in-sample returns and returns asset weights.
    """
    config = config or WalkForwardConfig()
    portfolio_returns: list[pd.Series] = []
    weights_by_date: list[pd.Series] = []

    start = config.train_window
    while start < len(returns):
        train = returns.iloc[start - config.train_window : start]
        test = returns.iloc[start : start + config.test_window]
        if test.empty:
            break

        weights = weight_function(train).reindex(returns.columns).fillna(0)
        weights = weights / weights.sum()
        out_of_sample = test @ weights
        portfolio_returns.append(out_of_sample)

        for date in test.index:
            dated_weights = weights.copy()
            dated_weights.name = date
            weights_by_date.append(dated_weights)

        start += config.step_size

    if not portfolio_returns:
        raise ValueError("Not enough data for the requested walk-forward configuration.")

    returns_series = pd.concat(portfolio_returns).sort_index()
    returns_series.name = "portfolio_return"
    weights_frame = pd.DataFrame(weights_by_date).sort_index()
    return returns_series, weights_frame


def compare_strategies(
    returns: pd.DataFrame,
    strategies: dict[str, WeightFunction],
    config: WalkForwardConfig | None = None,
) -> dict[str, tuple[pd.Series, pd.DataFrame]]:
    return {
        name: walk_forward_backtest(returns, strategy, config)
        for name, strategy in strategies.items()
    }
