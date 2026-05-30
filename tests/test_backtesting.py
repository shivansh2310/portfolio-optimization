import numpy as np
import pandas as pd

from src.backtesting import WalkForwardConfig, walk_forward_backtest
from src.optimization import equal_weight


def test_walk_forward_backtest_returns_series_and_weights():
    dates = pd.bdate_range("2020-01-01", periods=80)
    rng = np.random.default_rng(42)
    returns = pd.DataFrame(rng.normal(0.001, 0.01, size=(80, 3)), index=dates, columns=["A", "B", "C"])

    portfolio_returns, weights = walk_forward_backtest(
        returns,
        lambda train: equal_weight(train.columns),
        WalkForwardConfig(train_window=40, test_window=10, step_size=10),
    )

    assert len(portfolio_returns) == 40
    assert weights.shape == (40, 3)
    assert np.allclose(weights.sum(axis=1), 1)
