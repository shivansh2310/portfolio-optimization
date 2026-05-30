import numpy as np
import pandas as pd

from src.metrics import max_drawdown, performance_summary, sharpe_ratio


def test_sharpe_ratio_positive_for_positive_returns():
    returns = pd.Series([0.01, 0.02, -0.005, 0.015, 0.004])
    assert sharpe_ratio(returns) > 0


def test_max_drawdown_detects_peak_to_trough_loss():
    returns = pd.Series([0.1, -0.2, 0.05])
    assert np.isclose(max_drawdown(returns), -0.2)


def test_performance_summary_contains_expected_metrics():
    returns = pd.Series([0.01, -0.005, 0.007, 0.002])
    summary = performance_summary(returns)
    assert "sharpe_ratio" in summary.index
    assert "max_drawdown" in summary.index
