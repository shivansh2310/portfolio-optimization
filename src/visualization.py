from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_correlation_heatmap(correlation: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 7))
    sns.heatmap(correlation, cmap="vlag", center=0, annot=False, square=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_equity_curve(strategy_returns: dict[str, pd.Series], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    for name, returns in strategy_returns.items():
        equity = (1 + returns).cumprod()
        plt.plot(equity.index, equity.values, label=name)
    plt.legend()
    plt.ylabel("Growth of $1")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_drawdown_curve(strategy_returns: dict[str, pd.Series], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    for name, returns in strategy_returns.items():
        equity = (1 + returns).cumprod()
        drawdown = equity / equity.cummax() - 1
        plt.plot(drawdown.index, drawdown.values, label=name)
    plt.legend()
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
