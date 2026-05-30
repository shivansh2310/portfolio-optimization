from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.backtesting import WalkForwardConfig, compare_strategies
from src.clustering import hierarchical_clusters
from src.data_loader import DEFAULT_TICKERS, download_prices, get_adjusted_close, load_cached_prices
from src.denoising import crem_denoised_covariance, ledoit_wolf_covariance, oas_covariance
from src.feature_engineering import correlation_matrix, simple_returns
from src.metrics import performance_summary
from src.optimization import clustered_minimum_variance_weights, equal_weight, minimum_variance_weights
from src.preprocessing import clean_prices
from src.regime_detection import markov_regime_switching, regime_summary
from src.visualization import save_correlation_heatmap, save_drawdown_curve, save_equity_curve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run regime-aware robust portfolio optimization research pipeline.")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--cache", default="data/raw/prices.csv")
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--train-window", type=int, default=252)
    parser.add_argument("--test-window", type=int, default=21)
    parser.add_argument("--step-size", type=int, default=21)
    return parser.parse_args()


def build_strategies() -> dict:
    return {
        "equal_weight": lambda train: equal_weight(train.columns),
        "mvo_sample_cov": lambda train: minimum_variance_weights(train.cov() * 252),
        "mvo_ledoit_wolf": lambda train: minimum_variance_weights(ledoit_wolf_covariance(train)),
        "mvo_oas": lambda train: minimum_variance_weights(oas_covariance(train)),
        "mvo_crem": lambda train: minimum_variance_weights(crem_denoised_covariance(train)),
        "clustered_mvo": lambda train: clustered_minimum_variance_weights(
            ledoit_wolf_covariance(train),
            hierarchical_clusters(train, n_clusters=3),
        ),
    }


def main() -> None:
    args = parse_args()
    cache_path = Path(args.cache)

    if args.use_cache and cache_path.exists():
        raw = load_cached_prices(cache_path)
    else:
        raw = download_prices(args.tickers, start=args.start, end=args.end, cache_path=cache_path)

    prices = clean_prices(get_adjusted_close(raw))
    returns = simple_returns(prices).dropna()
    if len(returns) < args.train_window + args.test_window:
        raise ValueError("Not enough observations for the requested walk-forward windows.")

    market_proxy = returns.mean(axis=1)
    regimes = markov_regime_switching(market_proxy)
    regimes.to_csv("results/regimes.csv")
    regime_summary(market_proxy, regimes["regime"]).to_csv("results/regime_summary.csv")

    config = WalkForwardConfig(
        train_window=args.train_window,
        test_window=args.test_window,
        step_size=args.step_size,
    )
    results = compare_strategies(returns, build_strategies(), config)

    strategy_returns = {name: result[0] for name, result in results.items()}
    summaries = {
        name: performance_summary(portfolio_returns, weights)
        for name, (portfolio_returns, weights) in results.items()
    }
    summary_frame = pd.DataFrame(summaries).T.sort_values("sharpe_ratio", ascending=False)
    summary_frame.to_csv("results/performance_summary.csv")

    combined_returns = pd.DataFrame(strategy_returns)
    combined_returns.to_csv("results/strategy_returns.csv")

    latest_weights = pd.DataFrame({name: weights.iloc[-1] for name, (_, weights) in results.items()}).T
    latest_weights.to_csv("results/latest_weights.csv")

    save_correlation_heatmap(correlation_matrix(returns), "plots/correlation_heatmap.png")
    save_equity_curve(strategy_returns, "plots/equity_curve.png")
    save_drawdown_curve(strategy_returns, "plots/drawdown_curve.png")

    print(summary_frame.round(4))
    print("\nSaved outputs to results/ and plots/.")


if __name__ == "__main__":
    main()
