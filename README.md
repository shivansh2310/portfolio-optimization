# Regime-Aware Robust Portfolio Optimization

## Project Overview

This project aims to develop a quantitative portfolio construction framework that addresses one of the most important challenges in portfolio optimization: estimation error.

Traditional Mean-Variance Optimization (MVO) is highly sensitive to errors in expected returns and covariance estimation, often leading to unstable portfolio weights and poor out-of-sample performance.

The proposed framework combines:

1. Covariance Denoising
2. Covariance Shrinkage
3. Asset Clustering
4. Market Regime Detection
5. Walk-Forward Portfolio Backtesting

to construct robust portfolios that maintain attractive risk-adjusted returns while reducing volatility, drawdowns, and estimation risk.

The final system should provide a complete quantitative research workflow from raw market data to portfolio construction and backtesting.

---

# Research Objective

The primary objective is:

> Can portfolio performance be improved by combining market regime detection, covariance denoising, clustering, and robust validation techniques?

The project should compare traditional MVO against progressively enhanced portfolio construction methods.

---

# Core Research Components

## Module 1: Data Collection

### Assets

The framework should support:

* S&P 500 equities
* ETFs
* Sector ETFs
* Crypto assets (optional)

Initial testing universe:

* AAPL
* NVDA
* TSLA
* XOM
* JPM
* LLY
* REGN

Future versions should support 30–100 assets.

### Data Source

Preferred:

* Yahoo Finance (yfinance)

Data Required:

* Open
* High
* Low
* Close
* Adjusted Close
* Volume

---

## Module 2: Feature Engineering

Generate:

### Return Features

* Daily Returns
* Log Returns
* Rolling Returns

### Risk Features

* Rolling Volatility
* Realized Volatility
* Drawdowns

### Trend Features

* Moving Averages
* Momentum Indicators

### Correlation Features

* Rolling Correlation Matrix
* Covariance Matrix

---

## Module 3: Market Regime Detection

### Objective

Identify hidden market regimes.

Examples:

* Bull Market
* Bear Market
* High Volatility
* Low Volatility

### Methods

#### Baseline

Markov Regime Switching Model

Libraries:

* statsmodels

#### Future Extensions

* Hidden Markov Models
* Bayesian Regime Detection
* Gaussian Mixture Models

### Outputs

For each date:

* Regime Label
* Regime Probability

---

## Module 4: Covariance Denoising

### Objective

Reduce noise in covariance estimation.

### Methods

#### Ledoit-Wolf Shrinkage

Libraries:

* sklearn.covariance

#### Oracle Approximating Shrinkage (OAS)

Libraries:

* sklearn.covariance

#### Constant Residual Eigenvalue Method (CREM)

Implementation required.

Steps:

1. Eigenvalue decomposition
2. Noise threshold selection
3. Eigenvalue adjustment
4. Covariance reconstruction

### Outputs

* Raw Covariance Matrix
* Denoised Covariance Matrix

---

## Module 5: Asset Clustering

### Objective

Group similar assets before optimization.

### Methods

#### Hierarchical Clustering

Preferred

#### K-Means Clustering

Baseline

#### Hierarchical Risk Parity (HRP)

Target implementation

### Outputs

* Cluster Assignments
* Dendrogram
* Correlation Heatmap

---

## Module 6: Portfolio Optimization

### Baseline

Mean Variance Optimization

Objective:

Minimize:

wᵀΣw

Subject To:

* Sum of weights = 1
* Optional long-only constraint
* Optional leverage constraint

### Enhanced Models

1. MVO + Shrinkage
2. MVO + Denoising
3. MVO + Clustering
4. MVO + Denoising + Clustering
5. Regime-Aware Portfolio Optimization

---

## Module 7: Backtesting Framework

### Validation Methods

#### Walk-Forward Validation

Train -> Test -> Roll Forward

#### K-Fold Cross Validation

For comparison

#### Combinatorial Purged Cross Validation (CPCV)

Preferred robust validation method

### Backtest Outputs

For each strategy:

* Portfolio Returns
* Portfolio Volatility
* Sharpe Ratio
* Sortino Ratio
* Maximum Drawdown
* Calmar Ratio
* Turnover

---

## Module 8: Performance Evaluation

### Statistical Metrics

* Mean Return
* Standard Deviation
* Skewness
* Kurtosis

### Risk Metrics

* Sharpe Ratio
* Sortino Ratio
* Maximum Drawdown
* Expected Shortfall (CVaR)

### Portfolio Metrics

* CAGR
* Annualized Return
* Annualized Volatility

---

# Visualizations

The project should generate:

## Regime Analysis

* Price + Regime Overlay
* Regime Probability Plot

## Correlation Analysis

* Correlation Heatmap
* Cluster Dendrogram

## Portfolio Analysis

* Efficient Frontier
* Asset Weights

## Performance Analysis

* Equity Curve
* Rolling Sharpe Ratio
* Drawdown Curve

---

# Repository Structure

```text
regime-aware-portfolio-optimization/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   ├── regime_detection.ipynb
│   ├── portfolio_optimization.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── regime_detection.py
│   ├── denoising.py
│   ├── clustering.py
│   ├── optimization.py
│   ├── backtesting.py
│   ├── metrics.py
│   ├── visualization.py
│
├── plots/
│
├── results/
│
├── tests/
│
├── requirements.txt
│
├── README.md
│
└── main.py
```

---

# Technology Stack

### Programming

* Python

### Data Analysis

* Pandas
* NumPy

### Financial Data

* yfinance

### Machine Learning

* scikit-learn

### Statistical Modeling

* statsmodels
* scipy

### Optimization

* cvxpy
* PyPortfolioOpt

### Visualization

* matplotlib
* plotly
* seaborn

---

# Final Deliverables

The completed project should provide:

1. Reproducible portfolio construction workflow
2. Market regime detection engine
3. Covariance denoising framework
4. Asset clustering framework
5. Portfolio optimization engine
6. Robust backtesting system
7. Professional GitHub repository
8. Research-style report
9. Resume-ready quantitative finance project

---

# Expected Resume Description

Developed a regime-aware portfolio optimization framework combining covariance denoising, clustering, and market regime detection techniques to improve portfolio robustness under non-stationary market conditions.

Implemented Markov Regime Switching models, Ledoit-Wolf/OAS shrinkage estimators, CREM denoising, and walk-forward backtesting to evaluate risk-adjusted performance using Sharpe ratio, volatility, and maximum drawdown metrics.
