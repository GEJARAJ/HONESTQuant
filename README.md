# ⚖️ HonestQuant: Defensible ML Trading Research & Audit Platform

> **"Instead of claiming our AI predicts the market, we built a platform whose selling point is testing ML strategies rigorously and showing where they fail."**

---

## 🎯 Defensible Thesis & Scientific Framing

Most machine learning trading projects fail when reviewed by seasoned examiners and quant hedge funds because they claim unrealistic returns derived from:
1. **Lookahead Bias & Data Leakage:** Fitting scalers, feature transformations, or indicator lags across the entire dataset.
2. **Data Snooping & Selection Bias:** Backtesting 1,000 parameter combinations and reporting only the highest Sharpe ratio without deflation.
3. **Regime Overfitting:** Training on a multi-year bull market (e.g. 2020–2021) and collapsing during inflation/rate hikes (2022).
4. **Ignoring Execution Friction:** Generating frequent rebalances that generate phantom profits before 5–10 bps transaction fees and market impact slippage.

**HonestQuant** turns the paradigm upside down: it is an **academic and institutional audit engine** built to uncover the structural breakdown points of quantitative ML models and demonstrate where ML legitimately provides statistical edge (volatility forecasting) vs where it struggles (directional returns).

---

## 🏗️ Core Architecture & Modules

```
honest_quant_platform/
├── app.py                      # Main Glassmorphism Quant Studio UI (Streamlit + Plotly)
├── run_app.bat                 # 1-Click Windows Launcher
├── modules/
│   ├── data_pipeline.py        # OHLCV fetching, caching, cleaning & leakage-free feature engineering
│   ├── models.py               # Baselines, ML (Random Forest, XGBoost), LSTM, HMM Regime Detector, GARCH vs ML Vol
│   ├── backtester.py           # Walk-Forward Validation, Deflated Sharpe, PBO, Regime Autopsies
│   ├── risk_manager.py         # Dynamic Vol Targeting, Fractional Kelly, Trailing Stops, Max DD Kill-Switch
│   ├── paper_trading.py        # Live simulated forward paper trading ledger & feature attributions
│   └── visualizer.py           # Interactive Plotly dark quant charts & underwater drawdown profiles
└── data_cache/                 # High-speed parquet cache & paper trading state persistence
```

---

## 🚀 Key Modules & Capabilities

### 1. Data Pipeline (`data_pipeline.py`)
- Real-time and historical OHLCV data retrieval from Yahoo Finance with disk caching.
- Fallback synthetic geometric Brownian motion generator with stochastic volatility jumps for stress-testing crisis conditions.
- **20+ Leakage-Free Features:** Log returns, rolling realized volatilities ($5d, 10d, 21d, 63d$), RSI(14), MACD normalized divergence, Bollinger Bands (%B & Bandwidth), Normalized ATR, Volume Z-scores, Rolling Skewness & Kurtosis, SMA distance measures.
- **Next-Period Shifting:** All target vectors are shifted by $t+1$ to strictly avoid lookahead bias.

### 2. Models Tournament (`models.py`)
- **Baselines:** Buy & Hold, Simple Moving Average Crossover (20/50 SMA), RSI Mean Reversion.
- **Machine Learning:** Random Forest & XGBoost (Gradient Boosted Decision Trees).
- **Deep Learning:** PyTorch LSTM recurrent sequence network.
- **Market Regime Detector:** Unsupervised Gaussian Hidden Markov Model (HMM) classifying market environments into Low Volatility (Bull), Moderate Volatility, and High Volatility (Turbulence/Crisis).
- **Volatility Tournament:** Classical Econometric GARCH(1,1) vs Gradient Boosted Volatility Forecaster evaluated on Out-of-Sample RMSE, MAE, and QLIKE loss.

### 3. Walk-Forward Backtester & Overfitting Audit (`backtester.py`)
- **Walk-Forward Cross-Validation (WFCV):** Rolling or expanding windows (e.g. 500 bars In-Sample train, 125 bars Out-of-Sample test).
- **In-Sample vs Out-of-Sample Degradation (Sharpe Haircut):** Quantifies overfit decay percentage.
- **Deflated Sharpe Ratio (DSR) & Probability of Backtest Overfitting (PBO):** Bailey & López de Prado formula adjusting for non-normality, skewness, kurtosis, and multiple testing trials.
- **Regime Autopsy Breakdown:** Segregates PnL across HMM states to pinpoint model collapse during high-volatility crashes.
- **Monte Carlo 1-Year Forward Resampling:** 300 bootstrapped equity trajectories with 90% and 50% confidence envelopes.

### 4. Risk Management & Execution Friction (`risk_manager.py`)
- **Position Sizing:** Volatility Targeting (15% target), Fractional Kelly Criterion, or Fixed allocation.
- **Execution Frictions:** Configurable transaction cost (e.g. 10 bps) and dynamic slippage (e.g. 5 bps).
- **Trailing Stop-Loss & Take-Profit:** Real-time trade peak tracking with exit triggers.
- **Max Drawdown Kill-Switch (Circuit Breaker):** Automatically flattens positions and enters a cooldown period if portfolio drawdown exceeds threshold (e.g. 12%).

### 5. Live Paper Trading & Decision Explainer (`paper_trading.py`)
- Forward step-by-step simulated paper trading engine with virtual cash and share holdings.
- Complete order ledger tracking executed trades, slippage costs, and real-time model decision attribution.

---

## ⚡ Quickstart & Running the Platform

1. Double-click **`run_app.bat`** (on Windows) or run in terminal:
```bash
streamlit run app.py
```
2. Open your browser to `http://localhost:8501`.
3. Select an asset (e.g. `SPY`, `AAPL`, `NVDA`, `BTC-USD`), configure risk thresholds, and click **"Run Honest ML Audit"**!
