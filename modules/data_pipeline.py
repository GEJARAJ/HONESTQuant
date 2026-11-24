"""
HonestQuant: Honest ML Trading Research & Audit Platform
Data Pipeline Module
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional

try:
    import yfinance as yf
except ImportError:
    yf = None


class DataPipeline:
    """
    Robust financial data fetching, cleaning, feature engineering,
    and non-stationarity / leakage prevention pipeline.
    """

    def __init__(self, cache_dir: str = "data_cache"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def fetch_data(
        self,
        ticker: str = "SPY",
        start_date: str = "2018-01-01",
        end_date: str = "2024-01-01",
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from Yahoo Finance with disk caching or synthetic fallback.
        """
        ticker_clean = ticker.upper().strip()
        cache_file = os.path.join(self.cache_dir, f"{ticker_clean}_{start_date}_{end_date}.parquet")

        if use_cache and os.path.exists(cache_file):
            try:
                df = pd.read_parquet(cache_file)
                if not df.empty:
                    return df
            except Exception:
                pass

        df = None
        if yf is not None:
            try:
                raw_df = yf.download(
                    ticker_clean,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    auto_adjust=True
                )
                if raw_df is not None and not raw_df.empty:
                    # Handle MultiIndex columns if returned by newer yfinance
                    if isinstance(raw_df.columns, pd.MultiIndex):
                        raw_df.columns = [c[0] for c in raw_df.columns]
                    df = raw_df.copy()
            except Exception as e:
                print(f"[Warning] Failed to download {ticker_clean} from yfinance: {e}")

        # Fallback to realistic synthetic geometric Brownian motion with stochastic volatility & jumps
        if df is None or df.empty or len(df) < 50:
            df = self._generate_synthetic_market_data(start_date, end_date, ticker_clean)

        # Standardize columns
        df = self._clean_raw_data(df)

        # Save cache
        try:
            df.to_parquet(cache_file)
        except Exception:
            pass

        return df

    def _generate_synthetic_market_data(
        self,
        start_date: str,
        end_date: str,
        ticker: str = "SYNTH"
    ) -> pd.DataFrame:
        """
        Generate realistic synthetic financial time series with volatility clustering,
        fat tails, and regime shifts.
        """
        dates = pd.date_range(start=start_date, end=end_date, freq="B")
        n = len(dates)
        if n < 100:
            dates = pd.date_range(start="2018-01-01", end="2024-01-01", freq="B")
            n = len(dates)

        np.random.seed(abs(hash(ticker)) % 10000)

        # Heston / GARCH-like stochastic volatility regime simulation
        vol = np.zeros(n)
        returns = np.zeros(n)
        vol[0] = 0.012  # Daily ~19% annualized

        # Regime 1: Bull low vol, Regime 2: Crash/Turbulence, Regime 3: Recovery
        current_regime = 0
        regime_transition = np.array([
            [0.985, 0.015],
            [0.05, 0.95]
        ])

        for t in range(1, n):
            current_regime = np.random.choice([0, 1], p=regime_transition[current_regime])
            if current_regime == 0:
                mean_drift = 0.0004
                vol[t] = 0.008 + 0.7 * vol[t-1] + 0.1 * abs(returns[t-1])
            else:
                mean_drift = -0.0008
                vol[t] = 0.025 + 0.75 * vol[t-1] + 0.15 * abs(returns[t-1])
            
            # Student-t innovation for fat tails
            shock = np.random.standard_t(df=5) * (vol[t] / np.sqrt(5/3))
            returns[t] = mean_drift + shock

        initial_price = 100.0 if "SPY" in ticker or "SYNTH" in ticker else 150.0
        price_series = initial_price * np.exp(np.cumsum(returns))

        # Generate Open, High, Low, Close, Volume
        noise_high = np.random.uniform(1.002, 1.018, n) * (1 + vol * 10)
        noise_low = np.random.uniform(0.982, 0.998, n) * (1 - vol * 10)
        noise_open = np.random.uniform(0.995, 1.005, n)

        close = price_series
        open_p = np.roll(close, 1) * noise_open
        open_p[0] = initial_price
        high = np.maximum(np.maximum(open_p, close) * noise_high, np.maximum(open_p, close) + 0.05)
        low = np.minimum(np.minimum(open_p, close) * noise_low, np.minimum(open_p, close) - 0.05)
        volume = (np.random.lognormal(14.5, 0.5, n) * (1 + 25 * vol)).astype(int)

        df = pd.DataFrame({
            "Open": open_p,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume
        }, index=dates)
        return df

    def _clean_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data: enforce standard columns, sort index, handle missing or zero values.
        """
        df = df.copy()
        # Rename standard columns if needed
        cols = {c: c.capitalize() for c in df.columns}
        df.rename(columns=cols, inplace=True)

        req_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in req_cols:
            if col not in df.columns:
                if "Adj Close" in df.columns and col == "Close":
                    df["Close"] = df["Adj Close"]
                else:
                    df[col] = df["Close"] if "Close" in df.columns else 100.0

        df = df[req_cols].dropna()
        df = df[~df.index.duplicated(keep="first")]
        df.sort_index(inplace=True)

        # Fix zero volumes
        df["Volume"] = df["Volume"].replace(0, np.nan).bfill().ffill().fillna(100000)
        return df

    def engineer_features(
        self,
        df: pd.DataFrame,
        return_lags: list = [1, 2, 3, 5, 10],
        vol_windows: list = [5, 10, 21, 63],
        rsi_window: int = 14,
        target_horizon: int = 1
    ) -> Tuple[pd.DataFrame, list, str, str]:
        """
        Construct technically sound, stationary financial features with strictly NO lookahead bias.
        
        Outputs:
            df_feat: DataFrame with features and targets aligned
            feature_cols: list of engineered feature names
            target_col_dir: next-period directional target (1: Up, 0: Down)
            target_col_ret: next-period log return target
        """
        data = df.copy()

        # 1. Log Returns & Momentum
        data["log_ret"] = np.log(data["Close"] / data["Close"].shift(1))
        
        feature_cols = []

        # Lagged Returns
        for lag in return_lags:
            col = f"ret_lag_{lag}"
            data[col] = data["log_ret"].shift(lag)
            feature_cols.append(col)

        # 2. Rolling Realized Volatility (Annualized)
        for w in vol_windows:
            col = f"realized_vol_{w}d"
            data[col] = data["log_ret"].rolling(w).std() * np.sqrt(252)
            feature_cols.append(col)

        # 3. Relative Strength Index (RSI)
        delta = data["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=rsi_window, min_periods=rsi_window).mean()
        avg_loss = loss.rolling(window=rsi_window, min_periods=rsi_window).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        data["rsi_14"] = 100 - (100 / (1 + rs))
        feature_cols.append("rsi_14")

        # 4. MACD & Signal Line Divergence
        ema_12 = data["Close"].ewm(span=12, adjust=False).mean()
        ema_26 = data["Close"].ewm(span=26, adjust=False).mean()
        data["macd"] = (ema_12 - ema_26) / data["Close"]  # Normalized
        data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()
        data["macd_diff"] = data["macd"] - data["macd_signal"]
        feature_cols.extend(["macd", "macd_diff"])

        # 5. Bollinger Bands (%B and Bandwidth)
        sma_20 = data["Close"].rolling(20).mean()
        std_20 = data["Close"].rolling(20).std()
        upper_bb = sma_20 + 2 * std_20
        lower_bb = sma_20 - 2 * std_20
        data["bb_pct"] = (data["Close"] - lower_bb) / (upper_bb - lower_bb + 1e-9)
        data["bb_bandwidth"] = (upper_bb - lower_bb) / sma_20
        feature_cols.extend(["bb_pct", "bb_bandwidth"])

        # 6. Average True Range (ATR normalized)
        tr1 = data["High"] - data["Low"]
        tr2 = (data["High"] - data["Close"].shift(1)).abs()
        tr3 = (data["Low"] - data["Close"].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        data["natr_14"] = (tr.rolling(14).mean() / data["Close"]) * 100
        feature_cols.append("natr_14")

        # 7. Volume Features (Normalized Volume Z-Score & OBV Momentum)
        vol_mean = data["Volume"].rolling(20).mean()
        vol_std = data["Volume"].rolling(20).std() + 1e-5
        data["volume_zscore"] = (data["Volume"] - vol_mean) / vol_std
        feature_cols.append("volume_zscore")

        # 8. Rolling Skewness and Kurtosis (Tail Risk Indicators)
        data["rolling_skew_21d"] = data["log_ret"].rolling(21).skew()
        data["rolling_kurt_21d"] = data["log_ret"].rolling(21).kurt()
        feature_cols.extend(["rolling_skew_21d", "rolling_kurt_21d"])

        # 9. Moving Average Distances (Stationary trend measures)
        data["dist_sma_50"] = (data["Close"] - data["Close"].rolling(50).mean()) / data["Close"]
        data["dist_sma_200"] = (data["Close"] - data["Close"].rolling(200).mean()) / data["Close"]
        feature_cols.extend(["dist_sma_50", "dist_sma_200"])

        # 10. Targets (Forward Shifted by target_horizon)
        # Note: shifted negatively so row t contains target for t+horizon
        data["fwd_ret"] = np.log(data["Close"].shift(-target_horizon) / data["Close"])
        data["target_ret"] = data["fwd_ret"]
        data["target_dir"] = (data["fwd_ret"] > 0).astype(int)
        
        # Target for volatility forecasting: Realized 5-day future volatility
        data["target_vol_5d"] = (
            data["log_ret"].shift(-5).rolling(5).std() * np.sqrt(252)
        )

        # Drop NaN rows caused by rolling windows and forward shifts
        data_clean = data.dropna().copy()

        return data_clean, feature_cols, "target_dir", "target_ret"
