"""
HonestQuant: Models Module
Contains Baselines, Machine Learning (RF, XGBoost), Deep Learning (LSTM),
Regime Detectors (Gaussian HMM), and Volatility Forecasters (GARCH vs ML).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
import warnings
warnings.filterwarnings("ignore")

# Machine learning & stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, mean_absolute_error

# XGBoost with fallback
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# HMM with fallback to Gaussian Mixture or K-Means
try:
    from hmmlearn.hmm import GaussianHMM
    HAS_HMM = True
except ImportError:
    HAS_HMM = False
    from sklearn.mixture import GaussianMixture

# ARCH / GARCH with fallback to numerical GARCH optimizer
try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False

# PyTorch with fallback
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ==========================================
# 1. BASELINE STRATEGIES
# ==========================================

class BaselineStrategies:
    @staticmethod
    def buy_and_hold(df: pd.DataFrame) -> pd.Series:
        """Constant long position (+1)."""
        return pd.Series(1.0, index=df.index, name="signal_bh")

    @staticmethod
    def moving_average_crossover(
        df: pd.DataFrame,
        fast_window: int = 20,
        slow_window: int = 50
    ) -> pd.Series:
        """Trend following: +1 when Fast SMA > Slow SMA, else 0 (or -1)."""
        fast_sma = df["Close"].rolling(fast_window).mean()
        slow_sma = df["Close"].rolling(slow_window).mean()
        signal = (fast_sma > slow_sma).astype(float)
        # Shift 1 to ensure trade execution at next bar (no lookahead)
        return signal.shift(1).fillna(0).rename("signal_ma_cross")

    @staticmethod
    def rsi_mean_reversion(
        df: pd.DataFrame,
        rsi_window: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0
    ) -> pd.Series:
        """Mean reversion: +1 when RSI < oversold, 0 when RSI > overbought."""
        if "rsi_14" in df.columns:
            rsi = df["rsi_14"]
        else:
            delta = df["Close"].diff()
            gain = delta.clip(lower=0).rolling(rsi_window).mean()
            loss = (-delta.clip(upper=0)).rolling(rsi_window).mean()
            rs = gain / (loss + 1e-9)
            rsi = 100 - (100 / (1 + rs))

        signal = pd.Series(0.0, index=df.index)
        signal[rsi < oversold] = 1.0
        signal[rsi > overbought] = 0.0
        signal = signal.ffill().fillna(0).shift(1).fillna(0)
        return signal.rename("signal_rsi_revert")


# ==========================================
# 2. ML MODELS (Random Forest & XGBoost)
# ==========================================

class MLStrategyModel:
    """
    ML Classifier/Regressor wrapper for directional prediction or return forecasting.
    """
    def __init__(
        self,
        model_type: str = "rf",  # 'rf' or 'xgb'
        task: str = "classification",  # 'classification' or 'regression'
        n_estimators: int = 150,
        max_depth: int = 4,
        min_samples_leaf: int = 10,
        probability_threshold: float = 0.52
    ):
        self.model_type = model_type
        self.task = task
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.prob_threshold = probability_threshold
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.feature_names = list(X.columns)
        X_scaled = self.scaler.fit_transform(X)

        if self.model_type == "xgb" and HAS_XGBOOST:
            if self.task == "classification":
                self.model = xgb.XGBClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=0.03,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    eval_metric="logloss"
                )
            else:
                self.model = xgb.XGBRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=0.03,
                    subsample=0.8,
                    random_state=42
                )
        else:
            if self.task == "classification":
                self.model = RandomForestClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=42,
                    n_jobs=1
                )
            else:
                self.model = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=42,
                    n_jobs=1
                )

        self.model.fit(X_scaled, y)
        return self

    def predict_signals(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (discrete_signals [-1, 0, 1] or [0, 1], continuous_probabilities/predictions)
        """
        X_scaled = self.scaler.transform(X)
        if self.task == "classification":
            probs = self.model.predict_proba(X_scaled)[:, 1]
            # Long when probability > threshold, flat when < 1 - threshold (or < threshold)
            signals = (probs > self.prob_threshold).astype(float)
            return signals, probs
        else:
            preds = self.model.predict(X_scaled)
            signals = (preds > 0.0005).astype(float)
            return signals, preds

    def get_feature_importances(self) -> pd.DataFrame:
        """Extract normalized feature importances."""
        if hasattr(self.model, "feature_importances_"):
            fi = self.model.feature_importances_
            df_fi = pd.DataFrame({
                "Feature": self.feature_names,
                "Importance": fi
            }).sort_values("Importance", ascending=False)
            return df_fi
        return pd.DataFrame(columns=["Feature", "Importance"])


# ==========================================
# 3. DEEP LEARNING (LSTM) MODEL
# ==========================================

if HAS_TORCH:
    class _PyTorchLSTM(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int = 32, num_layers: int = 1, dropout: float = 0.2):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
            self.fc = nn.Linear(hidden_dim, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :])
            return self.sigmoid(out)

class LSTMStrategyModel:
    """
    LSTM Recurrent Neural Network for temporal pattern extraction.
    """
    def __init__(self, seq_len: int = 10, hidden_dim: int = 32, epochs: int = 25, lr: float = 0.005):
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.scaler = StandardScaler()
        self.model = None

    def _create_sequences(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        Xs = []
        ys = []
        for i in range(len(X) - self.seq_len + 1):
            Xs.append(X[i : i + self.seq_len])
            if y is not None:
                ys.append(y[i + self.seq_len - 1])
        return np.array(Xs), (np.array(ys) if y is not None else None)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if not HAS_TORCH:
            return self
        X_scaled = self.scaler.fit_transform(X)
        Xs, ys = self._create_sequences(X_scaled, y.values)
        if len(Xs) == 0:
            return self

        tensor_X = torch.tensor(Xs, dtype=torch.float32)
        tensor_y = torch.tensor(ys, dtype=torch.float32).unsqueeze(1)

        self.model = _PyTorchLSTM(input_dim=X.shape[1], hidden_dim=self.hidden_dim)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(tensor_X)
            loss = criterion(outputs, tensor_y)
            loss.backward()
            optimizer.step()
        return self

    def predict_signals(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if not HAS_TORCH or self.model is None:
            # Fallback
            return np.ones(len(X)), np.full(len(X), 0.55)

        X_scaled = self.scaler.transform(X)
        Xs, _ = self._create_sequences(X_scaled)
        if len(Xs) == 0:
            return np.zeros(len(X)), np.zeros(len(X))

        self.model.eval()
        with torch.no_grad():
            tensor_X = torch.tensor(Xs, dtype=torch.float32)
            probs_seq = self.model(tensor_X).numpy().flatten()

        # Pad the initial seq_len - 1 entries
        pad_len = len(X) - len(probs_seq)
        probs = np.pad(probs_seq, (pad_len, 0), mode="edge")
        signals = (probs > 0.52).astype(float)
        return signals, probs


# ==========================================
# 4. REGIME DETECTION (Hidden Markov Model)
# ==========================================

class MarketRegimeDetector:
    """
    Unsupervised Market Regime Detection using Gaussian Hidden Markov Models.
    Identifies Low Volatility / Bull, High Volatility / Bear, and Mean-Reverting states.
    """
    def __init__(self, n_regimes: int = 3, random_state: int = 42):
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.model = None
        self.regime_names = {}

    def fit_predict(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Fit HMM on log returns and rolling realized volatility.
        Returns:
            regime_series: Series of inferred regime IDs (sorted from lowest to highest volatility)
            regime_stats: DataFrame with regime summary statistics (mean return, annualized vol, state freq)
        """
        features = pd.DataFrame(index=df.index)
        features["log_ret"] = df["log_ret"] if "log_ret" in df.columns else np.log(df["Close"] / df["Close"].shift(1))
        features["vol"] = (
            df["realized_vol_21d"]
            if "realized_vol_21d" in df.columns
            else features["log_ret"].rolling(21).std() * np.sqrt(252)
        )
        clean_feat = features.dropna()
        X = clean_feat[["log_ret", "vol"]].values

        if HAS_HMM:
            hmm = GaussianHMM(
                n_components=self.n_regimes,
                covariance_type="full",
                n_iter=100,
                random_state=self.random_state
            )
            hmm.fit(X)
            hidden_states = hmm.predict(X)
            self.model = hmm
        else:
            # Fallback to Gaussian Mixture
            gmm = GaussianMixture(
                n_components=self.n_regimes,
                covariance_type="full",
                random_state=self.random_state
            )
            hidden_states = gmm.fit_predict(X)
            self.model = gmm

        # Order regimes by volatility (0: Low Vol / Bull, 1: Moderate, 2: High Vol / Crisis)
        temp_df = clean_feat.copy()
        temp_df["raw_state"] = hidden_states
        vol_by_state = temp_df.groupby("raw_state")["vol"].mean().sort_values()
        state_map = {old_s: new_s for new_s, old_s in enumerate(vol_by_state.index)}

        ordered_states = np.array([state_map[s] for s in hidden_states])
        regime_series = pd.Series(ordered_states, index=clean_feat.index, name="regime").reindex(df.index).bfill().ffill()

        # Compute regime summary
        stats_list = []
        named_labels = ["Low Vol (Bull Trend)", "Moderate Vol (Normal)", "High Vol (Turbulence/Crisis)"]
        if self.n_regimes == 2:
            named_labels = ["Low Volatility (Bull)", "High Volatility (Turbulence)"]

        for s in range(self.n_regimes):
            mask = regime_series == s
            r_ret = df.loc[mask, "log_ret"].dropna() if "log_ret" in df.columns else np.log(df.loc[mask, "Close"] / df.loc[mask, "Close"].shift(1)).dropna()
            ann_ret = r_ret.mean() * 252 * 100 if len(r_ret) > 0 else 0
            ann_vol = r_ret.std() * np.sqrt(252) * 100 if len(r_ret) > 0 else 0
            sharpe = (ann_ret / ann_vol) if ann_vol > 0 else 0
            stats_list.append({
                "Regime ID": s,
                "Label": named_labels[min(s, len(named_labels)-1)],
                "Frequency (%)": f"{mask.mean() * 100:.1f}%",
                "Ann. Return (%)": f"{ann_ret:.2f}%",
                "Ann. Volatility (%)": f"{ann_vol:.2f}%",
                "Sharpe": f"{sharpe:.2f}",
                "Observation Count": int(mask.sum())
            })

        regime_stats = pd.DataFrame(stats_list)
        return regime_series, regime_stats


# ==========================================
# 5. VOLATILITY FORECASTING (GARCH vs ML)
# ==========================================

class VolatilityResearchEngine:
    """
    Rigorously compares Classical Econometric GARCH(1,1) with Machine Learning
    (RandomForest / XGBoost) for realized volatility forecasting.
    """
    @staticmethod
    def forecast_comparison(df: pd.DataFrame, test_pct: float = 0.25) -> Dict[str, Any]:
        """
        Run out-of-sample volatility forecast comparison.
        """
        data = df.copy()
        if "log_ret" not in data.columns:
            data["log_ret"] = np.log(data["Close"] / data["Close"].shift(1))
        
        # Realized 5-day annualized volatility target
        realized_vol = data["log_ret"].rolling(5).std() * np.sqrt(252)
        data["target_vol"] = realized_vol.shift(-5)
        
        # Features for ML volatility model
        data["vol_lag1"] = realized_vol
        data["vol_lag5"] = realized_vol.shift(5)
        data["vol_lag21"] = data["log_ret"].rolling(21).std() * np.sqrt(252)
        data["abs_ret_lag1"] = data["log_ret"].abs()
        data["abs_ret_lag5"] = data["log_ret"].abs().rolling(5).mean()
        data["parkinson_vol"] = np.sqrt(1.0 / (4.0 * np.log(2.0)) * ((np.log(data["High"] / data["Low"])) ** 2)).rolling(5).mean() * np.sqrt(252)

        data = data.dropna()
        n = len(data)
        split_idx = int(n * (1 - test_pct))

        train = data.iloc[:split_idx]
        test = data.iloc[split_idx:]

        y_train = train["target_vol"]
        y_test = test["target_vol"]

        # 1. Classical GARCH(1,1)
        garch_forecasts = np.zeros(len(test))
        if HAS_ARCH:
            try:
                # Fit GARCH on train returns
                ret_train_pct = train["log_ret"] * 100
                am = arch_model(ret_train_pct, vol="Garch", p=1, q=1, dist="normal", rescale=False)
                res = am.fit(disp="off")
                
                # Out-of-sample rolling 1-step forecast
                ret_all_pct = data["log_ret"] * 100
                am_full = arch_model(ret_all_pct, vol="Garch", p=1, q=1, dist="normal", rescale=False)
                res_full = am_full.filter(res.params)
                cond_vol = res_full.conditional_volatility / 100.0 * np.sqrt(252)
                garch_forecasts = cond_vol.iloc[split_idx:].values
            except Exception:
                garch_forecasts = train["target_vol"].rolling(21).mean().iloc[-1] * np.ones(len(test))
        else:
            # Analytical EWMA RiskMetrics proxy (decay = 0.94)
            ewma = train["log_ret"].ewm(alpha=0.06).std().iloc[-1] * np.sqrt(252)
            garch_forecasts = np.full(len(test), ewma)

        # 2. Machine Learning Volatility Forecaster (RandomForest / Gradient Boosting)
        ml_features = ["vol_lag1", "vol_lag5", "vol_lag21", "abs_ret_lag1", "abs_ret_lag5", "parkinson_vol"]
        X_train = train[ml_features]
        X_test = test[ml_features]

        rf_vol = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=5, random_state=42)
        rf_vol.fit(X_train, y_train)
        ml_forecasts = rf_vol.predict(X_test)

        # Metrics calculation (RMSE, MAE, QLIKE Loss)
        def calc_qlike(y_true, y_pred):
            ratio = (y_true**2) / (np.maximum(y_pred, 1e-4)**2)
            return np.mean(ratio - np.log(np.maximum(ratio, 1e-4)) - 1.0)

        metrics = {
            "GARCH(1,1)": {
                "RMSE": float(np.sqrt(mean_squared_error(y_test, garch_forecasts))),
                "MAE": float(mean_absolute_error(y_test, garch_forecasts)),
                "QLIKE": float(calc_qlike(y_test.values, garch_forecasts))
            },
            "ML Volatility Model": {
                "RMSE": float(np.sqrt(mean_squared_error(y_test, ml_forecasts))),
                "MAE": float(mean_absolute_error(y_test, ml_forecasts)),
                "QLIKE": float(calc_qlike(y_test.values, ml_forecasts))
            }
        }

        comparison_df = pd.DataFrame({
            "Actual_Realized_Vol": y_test.values,
            "GARCH_Forecast": garch_forecasts,
            "ML_Forecast": ml_forecasts
        }, index=test.index)

        return {
            "metrics": metrics,
            "comparison_df": comparison_df,
            "rf_feature_importances": pd.DataFrame({
                "Feature": ml_features,
                "Importance": rf_vol.feature_importances_
            }).sort_values("Importance", ascending=False)
        }
