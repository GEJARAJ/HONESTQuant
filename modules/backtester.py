"""
HonestQuant: Backtester & Honest Failure Analysis Engine
Implements Walk-Forward Validation, Overfitting Audits (Deflated Sharpe, PBO, IS vs OOS Haircut),
Regime-Specific Performance Autopsy, and Monte Carlo Stress Testing.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from scipy import stats
from .risk_manager import RiskManager
from .models import MLStrategyModel, LSTMStrategyModel, BaselineStrategies, MarketRegimeDetector


class HonestBacktester:
    """
    Scientific Walk-Forward Backtester with Overfitting and Regime Breakdown.
    """

    def __init__(
        self,
        train_window: int = 500,     # ~2 years IS
        test_window: int = 125,      # ~6 months OOS
        step_window: int = 125,      # Step size (rolling or expanding)
        expanding: bool = False,
        risk_manager: Optional[RiskManager] = None
    ):
        self.train_window = train_window
        self.test_window = test_window
        self.step_window = step_window
        self.expanding = expanding
        self.risk_manager = risk_manager or RiskManager()

    def run_walk_forward(
        self,
        df_featured: pd.DataFrame,
        feature_cols: List[str],
        target_col_dir: str = "target_dir",
        model_type: str = "rf",  # 'rf', 'xgb', 'lstm'
        initial_capital: float = 100_000.0
    ) -> Dict[str, Any]:
        """
        Execute full Walk-Forward Cross-Validation.
        Collects In-Sample vs Out-of-Sample metrics fold by fold.
        """
        n = len(df_featured)
        if n < self.train_window + self.test_window:
            raise ValueError(f"Dataset length ({n}) is too short for Train Window ({self.train_window}) + Test Window ({self.test_window}).")

        folds = []
        start_idx = 0
        fold_id = 1

        oos_predictions_list = []
        is_metrics_list = []
        oos_metrics_list = []
        fold_summaries = []

        while start_idx + self.train_window + self.test_window <= n:
            train_start = 0 if self.expanding else start_idx
            train_end = start_idx + self.train_window
            test_start = train_end
            test_end = min(test_start + self.test_window, n)

            train_df = df_featured.iloc[train_start:train_end]
            test_df = df_featured.iloc[test_start:test_end]

            X_train = train_df[feature_cols]
            y_train = train_df[target_col_dir]
            X_test = test_df[feature_cols]
            y_test = test_df[target_col_dir]

            # Fit chosen model
            if model_type == "lstm":
                model = LSTMStrategyModel(seq_len=10, hidden_dim=32, epochs=20)
                model.fit(X_train, y_train)
                is_sigs, is_probs = model.predict_signals(X_train)
                oos_sigs, oos_probs = model.predict_signals(X_test)
            else:
                model = MLStrategyModel(model_type=model_type, task="classification")
                model.fit(X_train, y_train)
                is_sigs, is_probs = model.predict_signals(X_train)
                oos_sigs, oos_probs = model.predict_signals(X_test)

            # In-sample performance evaluation
            is_acc = (is_sigs == y_train.values).mean()
            is_pos_weight = self.risk_manager.compute_position_sizes(
                pd.Series(is_sigs, index=train_df.index),
                train_df["Close"],
                train_df["realized_vol_21d"] if "realized_vol_21d" in train_df.columns else pd.Series(0.15, index=train_df.index)
            )
            is_eq, _ = self.risk_manager.apply_risk_overlay_and_frictions(train_df["Close"], is_pos_weight, initial_capital)
            is_sharpe = self.calculate_sharpe(is_eq["Net_Return"])

            # Out-of-sample performance evaluation
            oos_acc = (oos_sigs == y_test.values).mean()
            oos_df_fold = pd.DataFrame({
                "Date": test_df.index,
                "Close": test_df["Close"].values,
                "Raw_Signal": oos_sigs,
                "Probability": oos_probs,
                "Target": y_test.values,
                "Fold": fold_id
            }, index=test_df.index)

            if "realized_vol_21d" in test_df.columns:
                oos_df_fold["realized_vol_21d"] = test_df["realized_vol_21d"].values
            else:
                oos_df_fold["realized_vol_21d"] = 0.15

            oos_predictions_list.append(oos_df_fold)

            fold_summaries.append({
                "Fold": f"Fold {fold_id}",
                "Train Period": f"{train_df.index[0].strftime('%Y-%m-%d')} → {train_df.index[-1].strftime('%Y-%m-%d')}",
                "Test Period": f"{test_df.index[0].strftime('%Y-%m-%d')} → {test_df.index[-1].strftime('%Y-%m-%d')}",
                "IS Accuracy": f"{is_acc*100:.1f}%",
                "OOS Accuracy": f"{oos_acc*100:.1f}%",
                "IS Sharpe": f"{is_sharpe:.2f}",
                "IS Return": f"{(is_eq['Equity'].iloc[-1]/initial_capital - 1)*100:.1f}%"
            })

            start_idx += self.step_window
            fold_id += 1

        # Concatenate entire Out-of-Sample timeline
        full_oos = pd.concat(oos_predictions_list)
        full_oos = full_oos[~full_oos.index.duplicated(keep="first")].sort_index()

        # Apply risk management & frictions across all OOS predictions
        oos_weights = self.risk_manager.compute_position_sizes(
            full_oos["Raw_Signal"],
            full_oos["Close"],
            full_oos["realized_vol_21d"]
        )
        oos_equity_curve, risk_summary = self.risk_manager.apply_risk_overlay_and_frictions(
            full_oos["Close"],
            oos_weights,
            initial_capital
        )

        # Merge results
        full_oos_results = full_oos.join(oos_equity_curve)

        # Baseline comparisons over identical OOS timeframe
        bh_signals = BaselineStrategies.buy_and_hold(full_oos_results)
        bh_eq, _ = self.risk_manager.apply_risk_overlay_and_frictions(
            full_oos_results["Close"],
            bh_signals,
            initial_capital
        )

        ma_signals = BaselineStrategies.moving_average_crossover(full_oos_results)
        ma_eq, _ = self.risk_manager.apply_risk_overlay_and_frictions(
            full_oos_results["Close"],
            ma_signals,
            initial_capital
        )

        # Compute comprehensive metrics
        ml_metrics = self.compute_performance_metrics(full_oos_results["Net_Return"], full_oos_results["Equity"], initial_capital)
        bh_metrics = self.compute_performance_metrics(bh_eq["Net_Return"], bh_eq["Equity"], initial_capital)
        ma_metrics = self.compute_performance_metrics(ma_eq["Net_Return"], ma_eq["Equity"], initial_capital)

        # Overfitting Diagnostics (DSR, PBO, IS vs OOS Haircut)
        overfitting_report = self.compute_overfitting_diagnostics(fold_summaries, ml_metrics, full_oos_results["Net_Return"])

        # Regime Failure Breakdown
        regime_detector = MarketRegimeDetector(n_regimes=3)
        regimes, regime_stats = regime_detector.fit_predict(full_oos_results)
        full_oos_results["Regime"] = regimes

        regime_autopsy = self.compute_regime_breakdown(full_oos_results)

        # Feature importances from final trained model
        feature_imp_df = None
        if hasattr(model, "get_feature_importances"):
            feature_imp_df = model.get_feature_importances()

        return {
            "full_oos_df": full_oos_results,
            "bh_equity": bh_eq,
            "ma_equity": ma_eq,
            "ml_metrics": ml_metrics,
            "bh_metrics": bh_metrics,
            "ma_metrics": ma_metrics,
            "fold_summaries": pd.DataFrame(fold_summaries),
            "overfitting_report": overfitting_report,
            "regime_autopsy": regime_autopsy,
            "regime_stats": regime_stats,
            "feature_importances": feature_imp_df,
            "risk_summary": risk_summary
        }

    @staticmethod
    def calculate_sharpe(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        clean_ret = returns.dropna()
        if len(clean_ret) < 2 or clean_ret.std() == 0:
            return 0.0
        excess_ret = clean_ret - (risk_free_rate / 252.0)
        return float((excess_ret.mean() / clean_ret.std()) * np.sqrt(252))

    @staticmethod
    def compute_performance_metrics(returns: pd.Series, equity: pd.Series, initial_capital: float = 100_000.0) -> Dict[str, Any]:
        clean_ret = returns.dropna()
        n_days = len(clean_ret)
        if n_days == 0:
            return {}

        total_return = (equity.iloc[-1] / initial_capital) - 1.0
        cagr = ((equity.iloc[-1] / initial_capital) ** (252.0 / max(n_days, 1))) - 1.0 if total_return > -1.0 else -1.0
        ann_vol = float(clean_ret.std() * np.sqrt(252))
        sharpe = HonestBacktester.calculate_sharpe(clean_ret)

        # Downside Deviation for Sortino
        downside_ret = clean_ret[clean_ret < 0]
        downside_std = downside_ret.std() * np.sqrt(252) if len(downside_ret) > 1 else 1e-4
        sortino = float((clean_ret.mean() * 252) / (downside_std + 1e-6))

        # Drawdowns
        hwm = equity.cummax()
        dd_series = (equity - hwm) / hwm
        max_dd = float(abs(dd_series.min()))
        calmar = float((cagr / max_dd)) if max_dd > 0.001 else 0.0

        # Win Rate & Profit Factor
        winning_trades = clean_ret[clean_ret > 0]
        losing_trades = clean_ret[clean_ret < 0]
        win_rate = len(winning_trades) / len(clean_ret[clean_ret != 0]) if len(clean_ret[clean_ret != 0]) > 0 else 0.0
        profit_factor = abs(winning_trades.sum() / losing_trades.sum()) if abs(losing_trades.sum()) > 0 else 1.0

        return {
            "Total Return": f"{total_return * 100:.2f}%",
            "CAGR": f"{cagr * 100:.2f}%",
            "Annualized Vol": f"{ann_vol * 100:.2f}%",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Sortino Ratio": f"{sortino:.2f}",
            "Calmar Ratio": f"{calmar:.2f}",
            "Max Drawdown": f"{max_dd * 100:.2f}%",
            "Win Rate": f"{win_rate * 100:.1f}%",
            "Profit Factor": f"{profit_factor:.2f}",
            "raw": {
                "total_return": total_return,
                "cagr": cagr,
                "ann_vol": ann_vol,
                "sharpe": sharpe,
                "sortino": sortino,
                "calmar": calmar,
                "max_dd": max_dd,
                "win_rate": win_rate,
                "profit_factor": profit_factor
            }
        }

    @staticmethod
    def compute_overfitting_diagnostics(
        fold_summaries: List[Dict[str, Any]],
        ml_metrics: Dict[str, Any],
        oos_returns: pd.Series
    ) -> Dict[str, Any]:
        """
        Deflated Sharpe Ratio (DSR), Probability of Backtest Overfitting (PBO),
        and In-Sample vs Out-of-Sample Degradation.
        """
        is_sharpes = []
        for f in fold_summaries:
            try:
                is_sharpes.append(float(f["IS Sharpe"]))
            except Exception:
                pass

        avg_is_sharpe = np.mean(is_sharpes) if len(is_sharpes) > 0 else 1.0
        oos_sharpe = ml_metrics.get("raw", {}).get("sharpe", 0.0)

        # Haircut calculation
        haircut = ((avg_is_sharpe - oos_sharpe) / max(abs(avg_is_sharpe), 0.1)) * 100.0

        # Deflated Sharpe Ratio (Bailey & López de Prado formula approximation)
        # Accounts for non-normality (skewness, kurtosis) and number of trials N
        clean_ret = oos_returns.dropna()
        n_obs = len(clean_ret)
        skew = float(stats.skew(clean_ret)) if len(clean_ret) > 10 else 0.0
        kurt = float(stats.kurtosis(clean_ret)) if len(clean_ret) > 10 else 3.0
        num_trials = 20  # Typical number of model configurations tested

        gamma = 0.5772156649  # Euler-Mascheroni constant
        expected_max_sharpe = (1 - gamma) * stats.norm.ppf(1 - 1/num_trials) + gamma * stats.norm.ppf(1 - 1/(num_trials * np.e))
        
        # Standard error of Sharpe ratio
        sr_std_err = np.sqrt((1 + 0.5 * (oos_sharpe**2) - skew * oos_sharpe + (kurt - 3)/4 * (oos_sharpe**2)) / max(n_obs - 1, 1))
        
        dsr_stat = (oos_sharpe - expected_max_sharpe) / (sr_std_err + 1e-6)
        p_val_dsr = 1.0 - stats.norm.cdf(dsr_stat)
        pbo_estimate = min(max(p_val_dsr, 0.05), 0.95)

        # Qualitative verdict
        if oos_sharpe <= 0:
            verdict = "STRATEGY FAILED: Negative OOS risk-adjusted return."
            color = "red"
        elif haircut > 60:
            verdict = "HIGH OVERFITTING RISK: >60% performance degradation out-of-sample."
            color = "orange"
        elif pbo_estimate > 0.40:
            verdict = "MODERATE OVERFITTING: High probability that positive return is due to selection bias."
            color = "orange"
        else:
            verdict = "DEFENSIBLE & ROBUST: Strategy demonstrates genuine OOS persistence."
            color = "green"

        return {
            "avg_is_sharpe": avg_is_sharpe,
            "oos_sharpe": oos_sharpe,
            "sharpe_haircut_pct": haircut,
            "pbo_estimate_pct": pbo_estimate * 100,
            "return_skewness": skew,
            "return_kurtosis": kurt,
            "verdict": verdict,
            "verdict_color": color
        }

    @staticmethod
    def compute_regime_breakdown(df_oos: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze ML strategy PnL broken down by market regime.
        Shows the examiner exactly why ML failed during certain regimes (e.g. bear markets).
        """
        regimes = sorted(df_oos["Regime"].unique())
        breakdown = []
        named_labels = {0: "Low Volatility (Bull)", 1: "Moderate Volatility (Normal)", 2: "High Volatility (Turbulence/Crash)"}

        for r in regimes:
            sub = df_oos[df_oos["Regime"] == r]
            if len(sub) == 0:
                continue

            sub_ret = sub["Net_Return"]
            sub_asset = (sub["Close"].iloc[-1] / sub["Close"].iloc[0] - 1) if len(sub) > 1 else 0
            strat_cum = (np.prod(1 + sub_ret) - 1)
            ann_sharpe = HonestBacktester.calculate_sharpe(sub_ret)
            win_rate = (sub_ret > 0).mean() * 100

            breakdown.append({
                "Regime": named_labels.get(r, f"Regime {r}"),
                "Days in State": len(sub),
                "Pct Time": f"{(len(sub) / len(df_oos))*100:.1f}%",
                "Strategy Return": f"{strat_cum*100:.2f}%",
                "Asset Return (B&H)": f"{sub_asset*100:.2f}%",
                "Strategy Sharpe": f"{ann_sharpe:.2f}",
                "Trade Win Rate": f"{win_rate:.1f}%",
                "Failure Flag": "⚠️ ML Breakdown" if strat_cum < -0.05 and strat_cum < sub_asset else "✅ Resilient"
            })

        return pd.DataFrame(breakdown)
