"""
HonestQuant: Risk Management Module
Implements dynamic position sizing (Volatility Targeting, Fractional Kelly),
Stop-Loss / Take-Profit rules, Slippage modeling, and Max Drawdown Kill-Switch.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional


class RiskManager:
    """
    Production-grade Risk Management & Execution Friction Engine.
    """

    def __init__(
        self,
        sizing_method: str = "vol_target",  # 'fixed', 'vol_target', 'kelly'
        target_annual_vol: float = 0.15,    # 15% target vol
        max_leverage: float = 1.5,
        stop_loss_pct: float = 0.03,        # 3% stop loss per trade
        trailing_stop_pct: float = 0.05,    # 5% trailing stop
        max_drawdown_kill: float = 0.12,    # 12% Max DD Circuit Breaker
        cooldown_bars: int = 15,            # Bars to stay flat after kill switch trigger
        transaction_cost_bps: float = 10.0, # 10 bps (0.10%) per one-way trade
        slippage_bps: float = 5.0           # 5 bps estimated slippage
    ):
        self.sizing_method = sizing_method
        self.target_annual_vol = target_annual_vol
        self.max_leverage = max_leverage
        self.stop_loss_pct = stop_loss_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.max_drawdown_kill = max_drawdown_kill
        self.cooldown_bars = cooldown_bars
        self.friction_rate = (transaction_cost_bps + slippage_bps) / 10000.0

    def compute_position_sizes(
        self,
        raw_signals: pd.Series,
        price_series: pd.Series,
        realized_vol_series: pd.Series,
        win_rate: float = 0.53,
        win_loss_ratio: float = 1.2
    ) -> pd.Series:
        """
        Adjust raw directional signals [-1, 0, 1] into sized portfolio allocations [-max_lev, +max_lev].
        """
        weights = raw_signals.copy().astype(float)

        if self.sizing_method == "vol_target":
            # Volatility targeting: Weight = Target_Vol / Current_Realized_Vol
            # Clipped between 0 and max_leverage
            vol_safe = realized_vol_series.replace(0, np.nan).ffill().fillna(0.20)
            vol_multiplier = self.target_annual_vol / vol_safe
            vol_multiplier = vol_multiplier.clip(0.1, self.max_leverage)
            weights = weights * vol_multiplier

        elif self.sizing_method == "kelly":
            # Half-Kelly for safety: f* = (p*b - q) / b * 0.5
            p = win_rate
            q = 1.0 - p
            b = max(win_loss_ratio, 0.1)
            raw_kelly = (p * b - q) / b
            kelly_fraction = max(0.0, min(raw_kelly * 0.5, self.max_leverage))
            weights = weights * kelly_fraction

        else:
            # Fixed sizing
            weights = weights.clip(-self.max_leverage, self.max_leverage)

        return weights

    def apply_risk_overlay_and_frictions(
        self,
        prices: pd.Series,
        position_weights: pd.Series,
        initial_capital: float = 100_000.0
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Simulate step-by-step risk management:
        - Trailing stop-loss
        - Max drawdown circuit breaker (kill switch)
        - Transaction fees and slippage on rebalances
        """
        n = len(prices)
        dates = prices.index

        equity = np.zeros(n)
        cash = np.zeros(n)
        holdings = np.zeros(n)
        net_returns = np.zeros(n)
        active_weights = np.zeros(n)
        turnover = np.zeros(n)
        friction_costs = np.zeros(n)
        kill_switch_active = np.zeros(n, dtype=bool)

        equity[0] = initial_capital
        cash[0] = initial_capital
        peak_equity = initial_capital
        cooldown_counter = 0

        # Position tracking for trailing stops
        in_trade = False
        entry_price = 0.0
        peak_trade_price = 0.0
        trade_dir = 0

        raw_w = position_weights.values
        p = prices.values

        for t in range(1, n):
            prev_equity = equity[t-1]
            if prev_equity > peak_equity:
                peak_equity = prev_equity

            current_dd = (prev_equity - peak_equity) / peak_equity

            # 1. Check Max Drawdown Kill Switch
            if current_dd < -self.max_drawdown_kill:
                cooldown_counter = self.cooldown_bars
                kill_switch_active[t] = True

            if cooldown_counter > 0:
                desired_weight = 0.0
                cooldown_counter -= 1
                kill_switch_active[t] = True
            else:
                desired_weight = raw_w[t-1]

            # 2. Check Trailing Stop-Loss & Trade Management
            if desired_weight != 0 and not in_trade:
                in_trade = True
                entry_price = p[t-1]
                peak_trade_price = p[t-1]
                trade_dir = 1 if desired_weight > 0 else -1

            if in_trade:
                if trade_dir == 1:
                    peak_trade_price = max(peak_trade_price, p[t-1])
                    dd_from_peak = (p[t-1] - peak_trade_price) / peak_trade_price
                    dd_from_entry = (p[t-1] - entry_price) / entry_price
                    if dd_from_peak < -self.trailing_stop_pct or dd_from_entry < -self.stop_loss_pct:
                        # Stop triggered! Force flat
                        desired_weight = 0.0
                        in_trade = False
                elif trade_dir == -1:
                    peak_trade_price = min(peak_trade_price, p[t-1])
                    dd_from_peak = (peak_trade_price - p[t-1]) / peak_trade_price
                    dd_from_entry = (entry_price - p[t-1]) / entry_price
                    if dd_from_peak < -self.trailing_stop_pct or dd_from_entry < -self.stop_loss_pct:
                        desired_weight = 0.0
                        in_trade = False

            if desired_weight == 0:
                in_trade = False

            active_weights[t] = desired_weight

            # 3. Compute asset return & friction
            asset_ret = (p[t] - p[t-1]) / p[t-1]
            weight_change = abs(active_weights[t] - active_weights[t-1])
            cost = weight_change * self.friction_rate
            turnover[t] = weight_change
            friction_costs[t] = cost * prev_equity

            period_ret = (active_weights[t] * asset_ret) - cost
            net_returns[t] = period_ret
            equity[t] = prev_equity * (1.0 + period_ret)

        equity_curve = pd.DataFrame({
            "Equity": equity,
            "Net_Return": net_returns,
            "Position_Weight": active_weights,
            "Turnover": turnover,
            "Friction_Cost": friction_costs,
            "Kill_Switch_Triggered": kill_switch_active
        }, index=dates)

        # High water mark and drawdown
        equity_curve["High_Water_Mark"] = equity_curve["Equity"].cummax()
        equity_curve["Drawdown"] = (equity_curve["Equity"] - equity_curve["High_Water_Mark"]) / equity_curve["High_Water_Mark"]

        summary = {
            "Total_Friction_Cost_Paid": float(equity_curve["Friction_Cost"].sum()),
            "Total_Turnover": float(equity_curve["Turnover"].sum()),
            "Kill_Switch_Events": int((equity_curve["Kill_Switch_Triggered"].diff() == 1).sum()),
            "Final_Equity": float(equity_curve["Equity"].iloc[-1])
        }

        return equity_curve, summary
