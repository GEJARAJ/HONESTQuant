"""
HonestQuant: Paper Trading & Live Simulation Engine
Simulates live out-of-sample forward trading with virtual portfolio balance,
order execution ledger, slippage simulation, and real-time decision explanations.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional


class PaperTradingEngine:
    """
    Virtual live paper trading simulation with realistic execution logs.
    """

    def __init__(
        self,
        initial_balance: float = 100_000.0,
        transaction_cost_bps: float = 10.0,
        slippage_bps: float = 5.0,
        state_file: str = "data_cache/paper_trading_state.json"
    ):
        self.initial_balance = initial_balance
        self.transaction_cost_bps = transaction_cost_bps
        self.slippage_bps = slippage_bps
        self.friction_rate = (transaction_cost_bps + slippage_bps) / 10000.0
        self.state_file = state_file

        self.cash = initial_balance
        self.shares = 0.0
        self.portfolio_value = initial_balance
        self.peak_value = initial_balance
        self.trade_history: List[Dict[str, Any]] = []
        self.daily_snapshots: List[Dict[str, Any]] = []
        self.load_state()

    def reset(self):
        """Reset paper trading state to initial balance."""
        self.cash = self.initial_balance
        self.shares = 0.0
        self.portfolio_value = self.initial_balance
        self.peak_value = self.initial_balance
        self.trade_history = []
        self.daily_snapshots = []
        if os.path.exists(self.state_file):
            os.remove(self.state_file)

    def execute_live_step(
        self,
        timestamp: str,
        current_price: float,
        target_signal: float,      # 1.0 (Long), 0.0 (Cash/Flat), -1.0 (Short)
        model_prob: float,
        target_allocation: float,  # Sized leverage / weight e.g. 0.8
        top_features: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Process a new market bar / live tick and adjust position to target allocation.
        """
        # Calculate current equity
        current_holdings_val = self.shares * current_price
        self.portfolio_value = self.cash + current_holdings_val
        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value

        current_dd = (self.portfolio_value - self.peak_value) / self.peak_value

        # Determine target dollar value
        desired_holding_val = self.portfolio_value * target_allocation
        target_shares = desired_holding_val / current_price if current_price > 0 else 0
        shares_diff = target_shares - self.shares

        trade_executed = False
        trade_details = None

        if abs(shares_diff) * current_price > 500:  # Minimum trade threshold $500
            # Apply slippage: buy higher, sell lower
            exec_price = current_price * (1.0 + (self.slippage_bps / 10000.0 if shares_diff > 0 else -self.slippage_bps / 10000.0))
            trade_cost = abs(shares_diff) * exec_price * (self.transaction_cost_bps / 10000.0)

            cash_flow = -(shares_diff * exec_price) - trade_cost
            self.cash += cash_flow
            self.shares = target_shares
            trade_executed = True

            trade_details = {
                "Timestamp": timestamp,
                "Action": "BUY" if shares_diff > 0 else "SELL",
                "Shares": round(abs(shares_diff), 4),
                "Exec_Price": round(exec_price, 2),
                "Trade_Value": round(abs(shares_diff * exec_price), 2),
                "Friction_Fee": round(trade_cost, 2),
                "Signal_Prob": f"{model_prob * 100:.1f}%",
                "Reasoning": f"ML Bullishness: {model_prob*100:.1f}%, Allocation: {target_allocation*100:.1f}%"
            }
            self.trade_history.append(trade_details)

        # Update final portfolio value
        self.portfolio_value = self.cash + (self.shares * current_price)
        total_pnl = self.portfolio_value - self.initial_balance
        total_pnl_pct = (total_pnl / self.initial_balance) * 100

        snapshot = {
            "Timestamp": timestamp,
            "Price": current_price,
            "Portfolio_Value": round(self.portfolio_value, 2),
            "Cash": round(self.cash, 2),
            "Holdings_Value": round(self.shares * current_price, 2),
            "Target_Weight": round(target_allocation, 2),
            "Total_PnL_Pct": round(total_pnl_pct, 2),
            "Drawdown_Pct": round(current_dd * 100, 2),
            "Signal_Prob": round(model_prob, 3)
        }
        self.daily_snapshots.append(snapshot)
        self.save_state()

        return {
            "trade_executed": trade_executed,
            "trade_details": trade_details,
            "snapshot": snapshot,
            "top_features": top_features
        }

    def save_state(self):
        try:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            data = {
                "cash": self.cash,
                "shares": self.shares,
                "portfolio_value": self.portfolio_value,
                "peak_value": self.peak_value,
                "trade_history": self.trade_history[-200:],
                "daily_snapshots": self.daily_snapshots[-500:]
            }
            with open(self.state_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                    self.cash = data.get("cash", self.initial_balance)
                    self.shares = data.get("shares", 0.0)
                    self.portfolio_value = data.get("portfolio_value", self.initial_balance)
                    self.peak_value = data.get("peak_value", self.initial_balance)
                    self.trade_history = data.get("trade_history", [])
                    self.daily_snapshots = data.get("daily_snapshots", [])
            except Exception:
                pass
