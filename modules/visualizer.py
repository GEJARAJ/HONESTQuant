"""
HonestQuant: Visualizer Module
Produces responsive, ultra-polished interactive Plotly charts with dark quant styling.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Optional

DARK_THEME = {
    "bg_color": "#0E1117",
    "paper_color": "#0E1117",
    "card_bg": "#161B22",
    "grid_color": "#21262D",
    "text_color": "#E6EDF3",
    "accent_cyan": "#00E5FF",
    "accent_emerald": "#00E676",
    "accent_crimson": "#FF1744",
    "accent_amber": "#FFD600",
    "accent_purple": "#7C4DFF",
    "accent_blue": "#2979FF"
}

def create_equity_and_drawdown_chart(
    ml_oos_df: pd.DataFrame,
    bh_df: pd.DataFrame,
    ma_df: pd.DataFrame
) -> go.Figure:
    """
    Combined Equity Curve and Underwater Drawdown Chart.
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.7, 0.3],
        subplot_titles=("Cumulative Portfolio Value ($100k Base)", "Underwater Drawdown Profile")
    )

    # 1. Equity Curves
    fig.add_trace(
        go.Scatter(
            x=ml_oos_df.index,
            y=ml_oos_df["Equity"],
            mode="lines",
            name="ML Strategy (Honest OOS)",
            line=dict(color=DARK_THEME["accent_emerald"], width=2.5)
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=bh_df.index,
            y=bh_df["Equity"],
            mode="lines",
            name="Buy & Hold (Baseline)",
            line=dict(color="#8B949E", width=1.5, dash="dash")
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=ma_df.index,
            y=ma_df["Equity"],
            mode="lines",
            name="MA Crossover (Baseline)",
            line=dict(color=DARK_THEME["accent_amber"], width=1.5, dash="dot")
        ),
        row=1, col=1
    )

    # Highlight Kill-switch triggers
    if "Kill_Switch_Triggered" in ml_oos_df.columns:
        kill_dates = ml_oos_df[ml_oos_df["Kill_Switch_Triggered"]].index
        if len(kill_dates) > 0:
            fig.add_trace(
                go.Scatter(
                    x=kill_dates,
                    y=ml_oos_df.loc[kill_dates, "Equity"],
                    mode="markers",
                    name="Circuit Breaker (Kill Switch)",
                    marker=dict(symbol="x", color=DARK_THEME["accent_crimson"], size=8)
                ),
                row=1, col=1
            )

    # 2. Drawdowns
    fig.add_trace(
        go.Scatter(
            x=ml_oos_df.index,
            y=ml_oos_df["Drawdown"] * 100,
            mode="lines",
            name="ML Drawdown %",
            fill="tozeroy",
            fillcolor="rgba(0, 230, 118, 0.15)",
            line=dict(color=DARK_THEME["accent_emerald"], width=1.5)
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=bh_df.index,
            y=bh_df["Drawdown"] * 100,
            mode="lines",
            name="Buy & Hold Drawdown %",
            fill="tozeroy",
            fillcolor="rgba(255, 23, 68, 0.08)",
            line=dict(color=DARK_THEME["accent_crimson"], width=1.2, dash="dash")
        ),
        row=2, col=1
    )

    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor=DARK_THEME["bg_color"],
        paper_bgcolor=DARK_THEME["paper_color"],
        font=dict(color=DARK_THEME["text_color"], family="Inter, sans-serif"),
        height=580,
        margin=dict(l=40, r=20, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    fig.update_xaxes(gridcolor=DARK_THEME["grid_color"], showspikes=True)
    fig.update_yaxes(gridcolor=DARK_THEME["grid_color"], row=1, col=1, title="Portfolio ($)")
    fig.update_yaxes(gridcolor=DARK_THEME["grid_color"], row=2, col=1, title="Drawdown (%)", ticksuffix="%")

    return fig


def create_regime_timeline_chart(df: pd.DataFrame) -> go.Figure:
    """
    Asset Price Chart overlayed with Market Regime bands (HMM).
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            name="Asset Price (Close)",
            line=dict(color="#FFFFFF", width=2.0)
        )
    )

    # Color map for regimes
    regime_colors = {
        0: "rgba(0, 230, 118, 0.2)",   # Low Vol Bull
        1: "rgba(255, 214, 0, 0.18)",   # Moderate
        2: "rgba(255, 23, 68, 0.25)"    # High Vol Turbulence
    }

    if "Regime" in df.columns:
        # Detect regime blocks
        df_reg = df.copy()
        df_reg["regime_change"] = (df_reg["Regime"] != df_reg["Regime"].shift(1)).astype(int)
        df_reg["block_id"] = df_reg["regime_change"].cumsum()

        for _, block in df_reg.groupby("block_id"):
            r_val = block["Regime"].iloc[0]
            start_date = block.index[0]
            end_date = block.index[-1]
            fig.add_vrect(
                x0=start_date,
                x1=end_date,
                fillcolor=regime_colors.get(r_val, "rgba(255, 255, 255, 0.05)"),
                layer="below",
                line_width=0,
            )

    fig.update_layout(
        template="plotly_dark",
        title="Hidden Markov Model Market Regimes Overlay (Green: Bull/Low Vol | Yellow: Normal | Red: Crash/Turbulence)",
        plot_bgcolor=DARK_THEME["bg_color"],
        paper_bgcolor=DARK_THEME["paper_color"],
        font=dict(color=DARK_THEME["text_color"], family="Inter, sans-serif"),
        height=380,
        margin=dict(l=40, r=20, t=50, b=30),
        hovermode="x unified"
    )
    fig.update_xaxes(gridcolor=DARK_THEME["grid_color"])
    fig.update_yaxes(gridcolor=DARK_THEME["grid_color"], title="Price ($)")
    return fig


def create_is_vs_oos_haircut_chart(fold_df: pd.DataFrame) -> go.Figure:
    """
    In-Sample vs Out-of-Sample Accuracy and Sharpe Degradation per Fold.
    """
    fig = go.Figure()

    # Extract numeric accuracy
    is_acc = [float(x.replace("%", "")) for x in fold_df["IS Accuracy"]]
    oos_acc = [float(x.replace("%", "")) for x in fold_df["OOS Accuracy"]]

    fig.add_trace(
        go.Bar(
            x=fold_df["Fold"],
            y=is_acc,
            name="In-Sample Accuracy %",
            marker_color=DARK_THEME["accent_cyan"]
        )
    )

    fig.add_trace(
        go.Bar(
            x=fold_df["Fold"],
            y=oos_acc,
            name="Out-of-Sample Accuracy %",
            marker_color=DARK_THEME["accent_amber"]
        )
    )

    # 50% Random Guess Reference Line
    fig.add_hline(y=50, line_dash="dash", line_color="#FF1744", annotation_text="Random Guess (50%)")

    fig.update_layout(
        template="plotly_dark",
        title="Walk-Forward Fold Haircut: In-Sample vs Out-of-Sample Directional Accuracy",
        barmode="group",
        plot_bgcolor=DARK_THEME["bg_color"],
        paper_bgcolor=DARK_THEME["paper_color"],
        font=dict(color=DARK_THEME["text_color"], family="Inter, sans-serif"),
        height=360,
        margin=dict(l=40, r=20, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(ticksuffix="%", range=[40, 75])
    )
    fig.update_xaxes(gridcolor=DARK_THEME["grid_color"])
    fig.update_yaxes(gridcolor=DARK_THEME["grid_color"])
    return fig


def create_volatility_duel_chart(vol_results: Dict[str, Any]) -> go.Figure:
    """
    Visual comparison of GARCH(1,1) vs ML Volatility Forecast.
    """
    comp_df = vol_results["comparison_df"]
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=comp_df.index,
            y=comp_df["Actual_Realized_Vol"] * 100,
            mode="lines",
            name="Realized 5D Volatility (Actual)",
            line=dict(color="#FFFFFF", width=2.0)
        )
    )

    fig.add_trace(
        go.Scatter(
            x=comp_df.index,
            y=comp_df["GARCH_Forecast"] * 100,
            mode="lines",
            name="GARCH(1,1) Forecast",
            line=dict(color=DARK_THEME["accent_amber"], width=1.5, dash="dash")
        )
    )

    fig.add_trace(
        go.Scatter(
            x=comp_df.index,
            y=comp_df["ML_Forecast"] * 100,
            mode="lines",
            name="ML Gradient Boosted Volatility",
            line=dict(color=DARK_THEME["accent_cyan"], width=1.8)
        )
    )

    fig.update_layout(
        template="plotly_dark",
        title="Volatility Forecast Duel: Classical GARCH(1,1) vs Machine Learning (Out-of-Sample)",
        plot_bgcolor=DARK_THEME["bg_color"],
        paper_bgcolor=DARK_THEME["paper_color"],
        font=dict(color=DARK_THEME["text_color"], family="Inter, sans-serif"),
        height=380,
        margin=dict(l=40, r=20, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(ticksuffix="%", title="Annualized Volatility (%)")
    )
    fig.update_xaxes(gridcolor=DARK_THEME["grid_color"])
    fig.update_yaxes(gridcolor=DARK_THEME["grid_color"])
    return fig


def create_feature_importance_chart(df_fi: pd.DataFrame) -> go.Figure:
    """
    Feature importance ranking horizontal bar chart.
    """
    top_10 = df_fi.head(10).iloc[::-1]
    fig = go.Figure(
        go.Bar(
            x=top_10["Importance"],
            y=top_10["Feature"],
            orientation="h",
            marker=dict(
                color=top_10["Importance"],
                colorscale="Viridis"
            )
        )
    )

    fig.update_layout(
        template="plotly_dark",
        title="Top 10 Feature Drivers of ML Predictions (Feature Attributions)",
        plot_bgcolor=DARK_THEME["bg_color"],
        paper_bgcolor=DARK_THEME["paper_color"],
        font=dict(color=DARK_THEME["text_color"], family="Inter, sans-serif"),
        height=350,
        margin=dict(l=120, r=20, t=50, b=30),
    )
    fig.update_xaxes(gridcolor=DARK_THEME["grid_color"], title="Normalized Importance")
    fig.update_yaxes(gridcolor=DARK_THEME["grid_color"])
    return fig


def create_monte_carlo_fan_chart(returns: pd.Series, n_sims: int = 300, horizon_days: int = 252) -> go.Figure:
    """
    Monte Carlo Bootstrap Fan Chart showing 90% and 50% confidence envelopes.
    """
    clean_ret = returns.dropna().values
    if len(clean_ret) < 20:
        return go.Figure()

    np.random.seed(42)
    sim_paths = np.zeros((n_sims, horizon_days + 1))
    sim_paths[:, 0] = 100_000.0

    for i in range(n_sims):
        bootstrapped_ret = np.random.choice(clean_ret, size=horizon_days, replace=True)
        sim_paths[i, 1:] = 100_000.0 * np.cumprod(1 + bootstrapped_ret)

    p05 = np.percentile(sim_paths, 5, axis=0)
    p25 = np.percentile(sim_paths, 25, axis=0)
    p50 = np.percentile(sim_paths, 50, axis=0)
    p75 = np.percentile(sim_paths, 75, axis=0)
    p95 = np.percentile(sim_paths, 95, axis=0)

    x_axis = list(range(horizon_days + 1))
    fig = go.Figure()

    # 90% Confidence band
    fig.add_trace(
        go.Scatter(
            x=x_axis + x_axis[::-1],
            y=list(p95) + list(p05[::-1]),
            fill="toself",
            fillcolor="rgba(0, 229, 255, 0.1)",
            line=dict(color="rgba(255,255,255,0)"),
            name="90% Confidence Interval",
            hoverinfo="skip"
        )
    )

    # 50% Confidence band
    fig.add_trace(
        go.Scatter(
            x=x_axis + x_axis[::-1],
            y=list(p75) + list(p25[::-1]),
            fill="toself",
            fillcolor="rgba(0, 229, 255, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="50% Confidence Interval",
            hoverinfo="skip"
        )
    )

    # Median Path
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=p50,
            mode="lines",
            name="Median Path (50th %ile)",
            line=dict(color=DARK_THEME["accent_cyan"], width=2.5)
        )
    )

    fig.update_layout(
        template="plotly_dark",
        title=f"Monte Carlo 1-Year Forward Resampling Stress Test ({n_sims} Paths)",
        plot_bgcolor=DARK_THEME["bg_color"],
        paper_bgcolor=DARK_THEME["paper_color"],
        font=dict(color=DARK_THEME["text_color"], family="Inter, sans-serif"),
        height=380,
        margin=dict(l=50, r=20, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(title="Portfolio Equity ($)")
    )
    fig.update_xaxes(gridcolor=DARK_THEME["grid_color"], title="Forward Trading Days")
    fig.update_yaxes(gridcolor=DARK_THEME["grid_color"])
    return fig
