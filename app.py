"""
HonestQuant: Defensible ML Trading Research & Audit Platform
Streamlit Main Application with High-End Glassmorphism Quant Studio UI/UX.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import streamlit as st

# Add modules path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.data_pipeline import DataPipeline
from modules.models import MLStrategyModel, LSTMStrategyModel, BaselineStrategies, MarketRegimeDetector, VolatilityResearchEngine
from modules.risk_manager import RiskManager
from modules.backtester import HonestBacktester
from modules.paper_trading import PaperTradingEngine
from modules import visualizer

# ==========================================
# STREAMLIT CONFIG & CUSTOM CSS
# ==========================================
st.set_page_config(
    page_title="HonestQuant | ML Trading Research Platform",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    /* Dark Quant Terminal Theme & Global Typography */
    .stApp {
        background-color: #0B0E14 !important;
        color: #F0F6FC !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Top Streamlit Header */
    header[data-testid="stHeader"] {
        background-color: #0B0E14 !important;
    }
    
    /* Ensure all markdown headings and text are bright and readable */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
    p, span, label, div {
        color: #E2E8F0;
    }
    
    /* Sidebar Styling - Dark Sleek High Contrast */
    section[data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid #1F2937 !important;
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {
        color: #38BDF8 !important;
        font-weight: 700 !important;
        margin-top: 10px !important;
        margin-bottom: 8px !important;
    }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] span {
        color: #F8FAFC !important;
        font-weight: 600 !important;
        font-size: 13px !important;
    }
    section[data-testid="stSidebar"] .stSelectbox,
    section[data-testid="stSidebar"] .stNumberInput,
    section[data-testid="stSidebar"] .stSlider {
        margin-bottom: 12px !important;
    }

    /* Input Fields, Selectboxes, and Sliders */
    div[data-baseweb="select"] > div {
        background-color: #1E293B !important;
        color: #FFFFFF !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="select"] * {
        color: #FFFFFF !important;
    }
    div[data-baseweb="input"] {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="input"] input {
        color: #FFFFFF !important;
        background-color: #1E293B !important;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #161B22 0%, #0D1117 100%);
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #30363D;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    }
    .main-title {
        font-size: 28px;
        font-weight: 800;
        background: linear-gradient(90deg, #00E5FF, #00E676);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    .sub-title {
        color: #94A3B8 !important;
        font-size: 14px;
        margin-bottom: 0px;
        font-weight: 500;
    }
    
    /* Metric Cards */
    .metric-card {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #00E5FF;
    }
    .metric-label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #94A3B8 !important;
        margin-bottom: 6px;
        font-weight: 700;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 800;
        color: #FFFFFF !important;
    }
    .metric-sub {
        font-size: 12px;
        margin-top: 6px;
        color: #94A3B8 !important;
    }
    .val-green { color: #00E676 !important; font-weight: 700; }
    .val-red { color: #FF1744 !important; font-weight: 700; }
    .val-cyan { color: #00E5FF !important; font-weight: 700; }
    .val-amber { color: #FFD600 !important; font-weight: 700; }

    /* Streamlit Native Metric Fix */
    div[data-testid="stMetric"] {
        background-color: #161B22 !important;
        border: 1px solid #30363D !important;
        border-radius: 10px !important;
        padding: 14px 18px !important;
    }
    div[data-testid="stMetricLabel"] p {
        color: #94A3B8 !important;
        font-weight: 700 !important;
        font-size: 13px !important;
        text-transform: uppercase !important;
    }
    div[data-testid="stMetricValue"] div {
        color: #FFFFFF !important;
        font-weight: 800 !important;
        font-size: 24px !important;
    }

    /* Defensible Verdict Banner */
    .audit-box {
        padding: 18px 22px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-left: 5px solid;
    }
    .audit-green {
        background-color: rgba(0, 230, 118, 0.12);
        border-left-color: #00E676;
        border-top: 1px solid rgba(0, 230, 118, 0.3);
        border-right: 1px solid rgba(0, 230, 118, 0.3);
        border-bottom: 1px solid rgba(0, 230, 118, 0.3);
    }
    .audit-orange {
        background-color: rgba(255, 214, 0, 0.12);
        border-left-color: #FFD600;
        border-top: 1px solid rgba(255, 214, 0, 0.3);
        border-right: 1px solid rgba(255, 214, 0, 0.3);
        border-bottom: 1px solid rgba(255, 214, 0, 0.3);
    }
    .audit-red {
        background-color: rgba(255, 23, 68, 0.12);
        border-left-color: #FF1744;
        border-top: 1px solid rgba(255, 23, 68, 0.3);
        border-right: 1px solid rgba(255, 23, 68, 0.3);
        border-bottom: 1px solid rgba(255, 23, 68, 0.3);
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #161B22;
        padding: 8px;
        border-radius: 10px;
        border: 1px solid #30363D;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        color: #94A3B8 !important;
        padding: 8px 18px;
        font-weight: 600;
        font-size: 14px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #21262D !important;
        color: #00E5FF !important;
        font-weight: 800;
        border-bottom: 2px solid #00E5FF !important;
    }

    /* Alert / Notification styling */
    div[data-testid="stNotification"], .stAlert {
        background-color: #161B22 !important;
        border: 1px solid #38BDF8 !important;
        color: #F8FAFC !important;
        border-radius: 8px !important;
    }
    .stAlert p {
        color: #F8FAFC !important;
        font-weight: 500 !important;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 700 !important;
        letter-spacing: 0.3px;
        transition: all 0.2s ease-in-out;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 229, 255, 0.25);
    }

    /* Dataframe table */
    .stDataFrame {
        border-radius: 8px;
        border: 1px solid #30363D;
        background-color: #161B22;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if "data_pipeline" not in st.session_state:
    st.session_state.data_pipeline = DataPipeline()
if "paper_engine" not in st.session_state:
    st.session_state.paper_engine = PaperTradingEngine()
if "last_results" not in st.session_state:
    st.session_state.last_results = None
if "last_vol_results" not in st.session_state:
    st.session_state.last_vol_results = None


# ==========================================
# SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.markdown("### ⚙️ **Research Config**")
    
    ticker_choice = st.selectbox(
        "Asset / Ticker",
        ["SPY", "QQQ", "AAPL", "NVDA", "BTC-USD", "GLD", "SYNTHETIC_CRASH_SERIES"],
        index=0,
        help="Select a market benchmark, stock, crypto, or synthetic crisis dataset."
    )

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        start_year = st.selectbox("Start Year", [2016, 2018, 2020, 2021], index=1)
    with col_s2:
        end_year = st.selectbox("End Year", [2023, 2024, 2025, 2026], index=2)

    st.markdown("---")
    st.markdown("### 🤖 **Model & Validation**")
    model_choice = st.selectbox(
        "ML Model Architecture",
        ["Random Forest", "XGBoost (Boosted Trees)", "PyTorch LSTM (Deep Learning)"],
        index=0
    )
    model_type_key = "rf" if "Random" in model_choice else ("xgb" if "XGBoost" in model_choice else "lstm")

    col_w1, col_w2 = st.columns(2)
    with col_w1:
        train_window = st.number_input("IS Train Bars", min_value=150, max_value=1500, value=500, step=50, help="~2 years daily bars")
    with col_w2:
        test_window = st.number_input("OOS Test Bars", min_value=30, max_value=500, value=125, step=25, help="~6 months walk-forward test")

    expanding_mode = st.checkbox("Expanding Window (vs Rolling)", value=False)

    st.markdown("---")
    st.markdown("### 🛡️ **Risk & Frictions**")
    sizing_method = st.selectbox(
        "Position Sizing",
        ["Volatility Targeting (15% Target)", "Fractional Kelly Criterion", "Fixed 100% Allocation"],
        index=0
    )
    sizing_key = "vol_target" if "Volatility" in sizing_method else ("kelly" if "Kelly" in sizing_method else "fixed")

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        stop_loss_pct = st.slider("Stop-Loss %", min_value=1.0, max_value=10.0, value=3.5, step=0.5) / 100.0
        tx_cost_bps = st.number_input("Fee (bps)", min_value=0.0, max_value=50.0, value=10.0, step=1.0)
    with col_r2:
        trailing_stop_pct = st.slider("Trailing Stop %", min_value=2.0, max_value=15.0, value=5.0, step=0.5) / 100.0
        slippage_bps = st.number_input("Slippage (bps)", min_value=0.0, max_value=50.0, value=5.0, step=1.0)

    kill_switch_dd = st.slider("Max DD Kill-Switch %", min_value=5.0, max_value=25.0, value=12.0, step=1.0) / 100.0

    st.markdown("---")
    run_btn = st.button("🚀 Run Honest ML Audit", type="primary", use_container_width=True)


# ==========================================
# MAIN HEADER
# ==========================================
st.markdown("""
<div class="main-header">
    <div class="main-title">⚖️ HonestQuant: ML Trading Research & Failure-Mode Platform</div>
    <div class="sub-title">A Defensible Quantitative Research Environment • Rigorous Walk-Forward Validation • Overfitting Audits • Regime Autopsy • Volatility Duel</div>
</div>
""", unsafe_allow_html=True)


# ==========================================
# EXECUTION CONTROLLER
# ==========================================
if run_btn or st.session_state.last_results is None:
    with st.spinner(f"Fetching {ticker_choice} data, engineering leakage-free features, and running Walk-Forward validation..."):
        # 1. Fetch data
        df_raw = st.session_state.data_pipeline.fetch_data(
            ticker=ticker_choice,
            start_date=f"{start_year}-01-01",
            end_date=f"{end_year}-01-01"
        )
        
        # 2. Engineer features
        df_feat, feat_cols, dir_col, ret_col = st.session_state.data_pipeline.engineer_features(df_raw)

        # 3. Setup risk manager
        rm = RiskManager(
            sizing_method=sizing_key,
            target_annual_vol=0.15,
            stop_loss_pct=stop_loss_pct,
            trailing_stop_pct=trailing_stop_pct,
            max_drawdown_kill=kill_switch_dd,
            transaction_cost_bps=tx_cost_bps,
            slippage_bps=slippage_bps
        )

        # 4. Run walk-forward backtest
        backtester = HonestBacktester(
            train_window=int(train_window),
            test_window=int(test_window),
            step_window=int(test_window),
            expanding=expanding_mode,
            risk_manager=rm
        )

        try:
            results = backtester.run_walk_forward(
                df_featured=df_feat,
                feature_cols=feat_cols,
                target_col_dir=dir_col,
                model_type=model_type_key
            )
            st.session_state.last_results = results
            st.session_state.last_feat_df = df_feat
            st.session_state.last_feat_cols = feat_cols
        except Exception as e:
            st.error(f"Execution Error: {e}")
            st.stop()

        # 5. Run Volatility Duel (GARCH vs ML)
        try:
            vol_res = VolatilityResearchEngine.forecast_comparison(df_feat)
            st.session_state.last_vol_results = vol_res
        except Exception as e:
            st.warning(f"Volatility engine note: {e}")

results = st.session_state.last_results
vol_results = st.session_state.last_vol_results

if results is None:
    st.info("Click 'Run Honest ML Audit' in the sidebar to execute the research pipeline.")
    st.stop()

ml_m = results["ml_metrics"]
bh_m = results["bh_metrics"]
ma_m = results["ma_metrics"]
audit = results["overfitting_report"]
oos_df = results["full_oos_df"]


# ==========================================
# APPLICATION TABS
# ==========================================
tabs = st.tabs([
    "🔬 Executive Audit",
    "🌊 Regime Autopsy",
    "📊 Walk-Forward Folds",
    "⚡ Volatility Duel (GARCH vs ML)",
    "🎲 Monte Carlo Stress Test",
    "🟢 Live Paper Trading",
    "📑 Defensible Thesis Guide"
])


# -------------------------------------------------------------
# TAB 1: EXECUTIVE AUDIT
# -------------------------------------------------------------
with tabs[0]:
    # 1. Defensible Audit Verdict Box
    v_class = f"audit-{audit['verdict_color']}"
    st.markdown(f"""
    <div class="audit-box {v_class}">
        <div style="font-size: 16px; font-weight: 700; margin-bottom: 6px;">
            {audit['verdict']}
        </div>
        <div style="font-size: 13px; color: #C9D1D9;">
            <b>In-Sample vs Out-of-Sample Haircut:</b> <span class="val-{'red' if audit['sharpe_haircut_pct'] > 50 else 'green'}">{audit['sharpe_haircut_pct']:.1f}%</span> &nbsp;|&nbsp;
            <b>In-Sample Avg Sharpe:</b> {audit['avg_is_sharpe']:.2f} &nbsp;|&nbsp;
            <b>True OOS Sharpe:</b> {audit['oos_sharpe']:.2f} &nbsp;|&nbsp;
            <b>Prob. of Backtest Overfitting (PBO):</b> <span class="val-{'red' if audit['pbo_estimate_pct'] > 40 else 'green'}">{audit['pbo_estimate_pct']:.1f}%</span> &nbsp;|&nbsp;
            <b>Returns Skew / Kurt:</b> {audit['return_skewness']:.2f} / {audit['return_kurtosis']:.2f}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Top KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Out-of-Sample Return</div>
            <div class="metric-value { 'val-green' if ml_m['raw']['total_return'] > 0 else 'val-red' }">{ml_m['Total Return']}</div>
            <div class="metric-sub">B&H: {bh_m['Total Return']} | MA: {ma_m['Total Return']}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">OOS Sharpe Ratio</div>
            <div class="metric-value val-cyan">{ml_m['Sharpe Ratio']}</div>
            <div class="metric-sub">B&H: {bh_m['Sharpe Ratio']} | MA: {ma_m['Sharpe Ratio']}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Max Drawdown</div>
            <div class="metric-value val-amber">{ml_m['Max Drawdown']}</div>
            <div class="metric-sub">B&H: {bh_m['Max Drawdown']} | MA: {ma_m['Max Drawdown']}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Win Rate / Sortino</div>
            <div class="metric-value val-green">{ml_m['Win Rate']}</div>
            <div class="metric-sub">Sortino: {ml_m['Sortino Ratio']} | Calmar: {ml_m['Calmar Ratio']}</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Friction Costs Paid</div>
            <div class="metric-value val-red">${results['risk_summary']['Total_Friction_Cost_Paid']:,.0f}</div>
            <div class="metric-sub">Circuit Breakers: {results['risk_summary']['Kill_Switch_Events']} events</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Main Plotly Equity & Drawdown Chart
    fig_equity = visualizer.create_equity_and_drawdown_chart(oos_df, results["bh_equity"], results["ma_equity"])
    st.plotly_chart(fig_equity, use_container_width=True)

    # 4. Detailed Strategy Comparison Table
    st.markdown("### 📋 **Comprehensive Strategy Benchmark Matrix**")
    comparison_table = pd.DataFrame({
        "Metric": ["Total Return", "CAGR (Annualized)", "Annualized Volatility", "Sharpe Ratio (Rf=0)", "Sortino Ratio", "Calmar Ratio", "Max Drawdown", "Win Rate", "Profit Factor"],
        "ML Strategy (Honest OOS)": [ml_m[k] for k in ["Total Return", "CAGR", "Annualized Vol", "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", "Max Drawdown", "Win Rate", "Profit Factor"]],
        "Buy & Hold Baseline": [bh_m[k] for k in ["Total Return", "CAGR", "Annualized Vol", "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", "Max Drawdown", "Win Rate", "Profit Factor"]],
        "Moving Average Crossover": [ma_m[k] for k in ["Total Return", "CAGR", "Annualized Vol", "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", "Max Drawdown", "Win Rate", "Profit Factor"]]
    })
    st.dataframe(comparison_table, use_container_width=True, hide_index=True)


# -------------------------------------------------------------
# TAB 2: REGIME AUTOPSY
# -------------------------------------------------------------
with tabs[1]:
    st.markdown("### 🌊 **Hidden Markov Model (HMM) Market Regime Autopsy**")
    st.markdown("""
    Financial markets undergo structural regime shifts between **Low Volatility (Bull Trends)**, **Moderate Volatility (Normal)**, and **High Volatility (Turbulence/Crash)**.
    Standard ML models fail because they implicitly assume data is Independent and Identically Distributed (IID).
    """)

    # Interactive Regime Timeline Chart
    fig_regime = visualizer.create_regime_timeline_chart(oos_df)
    st.plotly_chart(fig_regime, use_container_width=True)

    col_r1, col_r2 = st.columns([3, 2])
    with col_r1:
        st.markdown("#### 🔍 **Strategy Performance Broken Down by Inferred Market Regime**")
        st.dataframe(results["regime_autopsy"], use_container_width=True, hide_index=True)
    with col_r2:
        st.markdown("#### 📊 **HMM State Emission & Frequency Stats**")
        st.dataframe(results["regime_stats"], use_container_width=True, hide_index=True)


# -------------------------------------------------------------
# TAB 3: WALK-FORWARD FOLDS
# -------------------------------------------------------------
with tabs[2]:
    st.markdown("### 📊 **Walk-Forward Cross-Validation & Overfitting Diagnostics**")
    st.markdown("""
    Walk-forward validation avoids lookahead bias by training strictly on the past and testing strictly on future unseen bars before rolling forward.
    """)

    # Bar chart of IS vs OOS Accuracy
    fig_haircut = visualizer.create_is_vs_oos_haircut_chart(results["fold_summaries"])
    st.plotly_chart(fig_haircut, use_container_width=True)

    col_f1, col_f2 = st.columns([3, 2])
    with col_f1:
        st.markdown("#### 📁 **Fold-by-Fold Performance Ledger**")
        st.dataframe(results["fold_summaries"], use_container_width=True, hide_index=True)
    with col_f2:
        st.markdown("#### 🌲 **Feature Importances (Top Signal Drivers)**")
        if results["feature_importances"] is not None and not results["feature_importances"].empty:
            fig_fi = visualizer.create_feature_importance_chart(results["feature_importances"])
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info("Feature importances available for tree-based models.")


# -------------------------------------------------------------
# TAB 4: VOLATILITY DUEL (GARCH VS ML)
# -------------------------------------------------------------
with tabs[3]:
    st.markdown("### ⚡ **Volatility Forecasting Duel: Classical GARCH(1,1) vs Machine Learning**")
    st.markdown("""
    **The Defensible Thesis:** Predicting asset *returns* (direction) with ML has a low signal-to-noise ratio ($R^2 < 1\%$).
    However, **volatility is clustered and highly autocorrelated ($R^2 > 25\%$)**. Machine learning models (Gradient Boosted Trees) can capture non-linear leverage effects and regime jumps better than classical parametric GARCH.
    """)

    if vol_results is not None:
        fig_vol = visualizer.create_volatility_duel_chart(vol_results)
        st.plotly_chart(fig_vol, use_container_width=True)

        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.markdown("#### 🎯 **Out-of-Sample Forecast Accuracy Comparison**")
            m = vol_results["metrics"]
            v_df = pd.DataFrame({
                "Metric": ["RMSE (Root Mean Sq Error)", "MAE (Mean Absolute Error)", "QLIKE Loss (Quasi-Likelihood)"],
                "Classical GARCH(1,1)": [f"{m['GARCH(1,1)']['RMSE']:.4f}", f"{m['GARCH(1,1)']['MAE']:.4f}", f"{m['GARCH(1,1)']['QLIKE']:.4f}"],
                "ML Gradient Boosted Vol": [f"{m['ML Volatility Model']['RMSE']:.4f}", f"{m['ML Volatility Model']['MAE']:.4f}", f"{m['ML Volatility Model']['QLIKE']:.4f}"]
            })
            st.dataframe(v_df, use_container_width=True, hide_index=True)
        with col_v2:
            st.markdown("#### 💡 **Why ML Wins in Volatility Research**")
            st.markdown("""
            - **Asymmetry / Leverage Effect:** Negative shocks spike volatility more than positive shocks. ML naturally learns this non-linear interaction without rigid parametric assumptions.
            - **Multi-Scale Memory:** ML incorporates both fast 5-day Parkinson range volatility and slow 63-day historical trends simultaneously.
            - **QLIKE Loss:** Lower QLIKE loss proves superior variance parameter estimation under fat tails.
            """)


# -------------------------------------------------------------
# TAB 5: MONTE CARLO STRESS TEST
# -------------------------------------------------------------
with tabs[4]:
    st.markdown("### 🎲 **Monte Carlo 1-Year Forward Resampling Stress Test**")
    st.markdown("""
    To verify that the backtest wasn't just a lucky sequence of market days, we run **300 bootstrap resamplings** of Out-of-Sample daily returns to generate probabilistic confidence bands.
    """)
    fig_mc = visualizer.create_monte_carlo_fan_chart(oos_df["Net_Return"], n_sims=300, horizon_days=252)
    st.plotly_chart(fig_mc, use_container_width=True)


# -------------------------------------------------------------
# TAB 6: LIVE PAPER TRADING SIMULATOR
# -------------------------------------------------------------
with tabs[5]:
    st.markdown("### 🟢 **Live Paper Trading Simulation & Decision Explainer**")
    st.markdown("""
    Run the strategy forward step-by-step on live/out-of-sample data with virtual cash, slippage modeling, and real-time decision logging.
    """)

    paper = st.session_state.paper_engine

    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    with col_p1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Portfolio Balance</div>
            <div class="metric-value val-cyan">${paper.portfolio_value:,.2f}</div>
            <div class="metric-sub">Initial Base: ${paper.initial_balance:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_p2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Available Cash</div>
            <div class="metric-value">${paper.cash:,.2f}</div>
            <div class="metric-sub">Unallocated Capital</div>
        </div>
        """, unsafe_allow_html=True)
    with col_p3:
        holdings_val = paper.shares * (oos_df['Close'].iloc[-1] if len(oos_df)>0 else 100)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Active Holdings</div>
            <div class="metric-value val-amber">${holdings_val:,.2f}</div>
            <div class="metric-sub">Shares: {paper.shares:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_p4:
        total_pnl = paper.portfolio_value - paper.initial_balance
        total_pnl_pct = (total_pnl / paper.initial_balance) * 100
        pnl_class = "val-green" if total_pnl >= 0 else "val-red"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total PnL ($)</div>
            <div class="metric-value {pnl_class}">${total_pnl:,.2f}</div>
            <div class="metric-sub"><span class="{pnl_class}">{total_pnl_pct:+.2f}%</span> All-Time</div>
        </div>
        """, unsafe_allow_html=True)

    col_b1, col_b2 = st.columns([1, 4])
    with col_b1:
        if st.button("⏩ Simulate Next Step", type="primary", use_container_width=True):
            # Take next bar from oos_df
            step_idx = len(paper.daily_snapshots) % len(oos_df)
            current_row = oos_df.iloc[step_idx]
            
            top_feats = {}
            if results["feature_importances"] is not None and not results["feature_importances"].empty:
                for _, r in results["feature_importances"].head(3).iterrows():
                    top_feats[r["Feature"]] = round(float(r["Importance"]), 3)

            sim_res = paper.execute_live_step(
                timestamp=str(current_row.name.strftime("%Y-%m-%d")),
                current_price=float(current_row["Close"]),
                target_signal=float(current_row["Raw_Signal"]),
                model_prob=float(current_row["Probability"]),
                target_allocation=float(current_row["Position_Weight"]),
                top_features=top_feats
            )
            st.rerun()

    with col_b2:
        if st.button("🔄 Reset Paper Portfolio", use_container_width=False):
            paper.reset()
            st.rerun()

    st.markdown("#### 📜 **Live Order Ledger & Trade Executions**")
    if len(paper.trade_history) > 0:
        trades_df = pd.DataFrame(paper.trade_history).iloc[::-1]
        st.dataframe(trades_df, use_container_width=True, hide_index=True)
    else:
        st.info("No trades executed yet. Click 'Simulate Next Step' to step forward in time.")


# -------------------------------------------------------------
# TAB 7: DEFENSIBLE THESIS GUIDE
# -------------------------------------------------------------
with tabs[6]:
    st.markdown("### 📑 **How to Defend this Project to Academic & Industry Examiners**")
    st.markdown("""
    #### 🎯 The Core Thesis
    > *"Most machine learning trading claims are products of data snooping, lookahead bias, and overfitting to bull market regimes. This platform provides an audit-grade framework that proves where ML models fail, why they fail, and where ML actually adds statistical edge (volatility forecasting vs return prediction)."*

    ---

    #### 💡 Key Defense Points & Examiner Questions

    **1. Q: "Isn't the stock market a random walk? How can your ML model predict anything?"**
    * **A:** *"We acknowledge the Efficient Market Hypothesis. Directional return prediction ($R^2 < 1\%$) suffers from severe non-stationarity and low signal-to-noise. That is why our platform measures the **In-Sample vs Out-of-Sample Haircut** and calculates the **Deflated Sharpe Ratio (DSR)**. Furthermore, we show that ML achieves strong empirical edge in **volatility forecasting** where autocorrelation and clustering are high."*

    **2. Q: "How do you guarantee there is zero lookahead bias?"**
    * **A:** *"We enforce strict Walk-Forward Cross-Validation (WFCV). Feature scaling, indicator calculations, and hyperparameter fitting are performed exclusively on the In-Sample slice. In addition, trading decisions are shifted by 1 bar to simulate realistic next-open execution."*

    **3. Q: "Why did you add transaction costs and slippage?"**
    * **A:** *"High-frequency rebalancing often creates an illusion of alpha that evaporates when facing 5–10 bps of bid-ask friction. Our platform models dynamic turnover costs and market impact, showing the net executable return."*

    **4. Q: "What role does the Hidden Markov Model (HMM) play?"**
    * **A:** *"HMM acts as our **Regime Autopsy Engine**. Rather than reporting a single aggregate Sharpe ratio, we segment performance across Bull (Low Vol), Normal, and Crisis (High Vol) states. This allows quant researchers to identify structural breakdown points during market crashes."*
    """)
