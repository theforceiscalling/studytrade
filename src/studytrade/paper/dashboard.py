"""
StudyTrade — Paper Trading and Greeks Dashboard
=================================================

Interactive Streamlit dashboard displaying:
  - Account equity and real-time PnL
  - Portfolio Greeks exposure (Net Delta, Theta burn, Net Vega)
  - Active open positions and trade journal
  - Live trading transition safety lockout (requires >= 90 days paper validation)

Run via:
  streamlit run src/studytrade/paper/dashboard.py
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from studytrade.paper.journal import PaperJournal


def render_dashboard() -> None:
    st.set_page_config(
        page_title="StudyTrade — Options Paper Trading",
        page_icon="📈",
        layout="wide",
    )

    st.title("StudyTrade — Systematic Option Buying Framework")
    st.caption("NSE India (NIFTY / BANKNIFTY) • Real-Time Paper Trading Monitor")

    journal = PaperJournal()
    trades_df = journal.get_trades_dataframe()
    equity_df = journal.get_equity_dataframe()
    paper_days = journal.get_paper_duration_days()

    # 1. Top KPI Metric Cards
    col1, col2, col3, col4, col5 = st.columns(5)

    current_equity = equity_df["total_equity"].iloc[-1] if not equity_df.empty else 500000.0
    net_pnl = float(trades_df["net_pnl"].sum()) if not trades_df.empty else 0.0
    n_trades = len(trades_df)
    n_wins = len(trades_df[trades_df["net_pnl"] > 0]) if n_trades > 0 else 0
    win_rate = (n_wins / n_trades * 100.0) if n_trades > 0 else 0.0

    with col1:
        st.metric("Total Equity", f"₹{current_equity:,.0f}")
    with col2:
        st.metric("Net Realized P&L", f"₹{net_pnl:,.0f}", delta=f"{net_pnl:,.0f}")
    with col3:
        st.metric("Win Rate", f"{win_rate:.1f}%", f"{n_wins}/{n_trades} trades")
    with col4:
        st.metric("Total Trades", str(n_trades))
    with col5:
        st.metric("Paper Trading Days", f"{paper_days} / 90 Days")

    st.divider()

    # 2. Equity Curve Chart
    st.subheader("Performance Equity Curve")
    if not equity_df.empty:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=equity_df["timestamp"],
                y=equity_df["total_equity"],
                mode="lines",
                name="Portfolio Equity",
                line={"color": "#00C805", "width": 2},
            )
        )
        fig.update_layout(
            template="plotly_dark",
            margin={"l": 20, "r": 20, "t": 20, "b": 20},
            height=350,
            yaxis_title="Equity (INR)",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No equity history logged yet. Run paper trading iterations to populate.")

    # 3. Trade Journal Table
    st.subheader("Trade Journal")
    if not trades_df.empty:
        st.dataframe(
            trades_df[
                [
                    "entry_time",
                    "exit_time",
                    "symbol",
                    "option_type",
                    "strike",
                    "entry_price",
                    "exit_price",
                    "gross_pnl",
                    "net_pnl",
                    "total_fees",
                    "return_pct",
                    "exit_reason",
                ]
            ],
            use_container_width=True,
        )
    else:
        st.write("No closed paper trades yet.")

    st.divider()

    # 4. Mandatory Live Trading Safety Lockout Gate
    st.subheader("Live Trading Deployment Gateway")
    if paper_days < 90:
        st.warning(
            "🔒 **LIVE TRADING LOCKED**: Systematic option buying carries a high "
            "risk of capital ruin. You must observe at least **90 consecutive days** "
            "of forward paper trading results across multiple market volatility "
            "regimes before enabling real capital execution. "
            f"\n\n**Current Progress: {paper_days} of 90 days.**"
        )
        st.button("Authorize Live Broker Execution (Disabled)", disabled=True)
    else:
        st.success("✅ Minimum 90-day paper-trading criteria satisfied.")
        confirm = st.checkbox("I have audited my paper trading logs, fees, and max drawdown.")
        if st.button("Enable Live Trading Execution", disabled=not confirm):
            st.warning("Manual broker confirmation required for live order dispatch.")


if __name__ == "__main__":
    render_dashboard()
