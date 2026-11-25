"""Live dashboard for the bot.

    streamlit run trader/dashboard.py
"""
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trader import config  # noqa: E402

st.set_page_config(page_title="Trader bot", layout="wide")
st.title("Trader bot — paper trading")

if not config.JOURNAL_DB.exists():
    st.info("No journal yet. Start the bot with: python -m trader.bot")
    st.stop()

db = sqlite3.connect(config.JOURNAL_DB)
universes = pd.read_sql("SELECT DISTINCT universe FROM decisions", db)["universe"].tolist() or [config.UNIVERSE]
universe = st.sidebar.selectbox("Universe", universes)
q = lambda sql: pd.read_sql(sql, db, params=(universe,))  # noqa: E731

equity = q("SELECT time, equity, cash, open_positions FROM equity WHERE universe=? ORDER BY time")
decisions = q("SELECT * FROM decisions WHERE universe=? ORDER BY time DESC LIMIT 500")
orders = q("SELECT * FROM orders WHERE universe=? ORDER BY time DESC LIMIT 200")

if not equity.empty:
    equity["time"] = pd.to_datetime(equity["time"])
    first, last = equity["equity"].iloc[0], equity["equity"].iloc[-1]
    peak = equity["equity"].cummax()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Equity", f"${last:,.0f}", f"{last / first - 1:+.2%} since start")
    c2.metric("Cash", f"${equity['cash'].iloc[-1]:,.0f}")
    c3.metric("Open positions", int(equity["open_positions"].iloc[-1]))
    c4.metric("Drawdown from peak", f"{last / peak.iloc[-1] - 1:.2%}")
    st.subheader("Equity")
    st.line_chart(equity.set_index("time")["equity"])

if not decisions.empty:
    st.subheader("Model confidence (P beats fees over next 24h)")
    d = decisions.copy()
    d["time"] = pd.to_datetime(d["time"])
    st.line_chart(d.pivot_table(index="time", columns="symbol", values="p_up"))
    st.subheader("Latest decisions")
    st.dataframe(decisions.head(50), width="stretch")

st.subheader("Orders")
st.dataframe(orders, width="stretch") if not orders.empty else st.write("No orders yet.")

report = config.REPORT_DIR / f"{universe}_equity_mlp.png"
if report.exists():
    st.subheader("Backtest (out-of-sample, after costs)")
    st.image(str(report))
