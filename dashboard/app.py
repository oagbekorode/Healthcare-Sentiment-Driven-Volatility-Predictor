import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline.features import (
    add_realized_log_volatility,
    merge_prices_sentiment_nearest,
)

st.set_page_config(page_title="Healthcare Sentiment & Volatility", layout="wide")

BASE_DIR = _ROOT
DB_PATH = BASE_DIR / "data" / "sentiment.db"

if not DB_PATH.exists():
    st.error(f"Database not found at {DB_PATH}. From the repo root run: `python run_pipeline.py`")
    st.stop()

engine = create_engine(f"sqlite:///{DB_PATH.as_posix()}")

st.title("Healthcare Sentiment → Volatility")
st.caption(
    "FinBERT headline scores vs prices. Summary stats use rolling volatility of log returns "
    "(aligned to the nearest headline time per bar)."
)

tickers = pd.read_sql("SELECT DISTINCT Ticker FROM prices", engine)["Ticker"].tolist()

if not tickers:
    st.warning("No ticker data found. Run `python -m pipeline.ingest_prices` first.")
    st.stop()

ticker = st.selectbox("Ticker", sorted(tickers))

prices = pd.read_sql("SELECT * FROM prices WHERE Ticker = :t", engine, params={"t": ticker})
sentiment = pd.read_sql("SELECT * FROM sentiment WHERE Ticker = :t", engine, params={"t": ticker})

if prices.empty:
    st.warning(f"No price data for {ticker}.")
    st.stop()

if sentiment.empty:
    st.warning(f"No sentiment data for {ticker}. Run news ingest + sentiment for the full universe.")
    st.stop()

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=prices["Date"],
        y=prices["Price_Close"],
        name="Price",
        yaxis="y1",
        line=dict(color="#1f77b4", width=2),
    )
)

fig.add_trace(
    go.Bar(
        x=sentiment["Date"],
        y=sentiment["Sentiment_Score"],
        name="Sentiment",
        yaxis="y2",
        opacity=0.45,
        marker_color=["#2ca02c" if s > 0 else "#d62728" for s in sentiment["Sentiment_Score"]],
    )
)

fig.update_layout(
    title=f"{ticker} — price vs headline sentiment",
    yaxis=dict(title="Price (USD)"),
    yaxis2=dict(
        title="Sentiment (P(pos) − P(neg))",
        overlaying="y",
        side="right",
        range=[-1, 1],
    ),
    legend=dict(x=0, y=1.12, orientation="h"),
    hovermode="x unified",
)

st.plotly_chart(fig, width="stretch")

merged = merge_prices_sentiment_nearest(prices, sentiment)
merged = add_realized_log_volatility(merged, window=6)
merged_stats = merged.dropna(subset=["Sentiment_Score", "Realized_Vol"])

c1, c2, c3, c4 = st.columns(4)
if not merged_stats.empty and len(merged_stats) >= 3:
    pv = merged["Price_Close"].corr(merged["Sentiment_Score"])
    sv = merged_stats["Sentiment_Score"].corr(merged_stats["Realized_Vol"], method="pearson")
    sp = merged_stats["Sentiment_Score"].corr(merged_stats["Realized_Vol"], method="spearman")
    c1.metric("Pearson (sentiment vs realized vol)", f"{sv:.3f}")
    c2.metric("Spearman (sentiment vs realized vol)", f"{sp:.3f}")
    c3.metric("Pearson (sentiment vs price level)", f"{pv:.3f}")
    c4.metric(
        "Bars used (vol stats)",
        str(len(merged_stats)),
        help="Needs overlapping news and price history",
    )
else:
    st.info("Not enough overlapping bars to compute volatility correlations for this ticker.")
