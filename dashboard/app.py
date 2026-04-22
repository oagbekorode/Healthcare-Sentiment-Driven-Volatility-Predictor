import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from sqlalchemy import create_engine
from pathlib import Path

# Correct absolute path resolution
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "sentiment.db"

if not DB_PATH.exists():
    st.error(f"Database not found at {DB_PATH}. Run your ingestion scripts first.")
    st.stop()

engine = create_engine(f"sqlite:///{DB_PATH.as_posix()}")

st.title("Healthcare Sentiment Volatility Predictor")

# Safe ticker load
tickers = pd.read_sql("SELECT DISTINCT Ticker FROM prices", engine)["Ticker"].tolist()

if not tickers:
    st.warning("No ticker data found. Run ingest_prices.py first.")
    st.stop()

ticker = st.selectbox("Select ticker", tickers)

prices = pd.read_sql(
    "SELECT * FROM prices WHERE Ticker = :t", engine, params={"t": ticker}
)
sentiment = pd.read_sql(
    "SELECT * FROM sentiment WHERE Ticker = :t", engine, params={"t": ticker}
)

# Guard: need both tables to have data
if prices.empty:
    st.warning(f"No price data for {ticker}.")
    st.stop()

if sentiment.empty:
    st.warning(f"No sentiment data for {ticker}.")
    st.stop()

# Dual-axis chart
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=prices["Date"],
    y=prices["Price_Close"],
    name="Price",
    yaxis="y1",
    line=dict(color="#1f77b4", width=2)
))

fig.add_trace(go.Bar(
    x=sentiment["Date"],
    y=sentiment["Sentiment_Score"],
    name="Sentiment",
    yaxis="y2",
    opacity=0.5,
    marker_color=["green" if s > 0 else "red" for s in sentiment["Sentiment_Score"]]
))

fig.update_layout(
    title=f"{ticker} — Price vs Sentiment",
    yaxis=dict(title="Price (USD)"),
    yaxis2=dict(
        title="Sentiment Score",
        overlaying="y",
        side="right",
        range=[-1, 1]
    ),
    legend=dict(x=0, y=1.1, orientation="h"),
    hovermode="x unified"
)

st.plotly_chart(fig, width='stretch')

# Correlation metrics
# Merge on date so the two series actually align before correlating
prices["Date"] = pd.to_datetime(prices["Date"], utc=True, errors="coerce").dt.tz_localize(None)
sentiment["Date"] = pd.to_datetime(sentiment["Date"], utc=True, errors="coerce").dt.tz_localize(None)

# Drop unparsable timestamps before asof merge.
prices = prices.dropna(subset=["Date"]).copy()
sentiment = sentiment.dropna(subset=["Date"]).copy()

merged = pd.merge_asof(
    prices.sort_values("Date"),
    sentiment[["Date", "Sentiment_Score"]].sort_values("Date"),
    on="Date",
    direction="nearest"
)

pearson = merged["Price_Close"].corr(merged["Sentiment_Score"])
spearman = merged["Price_Close"].corr(merged["Sentiment_Score"], method="spearman")

col1, col2 = st.columns(2)
col1.metric("Pearson Correlation", f"{pearson:.3f}")
col2.metric("Spearman Correlation", f"{spearman:.3f}")