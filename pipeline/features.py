"""Shared time-alignment and volatility helpers for pipeline + dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd


def parse_dates(df: pd.DataFrame, col: str = "Date") -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_datetime(out[col], utc=True, errors="coerce").dt.tz_localize(None)
    return out.dropna(subset=[col])


def merge_prices_sentiment_nearest(
    prices: pd.DataFrame,
    sentiment: pd.DataFrame,
    *,
    date_col: str = "Date",
) -> pd.DataFrame:
    """Align each price bar to the nearest headline timestamp (single-ticker panels)."""
    p = parse_dates(prices, date_col).sort_values(date_col)
    s = parse_dates(sentiment, date_col).sort_values(date_col)
    cols = [c for c in sentiment.columns if c in (date_col, "Sentiment_Score")]
    return pd.merge_asof(
        p,
        s[cols].sort_values(date_col),
        on=date_col,
        direction="nearest",
    )


def merge_prices_sentiment_backward_by_ticker(
    prices: pd.DataFrame,
    sentiment: pd.DataFrame,
    *,
    date_col: str = "Date",
    ticker_col: str = "Ticker",
) -> pd.DataFrame:
    """For each price row, use the latest sentiment on or before that time (per ticker)."""
    p = parse_dates(prices, date_col).sort_values([ticker_col, date_col])
    s = parse_dates(sentiment, date_col).sort_values([ticker_col, date_col])
    return pd.merge_asof(
        p,
        s,
        on=date_col,
        by=ticker_col,
        direction="backward",
    )


def add_realized_log_volatility(
    df: pd.DataFrame,
    *,
    ticker_col: str = "Ticker",
    date_col: str = "Date",
    price_col: str = "Price_Close",
    window: int = 6,
    min_periods: int = 2,
) -> pd.DataFrame:
    """
    Rolling std of log returns per ticker — short-horizon realized volatility proxy.
    window=6 matches hourly bars (~1.5 trading days of overlap at 6h bars).
    """
    out = df.sort_values([ticker_col, date_col]).copy()
    g = out.groupby(ticker_col, group_keys=False)
    log_ret = g[price_col].transform(lambda s: np.log(s / s.shift(1)))
    out["Log_Return"] = log_ret
    out["Realized_Vol"] = g["Log_Return"].transform(
        lambda s: s.rolling(window, min_periods=min_periods).std()
    )
    return out
