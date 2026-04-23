import numpy as np
import pandas as pd

from pipeline.features import (
    add_realized_log_volatility,
    merge_prices_sentiment_backward_by_ticker,
    merge_prices_sentiment_nearest,
)


def test_merge_prices_sentiment_nearest():
    prices = pd.DataFrame(
        {
            "Date": ["2024-01-01 10:00:00+00:00", "2024-01-01 11:00:00+00:00"],
            "Ticker": ["AAA", "AAA"],
            "Price_Close": [100.0, 101.0],
            "Volume": [1, 1],
        }
    )
    sentiment = pd.DataFrame(
        {
            "Date": ["2024-01-01 10:15:00+00:00"],
            "Ticker": ["AAA"],
            "Headline": ["h"],
            "Sentiment_Score": [0.25],
        }
    )
    m = merge_prices_sentiment_nearest(prices, sentiment)
    assert len(m) == 2
    assert m["Sentiment_Score"].notna().all()


def test_realized_vol_positive_on_trend():
    # Stepping prices -> non-zero log returns -> rolling std > 0 after warm-up
    dates = pd.date_range("2024-01-01", periods=8, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "Ticker": ["X"] * 8,
            "Date": dates,
            "Price_Close": np.linspace(100, 107, 8),
        }
    )
    out = add_realized_log_volatility(df, window=3)
    assert out["Realized_Vol"].iloc[-1] > 0


def test_backward_merge_handles_multiple_tickers():
    prices = pd.DataFrame(
        {
            "Date": [
                "2024-01-01 10:00:00+00:00",
                "2024-01-01 10:30:00+00:00",
                "2024-01-01 11:00:00+00:00",
                "2024-01-01 11:30:00+00:00",
            ],
            "Ticker": ["AAA", "BBB", "AAA", "BBB"],
            "Price_Close": [100.0, 50.0, 101.0, 49.0],
            "Volume": [1, 1, 1, 1],
        }
    )
    sentiment = pd.DataFrame(
        {
            "Date": ["2024-01-01 09:55:00+00:00", "2024-01-01 10:25:00+00:00"],
            "Ticker": ["AAA", "BBB"],
            "Sentiment_Score": [0.2, -0.1],
            "Headline": ["a", "b"],
        }
    )
    merged = merge_prices_sentiment_backward_by_ticker(prices, sentiment)
    assert len(merged) == len(prices)
    assert "Ticker" in merged.columns
    assert merged["Sentiment_Score"].notna().all()
