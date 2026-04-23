"""Align sentiment with prices and summarize volatility vs FinBERT scores."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy import stats

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline.db import get_engine
from pipeline.features import add_realized_log_volatility, merge_prices_sentiment_backward_by_ticker

SPIKE_THRESHOLD = 0.5


def main() -> None:
    engine = get_engine()
    prices = pd.read_sql("SELECT * FROM prices", engine)
    sentiment = pd.read_sql("SELECT * FROM sentiment", engine)

    if prices.empty or sentiment.empty:
        raise SystemExit(
            "Need non-empty `prices` and `sentiment` tables. Run the ingest scripts first."
        )

    merged = merge_prices_sentiment_backward_by_ticker(prices, sentiment)
    merged = add_realized_log_volatility(merged)

    spike_mask = merged["Sentiment_Score"].abs() > SPIKE_THRESHOLD
    vol_with = merged.loc[spike_mask, "Realized_Vol"].dropna()
    vol_without = merged.loc[~spike_mask, "Realized_Vol"].dropna()

    if len(vol_with) >= 2 and len(vol_without) >= 2:
        t_stat, p_value = stats.ttest_ind(vol_with, vol_without, equal_var=False)
        print(
            f"Welch t-test (|sentiment| > {SPIKE_THRESHOLD} vs not): "
            f"t={t_stat:.3f}, p={p_value:.4f}"
        )
    else:
        print(
            "Not enough rows for t-test after filtering; collect more overlapping news/price data."
        )

    for ticker in sorted(merged["Ticker"].dropna().unique()):
        sub = merged[merged["Ticker"] == ticker].dropna(subset=["Sentiment_Score", "Realized_Vol"])
        if len(sub) < 3:
            print(f"{ticker}: insufficient overlapping points")
            continue
        pearson = sub["Sentiment_Score"].corr(sub["Realized_Vol"], method="pearson")
        spearman = sub["Sentiment_Score"].corr(sub["Realized_Vol"], method="spearman")
        print(
            f"{ticker}: Pearson={pearson:.3f}, Spearman={spearman:.3f} "
            "(sentiment vs realized vol)"
        )


if __name__ == "__main__":
    main()
