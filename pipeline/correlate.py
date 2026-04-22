import pandas as pd
from scipy import stats
from sqlalchemy import create_engine

engine = create_engine("sqlite:///data/sentiment.db")

prices = pd.read_sql("SELECT * FROM prices", engine)
sentiment = pd.read_sql("SELECT * FROM sentiment", engine)

# Merge on date + ticker (you'll need to normalize timestamps to the same interval)
merged = pd.merge_asof(
    prices.sort_values("Date"),
    sentiment.sort_values("Date"),
    on="Date", by="Ticker", direction="backward"
)

# Realized volatility: rolling std of log returns
merged["Log_Return"] = merged.groupby("Ticker")["Price_Close"].transform(
    lambda x: x.pct_change().apply(lambda r: r).rolling(6).std()
)

# Hypothesis test: large sentiment spike → volatility spike?
spike_mask = merged["Sentiment_Score"].abs() > 0.5
vol_with_spike = merged.loc[spike_mask, "Log_Return"].dropna()
vol_without = merged.loc[~spike_mask, "Log_Return"].dropna()

t_stat, p_value = stats.ttest_ind(vol_with_spike, vol_without)
print(f"t-stat: {t_stat:.3f}, p-value: {p_value:.4f}")

# Pearson and Spearman correlation per ticker
for ticker in merged["Ticker"].unique():
    sub = merged[merged["Ticker"] == ticker].dropna(subset=["Sentiment_Score", "Log_Return"])
    pearson = sub["Sentiment_Score"].corr(sub["Log_Return"], method="pearson")
    spearman = sub["Sentiment_Score"].corr(sub["Log_Return"], method="spearman")
    print(f"{ticker}: Pearson={pearson:.3f}, Spearman={spearman:.3f}")