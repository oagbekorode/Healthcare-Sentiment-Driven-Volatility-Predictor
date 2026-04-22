import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine

TICKERS = ["MRNA", "VRTX", "BNTX", "REGN", "BIIB", 
           "ALNY", "INCY", "BMRN", "SGEN", "HZNP"]

engine = create_engine("sqlite:///data/sentiment.db")

def fetch_prices(ticker, period="5d", interval="1h"):
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        group_by="column",
        multi_level_index=False,
    )

    # yfinance can still return MultiIndex columns in some versions; flatten to base names.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    df["Ticker"] = ticker
    df.reset_index(inplace=True)
    if "Datetime" in df.columns:
        df.rename(columns={"Datetime": "Date"}, inplace=True)
    df.rename(columns={"Close": "Price_Close"}, inplace=True)

    required = ["Date", "Ticker", "Price_Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns for {ticker}: {missing}")

    return df[["Date", "Ticker", "Price_Close", "Volume"]]

for i, t in enumerate(TICKERS):
    df = fetch_prices(t)
    mode = "replace" if i == 0 else "append"
    df.to_sql("prices", engine, if_exists=mode, index=False)
    print(f"Stored {len(df)} rows for {t}")