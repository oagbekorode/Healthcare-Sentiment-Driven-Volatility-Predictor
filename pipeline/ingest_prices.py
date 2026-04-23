"""Download intraday prices for the biotech universe into SQLite."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline.db import get_engine
from pipeline.universe import tickers


def fetch_prices(ticker: str, period: str = "5d", interval: str = "1h") -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        group_by="column",
        multi_level_index=False,
        progress=False,
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    if df.empty:
        raise ValueError(f"No price data returned for {ticker}")

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


def main() -> None:
    engine = get_engine()
    tickers_list = tickers()
    appended = 0
    for i, t in enumerate(tickers_list):
        try:
            df = fetch_prices(t)
        except Exception as exc:
            print(f"Skipping {t}: {exc}")
            continue
        mode = "replace" if appended == 0 else "append"
        df.to_sql("prices", engine, if_exists=mode, index=False)
        appended += 1
        print(f"Stored {len(df)} rows for {t}")
    if appended == 0:
        raise SystemExit("No price rows written; check tickers and network.")


if __name__ == "__main__":
    main()
