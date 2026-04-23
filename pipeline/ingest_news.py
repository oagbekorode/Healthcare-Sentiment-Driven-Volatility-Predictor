"""Fetch headlines from GNews and merge idempotently into SQLite."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from sqlalchemy import inspect

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline.db import get_engine
from pipeline.universe import BIOTECH_UNIVERSE

load_dotenv(_ROOT / ".env")

API_KEY = os.getenv("GNEWS_API_KEY")
REQUEST_TIMEOUT = 25
MAX_ARTICLES = 15


def fetch_news(ticker: str, company_name: str) -> pd.DataFrame:
    if not API_KEY:
        raise RuntimeError(
            "GNEWS_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    url = (
        "https://gnews.io/api/v4/search"
        f"?q={requests.utils.quote(company_name)}&lang=en&token={API_KEY}&max={MAX_ARTICLES}"
    )
    resp = requests.get(url, timeout=REQUEST_TIMEOUT)
    try:
        payload = resp.json()
    except ValueError as exc:
        raise RuntimeError(f"GNews returned non-JSON (HTTP {resp.status_code})") from exc

    if resp.status_code != 200:
        err = payload.get("message", payload) if isinstance(payload, dict) else payload
        raise RuntimeError(f"GNews HTTP {resp.status_code}: {err}")

    articles = payload.get("articles") or []
    rows = []
    for a in articles:
        title = a.get("title")
        published = a.get("publishedAt")
        if not title or not published:
            continue
        rows.append({"Date": published, "Ticker": ticker, "Headline": title})
    return pd.DataFrame(rows)


def load_existing_news(engine) -> pd.DataFrame:
    if not inspect(engine).has_table("news"):
        return pd.DataFrame(columns=["Date", "Ticker", "Headline"])
    return pd.read_sql("SELECT Date, Ticker, Headline FROM news", engine)


def main() -> None:
    engine = get_engine()
    frames = [load_existing_news(engine)]
    for ticker, company in BIOTECH_UNIVERSE.items():
        print(f"Fetching news: {ticker} ({company})…")
        df = fetch_news(ticker, company)
        print(f"  -> {len(df)} articles")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["Ticker", "Headline"])
    combined = combined.drop_duplicates(subset=["Ticker", "Headline"], keep="last")
    combined.to_sql("news", engine, if_exists="replace", index=False)
    print(f"Stored {len(combined)} deduplicated news rows.")


if __name__ == "__main__":
    main()
