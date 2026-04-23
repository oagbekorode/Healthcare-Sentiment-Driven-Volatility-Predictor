"""Fetch headlines from GNews and merge idempotently into SQLite."""

from __future__ import annotations

import os
import sys
import time
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
MAX_RETRIES = 4
BASE_RETRY_SECONDS = 4


def fetch_news(ticker: str, company_name: str) -> pd.DataFrame:
    if not API_KEY:
        raise RuntimeError(
            "GNEWS_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    url = (
        "https://gnews.io/api/v4/search"
        f"?q={requests.utils.quote(company_name)}&lang=en&token={API_KEY}&max={MAX_ARTICLES}"
    )
    last_err = "Unknown GNews error"
    for attempt in range(1, MAX_RETRIES + 1):
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        try:
            payload = resp.json()
        except ValueError as exc:
            raise RuntimeError(f"GNews returned non-JSON (HTTP {resp.status_code})") from exc

        if resp.status_code == 200:
            articles = payload.get("articles") or []
            rows = []
            for a in articles:
                title = a.get("title")
                published = a.get("publishedAt")
                if not title or not published:
                    continue
                rows.append({"Date": published, "Ticker": ticker, "Headline": title})
            return pd.DataFrame(rows)

        err = payload.get("message", payload) if isinstance(payload, dict) else payload
        last_err = f"GNews HTTP {resp.status_code}: {err}"

        # Handle burst/rate limiting by waiting and retrying.
        if resp.status_code == 429 and attempt < MAX_RETRIES:
            retry_after = resp.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                wait_seconds = int(retry_after)
            else:
                wait_seconds = BASE_RETRY_SECONDS * attempt
            print(f"  -> rate-limited, retrying in {wait_seconds}s (attempt {attempt}/{MAX_RETRIES})")
            time.sleep(wait_seconds)
            continue
        break

    raise RuntimeError(last_err)


def load_existing_news(engine) -> pd.DataFrame:
    if not inspect(engine).has_table("news"):
        return pd.DataFrame(columns=["Date", "Ticker", "Headline"])
    return pd.read_sql("SELECT Date, Ticker, Headline FROM news", engine)


def main() -> None:
    engine = get_engine()
    existing = load_existing_news(engine)
    frames = [existing]
    fetched_ok = 0
    for ticker, company in BIOTECH_UNIVERSE.items():
        print(f"Fetching news: {ticker} ({company})…")
        try:
            df = fetch_news(ticker, company)
            print(f"  -> {len(df)} articles")
            frames.append(df)
            fetched_ok += 1
        except RuntimeError as exc:
            print(f"  -> skipped ({exc})")
        # Small pacing delay to reduce burst requests.
        time.sleep(1.2)

    if fetched_ok == 0 and existing.empty:
        raise SystemExit("No news fetched and no existing news table to fall back to.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["Ticker", "Headline"])
    combined = combined.drop_duplicates(subset=["Ticker", "Headline"], keep="last")
    combined.to_sql("news", engine, if_exists="replace", index=False)
    print(f"Stored {len(combined)} deduplicated news rows.")


if __name__ == "__main__":
    main()
