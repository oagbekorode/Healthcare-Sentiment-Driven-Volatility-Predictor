"""Score headlines in SQLite with FinBERT (ProsusAI/finbert)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from transformers import pipeline

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline.db import db_path, get_engine

_finbert = None


def _get_finbert():
    global _finbert
    if _finbert is None:
        _finbert = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            return_all_scores=True,
        )
    return _finbert


def clean_headline(text: str) -> str:
    noise = ["Breaking:", "BREAKING:", "Market Watch:", "Reuters -", "UPDATE -"]
    for n in noise:
        text = text.replace(n, "").strip()
    return text[:512]


def score_headline(text: str) -> float:
    finbert = _get_finbert()
    raw = finbert(clean_headline(str(text)))
    if not raw:
        return 0.0

    first = raw[0]

    if isinstance(first, dict) and "label" in first and "score" in first:
        label = str(first["label"]).lower()
        score = float(first["score"])
        if label == "positive":
            return score
        if label == "negative":
            return -score
        return 0.0

    candidates = first if isinstance(first, list) else raw
    scores = {
        str(item["label"]).lower(): float(item["score"])
        for item in candidates
        if isinstance(item, dict) and "label" in item and "score" in item
    }
    return scores.get("positive", 0.0) - scores.get("negative", 0.0)


def main() -> None:
    engine = get_engine()
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", engine)["name"].tolist()
    if "news" not in tables:
        raise ValueError(f"Missing table 'news' in {db_path()}. Run pipeline.ingest_news first.")

    df = pd.read_sql("SELECT * FROM news", engine)
    if df.empty:
        raise ValueError("No rows found in 'news'. Run pipeline.ingest_news first.")

    n = len(df)
    scores = []
    for i, headline in enumerate(df["Headline"].tolist(), start=1):
        scores.append(score_headline(headline))
        if i % 20 == 0 or i == n:
            print(f"Scored {i}/{n} headlines…")

    df["Sentiment_Score"] = scores
    df.to_sql("sentiment", engine, if_exists="replace", index=False)
    print("Sentiment scoring complete.")


if __name__ == "__main__":
    main()
