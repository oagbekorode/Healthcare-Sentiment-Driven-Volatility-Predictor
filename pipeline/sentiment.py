from transformers import pipeline
import pandas as pd
from sqlalchemy import create_engine
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "sentiment.db"
engine = create_engine(f"sqlite:///{DB_PATH.as_posix()}")

# Load FinBERT — downloads ~500MB on first run
finbert = pipeline("text-classification", 
                   model="ProsusAI/finbert", 
                   return_all_scores=True)

def clean_headline(text):
    # Strip common noise prefixes
    noise = ["Breaking:", "BREAKING:", "Market Watch:", "Reuters -", "UPDATE -"]
    for n in noise:
        text = text.replace(n, "").strip()
    return text[:512]  # FinBERT max token length

def score_headline(text):
    # Transformers outputs differ by version/config:
    # - all scores: [[{'label': 'positive', 'score': ...}, ...]]
    # - top label:  [{'label': 'positive', 'score': ...}]
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

tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", engine)["name"].tolist()
if "news" not in tables:
    raise ValueError(
        f"Missing table 'news' in {DB_PATH}. Run pipeline/ingest_news.py first."
    )

df = pd.read_sql("SELECT * FROM news", engine)
if df.empty:
    raise ValueError("No rows found in 'news'. Run pipeline/ingest_news.py first.")

df["Sentiment_Score"] = df["Headline"].apply(score_headline)
df.to_sql("sentiment", engine, if_exists="replace", index=False)
print("Sentiment scoring complete.")