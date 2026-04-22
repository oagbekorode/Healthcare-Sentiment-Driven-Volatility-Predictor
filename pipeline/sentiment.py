from transformers import pipeline
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("sqlite:///data/sentiment.db")

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
    result = finbert(clean_headline(text))[0]
    scores = {r["label"]: r["score"] for r in result}
    return scores["positive"] - scores["negative"]

df = pd.read_sql("SELECT * FROM news", engine)
df["Sentiment_Score"] = df["Headline"].apply(score_headline)
df.to_sql("sentiment", engine, if_exists="replace", index=False)
print("Sentiment scoring complete.")