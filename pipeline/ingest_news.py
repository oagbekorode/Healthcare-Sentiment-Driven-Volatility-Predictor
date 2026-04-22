import requests, os, pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
load_dotenv()

engine = create_engine("sqlite:///data/sentiment.db")
API_KEY = os.getenv("GNEWS_API_KEY")

def fetch_news(ticker, company_name):
    url = f"https://gnews.io/api/v4/search?q={company_name}&lang=en&token={API_KEY}&max=10"
    r = requests.get(url).json()
    articles = r.get("articles", [])
    rows = [{"Date": a["publishedAt"], "Ticker": ticker, "Headline": a["title"]} 
            for a in articles]
    return pd.DataFrame(rows)

TICKER_NAMES = {
    "MRNA": "Moderna",
    "VRTX": "Vertex Pharmaceuticals",
}

for ticker, name in TICKER_NAMES.items():
    df = fetch_news(ticker, name)
    df.to_sql("news", engine, if_exists="append", index=False)