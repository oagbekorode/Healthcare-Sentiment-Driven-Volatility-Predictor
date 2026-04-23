# Healthcare Sentiment-Driven Volatility Predictor
> Predicting Biotech Price Volatility via Financial NLP and Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![FinBERT](https://img.shields.io/badge/NLP-FinBERT-purple?logo=huggingface&logoColor=white)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Dashboard Preview

![Dashboard Overview](assets/dashboard_overview.png)

*Select any biotech ticker from the dropdown to view the real-time sentiment signal 
overlaid on the intraday price chart.*

![Sentiment Chart](assets/sentiment_chart.png)

![Correlation Metrics](assets/correlation_metrics.png)

---

## Abstract

This project investigates whether NLP-derived sentiment signals extracted from 
financial news headlines can predict intraday price volatility in mid-cap 
biotech and pharmaceutical stocks. Using **FinBERT** — a transformer model 
pre-trained on financial corpora — we compute daily sentiment scores for 
10–15 biotech tickers and test the hypothesis that significant sentiment 
spikes (|ΔScore| > 0.5) precede measurable volatility events (>2% intraday 
price movement). Results are visualized in an interactive Streamlit dashboard 
with Pearson and Spearman correlation coefficients displayed per ticker.

---

## Table of Contents

- [Motivation](#motivation)
- [Tech Stack](#tech-stack)
- [Project Architecture](#project-architecture)
- [Methodology](#methodology)
- [Findings](#findings)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Limitations & Future Work](#limitations--future-work)
- [License](#license)

---

## Motivation

Biotech stocks are uniquely sensitive to news — FDA rulings, clinical trial 
results, and earnings surprises can cause 20–40% single-day moves. Unlike 
large-cap equities where price is driven by macro factors, mid-cap biotech 
price action is heavily narrative-driven, making it an ideal domain for 
NLP-powered alpha research.

Standard sentiment tools like VADER or TextBlob are trained on general-purpose 
text and frequently misclassify financial language. A phrase like *"the drug 
failed to meet its primary endpoint"* registers as neutral in VADER but is 
strongly negative in context. FinBERT, trained on financial news and SEC 
filings, handles this domain-specific language correctly.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Price data | `yfinance` |
| News headlines | `GNews API` / `NewsAPI` |
| NLP model | `ProsusAI/finbert` via HuggingFace Transformers |
| Data storage | `SQLite` via `SQLAlchemy` |
| Statistical analysis | `pandas`, `scipy`, `scikit-learn` |
| Dashboard | `Streamlit` + `Plotly` |

---

## Project Architecture