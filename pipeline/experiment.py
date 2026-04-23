"""Research loop: feature build -> walk-forward backtest -> report."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline.db import get_engine
from pipeline.features import add_realized_log_volatility, merge_prices_sentiment_backward_by_ticker

REPORT_DIR = _ROOT / "reports"
REPORT_FILE = REPORT_DIR / "portfolio_report.md"
METRICS_FILE = REPORT_DIR / "backtest_metrics.csv"
LARGE_MOVE_THRESHOLD = 0.02
DAY_BARS = 24


def walk_forward_windows(n: int, min_train: int = 20, n_splits: int = 3) -> list[tuple[int, int]]:
    if n <= min_train + 5:
        return []
    step = max(10, (n - min_train) // n_splits)
    windows = []
    train_end = min_train
    while train_end + step <= n:
        windows.append((train_end, train_end + step))
        train_end += step
    return windows


def fit_linear_predictor(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame) -> np.ndarray:
    xtr = np.c_[np.ones(len(x_train)), x_train.to_numpy(dtype=float)]
    xte = np.c_[np.ones(len(x_test)), x_test.to_numpy(dtype=float)]
    beta, *_ = np.linalg.lstsq(xtr, y_train.to_numpy(dtype=float), rcond=None)
    return xte @ beta


def safe_auc(y_true: pd.Series, y_score: pd.Series) -> float:
    y = y_true.to_numpy(dtype=int)
    s = y_score.to_numpy(dtype=float)
    pos = y.sum()
    neg = len(y) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = pd.Series(s).rank(method="average").to_numpy()
    rank_pos = ranks[y == 1].sum()
    return float((rank_pos - (pos * (pos + 1) / 2)) / (pos * neg))


def regression_metrics(y: pd.Series, pred: np.ndarray) -> dict[str, float]:
    err = y.to_numpy(dtype=float) - pred
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    corr = float(pd.Series(y).corr(pd.Series(pred)))
    return {"rmse": rmse, "mae": mae, "corr": corr}


def classification_metrics(y: pd.Series, prob: np.ndarray) -> dict[str, float]:
    p = np.clip(prob, 0.0, 1.0)
    yv = y.to_numpy(dtype=int)
    cutoff = float(np.mean(yv))
    pred = (p >= cutoff).astype(int)
    tp = int(((pred == 1) & (yv == 1)).sum())
    fp = int(((pred == 1) & (yv == 0)).sum())
    fn = int(((pred == 0) & (yv == 1)).sum())
    tn = int(((pred == 0) & (yv == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    accuracy = (tp + tn) / len(yv) if len(yv) else float("nan")
    brier = float(np.mean((p - yv) ** 2))
    auc = safe_auc(pd.Series(yv), pd.Series(p))
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "brier": brier,
        "auc": auc,
    }


def build_dataset(prices: pd.DataFrame, sentiment: pd.DataFrame) -> pd.DataFrame:
    merged = merge_prices_sentiment_backward_by_ticker(prices, sentiment)
    merged = add_realized_log_volatility(merged)
    g = merged.groupby("Ticker", group_keys=False)
    merged["Sentiment_Score"] = merged["Sentiment_Score"].fillna(0.0)
    merged["Sentiment_Abs"] = merged["Sentiment_Score"].abs()
    merged["Lag_Realized_Vol"] = g["Realized_Vol"].shift(1)
    merged["RollMean_Realized_Vol"] = (
        g["Realized_Vol"].transform(lambda s: s.rolling(6, min_periods=2).mean()).shift(1)
    )
    merged["Next_Bar_Realized_Vol"] = g["Realized_Vol"].shift(-1)
    merged["Future_Return_1D"] = g["Price_Close"].shift(-DAY_BARS) / merged["Price_Close"] - 1.0
    merged["Next_Day_Large_Move"] = (
        merged["Future_Return_1D"].abs() > LARGE_MOVE_THRESHOLD
    ).astype(int)
    return merged.sort_values(["Ticker", "Date"]).reset_index(drop=True)


def evaluate_ticker(df_ticker: pd.DataFrame, rng_seed: int = 42) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    feature_cols = ["Sentiment_Score", "Sentiment_Abs", "Lag_Realized_Vol", "RollMean_Realized_Vol"]
    work = df_ticker.dropna(subset=feature_cols + ["Next_Bar_Realized_Vol"]).copy()
    windows = walk_forward_windows(len(work), min_train=20, n_splits=3)
    if not windows:
        return rows

    for split_id, (train_end, test_end) in enumerate(windows, start=1):
        tr = work.iloc[:train_end]
        te = work.iloc[train_end:test_end]
        xtr = tr[feature_cols]
        xte = te[feature_cols]
        ytr = tr["Next_Bar_Realized_Vol"]
        yte = te["Next_Bar_Realized_Vol"]

        reg_preds = {
            "lag_vol": te["Lag_Realized_Vol"].to_numpy(),
            "roll_mean_vol": te["RollMean_Realized_Vol"].to_numpy(),
            "train_mean": np.full(len(te), float(ytr.mean())),
            "finbert_linear": np.clip(fit_linear_predictor(xtr, ytr, xte), 0.0, None),
        }
        for model, pred in reg_preds.items():
            for metric, value in regression_metrics(yte, pred).items():
                rows.append(
                    {
                        "ticker": str(te["Ticker"].iloc[0]),
                        "split": split_id,
                        "task": "regression",
                        "model": model,
                        "metric": metric,
                        "value": value,
                    }
                )

        cls = work.dropna(subset=feature_cols + ["Next_Day_Large_Move"]).copy()
        trc = cls.iloc[:train_end]
        tec = cls.iloc[train_end:test_end]
        if len(tec) < 5 or trc["Next_Day_Large_Move"].nunique() < 2:
            continue

        xtrc = trc[feature_cols]
        xtec = tec[feature_cols]
        ytrc = trc["Next_Day_Large_Move"]
        ytec = tec["Next_Day_Large_Move"]

        lv = tec["Lag_Realized_Vol"].to_numpy(dtype=float)
        lv_min, lv_max = float(trc["Lag_Realized_Vol"].min()), float(trc["Lag_Realized_Vol"].max())
        lv_prob = np.zeros_like(lv) if lv_max <= lv_min else (lv - lv_min) / (lv_max - lv_min)

        cls_probs = {
            "random": np.random.default_rng(rng_seed + split_id).uniform(0.0, 1.0, len(tec)),
            "lag_vol_scaled": np.clip(lv_prob, 0.0, 1.0),
            "train_prevalence": np.full(len(tec), float(ytrc.mean())),
            "finbert_linear_prob": np.clip(fit_linear_predictor(xtrc, ytrc, xtec), 0.0, 1.0),
        }
        for model, prob in cls_probs.items():
            for metric, value in classification_metrics(ytec, prob).items():
                rows.append(
                    {
                        "ticker": str(tec["Ticker"].iloc[0]),
                        "split": split_id,
                        "task": "classification",
                        "model": model,
                        "metric": metric,
                        "value": value,
                    }
                )
    return rows


def build_report_text(summary: pd.DataFrame, used_tickers: int, total_rows: int) -> str:
    reg = summary[summary["task"] == "regression"].pivot_table(
        index="model", columns="metric", values="value"
    )
    cls = summary[summary["task"] == "classification"].pivot_table(
        index="model", columns="metric", values="value"
    )
    reg_leader = reg["rmse"].sort_values().index[0] if "rmse" in reg else "n/a"
    cls_leader = cls["auc"].sort_values(ascending=False).index[0] if "auc" in cls else "n/a"

    return (
        "# Healthcare Sentiment Volatility Research Report\n\n"
        "## Problem\n"
        "Can NLP sentiment from biotech headlines add predictive value for short-horizon volatility and next-day large-move risk?\n\n"
        "## Method\n"
        "- Build aligned sentiment/price panel with `merge_asof` and realized volatility from rolling log returns.\n"
        "- Targets: `Next_Bar_Realized_Vol` (regression) and `Next_Day_Large_Move` (classification, |return| > 2%).\n"
        "- Validation: expanding walk-forward windows per ticker (time-aware, out-of-sample only).\n"
        "- Compare FinBERT-informed linear model vs naive baselines (`lag_vol`, rolling mean, train mean/prevalence, random).\n\n"
        "## Results\n"
        f"- Rows evaluated: **{total_rows}** across **{used_tickers}** tickers.\n"
        f"- Best regression model by RMSE: **{reg_leader}**.\n"
        f"- Best classification model by AUC: **{cls_leader}**.\n\n"
        "### Aggregated Backtest Metrics (mean over tickers/splits)\n\n"
        f"{summary.to_markdown(index=False)}\n\n"
        "## Limitations\n"
        "- Free-tier news rate limits can reduce headline coverage.\n"
        "- Linear models are strong baselines but may underfit nonlinear event effects.\n"
        "- Intraday bars differ across tickers and trading sessions, adding noise.\n\n"
        "## Future Work\n"
        "- Add event type features (FDA decision, trial data, earnings).\n"
        "- Evaluate tree-based models and probabilistic calibration.\n"
        "- Add confidence intervals via block bootstrap per ticker.\n"
    )


def main() -> None:
    engine = get_engine()
    prices = pd.read_sql("SELECT * FROM prices", engine)
    sentiment = pd.read_sql("SELECT * FROM sentiment", engine)
    if prices.empty or sentiment.empty:
        raise SystemExit("Need prices and sentiment tables populated before running experiment.")

    dataset = build_dataset(prices, sentiment)
    all_rows: list[dict[str, object]] = []
    for ticker in sorted(dataset["Ticker"].dropna().unique()):
        all_rows.extend(evaluate_ticker(dataset[dataset["Ticker"] == ticker]))

    if not all_rows:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_FILE.write_text(
            (
                "# Healthcare Sentiment Volatility Research Report\n\n"
                "## Status\n"
                "Not enough overlapping data for walk-forward evaluation yet.\n\n"
                "## What this means\n"
                "- The pipeline ingested/scored data successfully.\n"
                "- Backtest windows require more overlapping bars and headlines per ticker.\n\n"
                "## Next step\n"
                "- Run this pipeline over multiple days to accumulate history.\n"
                "- Optionally increase the price lookback period in `pipeline.ingest_prices`.\n"
            ),
            encoding="utf-8",
        )
        pd.DataFrame(
            columns=["ticker", "split", "task", "model", "metric", "value"]
        ).to_csv(METRICS_FILE, index=False)
        print("Not enough data for walk-forward evaluation yet.")
        print(f"Wrote placeholder metrics: {METRICS_FILE}")
        print(f"Wrote placeholder report:  {REPORT_FILE}")
        return

    metrics = pd.DataFrame(all_rows)
    summary = (
        metrics.groupby(["task", "model", "metric"], as_index=False)["value"]
        .mean()
        .sort_values(["task", "metric", "value"], ascending=[True, True, True])
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS_FILE, index=False)
    report_text = build_report_text(
        summary=summary,
        used_tickers=int(metrics["ticker"].nunique()),
        total_rows=int(len(dataset)),
    )
    REPORT_FILE.write_text(report_text, encoding="utf-8")

    print(f"Wrote metrics: {METRICS_FILE}")
    print(f"Wrote report:  {REPORT_FILE}")


if __name__ == "__main__":
    main()

