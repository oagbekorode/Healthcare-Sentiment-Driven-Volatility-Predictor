# Healthcare Sentiment Volatility Research Report

## Status
Not enough overlapping data for walk-forward evaluation yet.

## What this means
- The pipeline ingested/scored data successfully.
- Backtest windows require more overlapping bars and headlines per ticker.

## Next step
- Run this pipeline over multiple days to accumulate history.
- Optionally increase the price lookback period in `pipeline.ingest_prices`.
