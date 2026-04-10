# crispy-ML

A machine learning forecasting project in Python.

## Enterprise EDA-first ensemble forecasting

The repository includes `ensemble_forecasting.py`, a production-oriented template for univariate time-series forecasting.

### What it does

1. **Validates input data**
   - Verifies required columns exist.
   - Parses timestamps/targets and drops invalid rows.
   - Enforces unique timestamps.
2. **Runs EDA first**
   - Captures schema, missingness, date range, target distribution, and drift proxy.
   - Saves report as JSON (`eda_report.json`).
3. **Builds leakage-safe features**
   - Calendar features (month, dayofweek, quarter, etc.).
   - Lag features.
   - Shifted rolling statistics.
4. **Backtests with time-series CV**
   - Uses `TimeSeriesSplit`.
   - Reports fold metrics and aggregate summary (MAE, RMSE, MAPE).
5. **Trains and saves final ensemble model**
   - `VotingRegressor` over RandomForest + GradientBoosting + ExtraTrees.
   - Saves model and feature metadata.

## Run

```bash
python ensemble_forecasting.py \
  --data your_data.csv \
  --date-col date \
  --target-col target \
  --n-splits 5 \
  --lags 1,2,3,7,14,28 \
  --rolling-windows 3,7,14 \
  --artifacts-dir artifacts
```

## Artifacts produced

- `artifacts/eda_report.json`
- `artifacts/cv_metrics.csv`
- `artifacts/cv_metrics_summary.json`
- `artifacts/forecast_ensemble.pkl`
- `artifacts/feature_columns.json`

## Expected CSV columns

- `date` (or custom `--date-col`)
- `target` (or custom `--target-col`)
