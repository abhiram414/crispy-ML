"""Enterprise-grade EDA-first ensemble forecasting pipeline.

Key capabilities:
- Strict data validation and schema checks
- Lightweight EDA report persisted to disk
- Leakage-safe feature generation
- TimeSeriesSplit backtesting with fold + aggregate metrics
- Model artifact persistence for deployment handoff

Example:
    python ensemble_forecasting.py \
      --data data.csv \
      --date-col ds \
      --target-col y \
      --artifacts-dir artifacts
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

LOGGER = logging.getLogger("forecasting")


@dataclass(frozen=True)
class ForecastConfig:
    data_path: Path
    date_col: str
    target_col: str
    n_splits: int
    lags: tuple[int, ...]
    rolling_windows: tuple[int, ...]
    artifacts_dir: Path
    random_state: int


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_time_series(path: Path, date_col: str, target_col: str) -> pd.DataFrame:
    """Load and validate a univariate time-series dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    required = {date_col, target_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {sorted(missing)}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[date_col, target_col]).sort_values(date_col).reset_index(drop=True)

    if df.empty:
        raise ValueError("Dataset is empty after parsing date/target columns.")

    duplicate_dates = int(df.duplicated(subset=[date_col]).sum())
    if duplicate_dates > 0:
        raise ValueError(
            f"Detected {duplicate_dates} duplicated timestamps in `{date_col}`. "
            "Pre-aggregate or deduplicate before training."
        )

    if not df[date_col].is_monotonic_increasing:
        raise ValueError(f"`{date_col}` must be monotonic after sorting.")

    LOGGER.info("Loaded %d rows from %s", len(df), path)
    return df


def generate_eda_report(df: pd.DataFrame, date_col: str, target_col: str) -> dict:
    """Generate EDA summary as a serializable dictionary."""
    midpoint = len(df) // 2
    first_half_mean = float(df[target_col].iloc[:midpoint].mean()) if midpoint > 0 else float("nan")
    second_half_mean = float(df[target_col].iloc[midpoint:].mean())

    report = {
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
        "missing_values": {k: int(v) for k, v in df.isna().sum().items()},
        "date_range": {
            "start": df[date_col].min().isoformat(),
            "end": df[date_col].max().isoformat(),
        },
        "target_summary": {
            k: float(v)
            for k, v in df[target_col]
            .describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
            .to_dict()
            .items()
        },
        "drift_proxy": {
            "first_half_mean": first_half_mean,
            "second_half_mean": second_half_mean,
            "delta": float(second_half_mean - first_half_mean) if not np.isnan(first_half_mean) else float("nan"),
        },
    }
    return report


def build_features(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    lags: Iterable[int],
    rolling_windows: Iterable[int],
) -> pd.DataFrame:
    """Create leakage-safe lag, rolling, and calendar features."""
    feat = df.copy()

    feat["year"] = feat[date_col].dt.year
    feat["month"] = feat[date_col].dt.month
    feat["day"] = feat[date_col].dt.day
    feat["dayofweek"] = feat[date_col].dt.dayofweek
    feat["dayofyear"] = feat[date_col].dt.dayofyear
    feat["quarter"] = feat[date_col].dt.quarter
    feat["is_month_start"] = feat[date_col].dt.is_month_start.astype(int)
    feat["is_month_end"] = feat[date_col].dt.is_month_end.astype(int)

    for lag in lags:
        feat[f"lag_{lag}"] = feat[target_col].shift(lag)

    for window in rolling_windows:
        shifted = feat[target_col].shift(1)
        feat[f"roll_mean_{window}"] = shifted.rolling(window=window).mean()
        feat[f"roll_std_{window}"] = shifted.rolling(window=window).std()
        feat[f"roll_min_{window}"] = shifted.rolling(window=window).min()
        feat[f"roll_max_{window}"] = shifted.rolling(window=window).max()

    feat = feat.dropna().reset_index(drop=True)
    if feat.empty:
        raise ValueError("Feature matrix is empty after lag/rolling creation. Reduce lag/window sizes.")

    return feat


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.where(np.abs(y_true) < 1e-8, 1e-8, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def build_ensemble(random_state: int) -> VotingRegressor:
    return VotingRegressor(
        estimators=[
            ("rf", RandomForestRegressor(n_estimators=400, random_state=random_state, n_jobs=-1)),
            ("gbr", GradientBoostingRegressor(random_state=random_state)),
            ("etr", ExtraTreesRegressor(n_estimators=400, random_state=random_state, n_jobs=-1)),
        ]
    )


def backtest(df_feat: pd.DataFrame, date_col: str, target_col: str, n_splits: int, random_state: int) -> pd.DataFrame:
    """Run time-series CV and return fold-level metrics."""
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")

    X = df_feat.drop(columns=[target_col, date_col], errors="ignore")
    y = df_feat[target_col]
    tscv = TimeSeriesSplit(n_splits=n_splits)

    rows: list[dict[str, float | int]] = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        model = build_ensemble(random_state=random_state)
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        rows.append(
            {
                "fold": fold,
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(test_idx)),
                "MAE": float(mean_absolute_error(y_test, pred)),
                "RMSE": rmse(y_test.to_numpy(), pred),
                "MAPE_%": mape(y_test.to_numpy(), pred),
            }
        )

    return pd.DataFrame(rows)


def train_final_model(df_feat: pd.DataFrame, date_col: str, target_col: str, random_state: int) -> VotingRegressor:
    """Fit final model on all available engineered data."""
    X = df_feat.drop(columns=[target_col, date_col], errors="ignore")
    y = df_feat[target_col]

    model = build_ensemble(random_state=random_state)
    model.fit(X, y)
    return model


def persist_artifacts(
    config: ForecastConfig,
    eda_report: dict,
    metrics_df: pd.DataFrame,
    model: VotingRegressor,
    feature_columns: list[str],
) -> None:
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)

    (config.artifacts_dir / "eda_report.json").write_text(json.dumps(eda_report, indent=2), encoding="utf-8")
    metrics_df.to_csv(config.artifacts_dir / "cv_metrics.csv", index=False)

    aggregate = metrics_df[["MAE", "RMSE", "MAPE_%"]].mean().to_dict()
    (config.artifacts_dir / "cv_metrics_summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    with (config.artifacts_dir / "forecast_ensemble.pkl").open("wb") as f:
        pickle.dump(model, f)
    (config.artifacts_dir / "feature_columns.json").write_text(json.dumps(feature_columns, indent=2), encoding="utf-8")


def parse_args() -> ForecastConfig:
    parser = argparse.ArgumentParser(description="Enterprise-grade EDA-first ensemble forecasting")
    parser.add_argument("--data", required=True, help="Path to CSV file")
    parser.add_argument("--date-col", default="date", help="Date column name")
    parser.add_argument("--target-col", required=True, help="Target column name")
    parser.add_argument("--n-splits", type=int, default=5, help="TimeSeriesSplit folds")
    parser.add_argument("--lags", default="1,2,3,7,14,28", help="Comma-separated lag values")
    parser.add_argument("--rolling-windows", default="3,7,14", help="Comma-separated rolling window sizes")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Directory to save reports/model")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    configure_logging(verbose=args.verbose)

    lags = tuple(int(x) for x in args.lags.split(",") if x.strip())
    windows = tuple(int(x) for x in args.rolling_windows.split(",") if x.strip())

    if any(v <= 0 for v in (*lags, *windows)):
        raise ValueError("All lag and rolling window values must be positive integers.")

    return ForecastConfig(
        data_path=Path(args.data),
        date_col=args.date_col,
        target_col=args.target_col,
        n_splits=args.n_splits,
        lags=lags,
        rolling_windows=windows,
        artifacts_dir=Path(args.artifacts_dir),
        random_state=args.random_state,
    )


def main() -> None:
    config = parse_args()
    LOGGER.info("Starting EDA-first forecasting workflow")

    raw = load_time_series(config.data_path, config.date_col, config.target_col)
    eda_report = generate_eda_report(raw, config.date_col, config.target_col)

    feat = build_features(
        raw,
        date_col=config.date_col,
        target_col=config.target_col,
        lags=config.lags,
        rolling_windows=config.rolling_windows,
    )
    LOGGER.info("Feature matrix shape: %s", feat.shape)

    metrics = backtest(
        feat,
        date_col=config.date_col,
        target_col=config.target_col,
        n_splits=config.n_splits,
        random_state=config.random_state,
    )
    LOGGER.info("Fold metrics:\n%s", metrics.to_string(index=False))
    LOGGER.info("Average metrics: %s", metrics[["MAE", "RMSE", "MAPE_%"]].mean().to_dict())

    model = train_final_model(
        feat,
        date_col=config.date_col,
        target_col=config.target_col,
        random_state=config.random_state,
    )

    feature_columns = list(feat.drop(columns=[config.target_col, config.date_col], errors="ignore").columns)
    persist_artifacts(config, eda_report, metrics, model, feature_columns)
    LOGGER.info("Artifacts written to %s", config.artifacts_dir.resolve())


if __name__ == "__main__":
    main()
