"""
Step 4.6 — STL + Gradient Boosting Hybrid with Rolling-Origin CV

Idea: Decompose the training slice with STL (period=12) → y = T + S + R.
- Fit a simple linear trend model on T to extrapolate T_{t+h}.
- Build a monthly seasonal lookup from S (mean seasonality by calendar month) for S_{t+h}.
- Train XGBoost on residuals R using engineered features (lags/rolling/Fourier/indices).
- Forecast R_{t+h} with XGB and then reconstruct: ŷ_{t+h} = T̂_{t+h} + Ŝ_{t+h} + R̂_{t+h}.

Artifacts:
  • data/processed/preds_stl_xgb_<project>.parquet
  • data/processed/metrics_stl_xgb_<project>.csv

Usage:
    python -u src/models/stl_xgb_backtest.py --config src/config/default.yaml

Requirements: statsmodels, xgboost, numpy, pandas, yaml
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np # type: ignore
import pandas as pd # type: ignore
import yaml # type: ignore
from statsmodels.tsa.seasonal import STL # type: ignore
from xgboost import XGBRegressor # type: ignore


# ---------------------------
# Helpers
# ---------------------------

def _load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _load_features(processed_dir: str, project: str) -> pd.DataFrame:
    path = os.path.join(processed_dir, f"features_{project}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing features file: {path}")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if "ssta" not in df.columns:
        raise ValueError("features file must include 'ssta' column (target anomalies).")
    return df


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred))
    sm = np.where(denom == 0, 0.0, np.abs(y_pred - y_true) / denom)
    return float(np.mean(sm) * 200)


def _fit_trend_extrapolator(t_idx: np.ndarray, trend: np.ndarray) -> Tuple[float, float]:
    """Fit a simple linear trend: trend ≈ a * t + b. Returns (a, b)."""
    a, b = np.polyfit(t_idx.astype(float), trend.astype(float), 1)
    return float(a), float(b)


def _trend_forecast(a: float, b: float, t_val: float) -> float:
    return float(a * t_val + b)


def _seasonal_lookup_by_month(dates: pd.Series, season: np.ndarray) -> Dict[int, float]:
    """Average STL seasonality by calendar month over the training slice."""
    months = pd.to_datetime(dates).dt.month.to_numpy()
    out = {}
    for m in range(1, 13):
        vals = season[months == m]
        out[m] = float(np.mean(vals)) if vals.size else 0.0
    return out


# ---------------------------
# Backtest
# ---------------------------

def backtest_stl_xgb(df: pd.DataFrame, horizons: List[int], initial_years: int, step_months: int,
                     random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy().reset_index(drop=True)
    dates = pd.to_datetime(df["date"]).reset_index(drop=True)

    y = df["ssta"].astype(float).reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in {"date", "ssta"}]

    start_idx = initial_years * 12
    if start_idx >= len(df) - max(horizons):
        raise ValueError("Not enough data for the requested initial training window.")

    rows = []

    t = start_idx
    while t + max(horizons) < len(df):
        train = df.iloc[:t].copy()
        y_hist = train["ssta"].astype(float).to_numpy()

        # --- STL on training slice only (no leakage) ---
        # period=12 for monthly; use robust to limit outlier impact
        stl = STL(y_hist, period=12, robust=True)
        res = stl.fit()
        T = res.trend
        S = res.seasonal
        R = res.resid

        # Trend extrapolator on STL trend
        t_idx = np.arange(len(T), dtype=float)
        a, b = _fit_trend_extrapolator(t_idx, T)

        # Seasonal lookup by calendar month
        season_map = _seasonal_lookup_by_month(train["date"], S)

        # Train GB on residuals (align features to residuals)
        train_feat = train.copy()
        train_feat["resid"] = R
        # Drop rows with NaNs from features/rolling
        mask = train_feat[feature_cols].notna().all(axis=1) & np.isfinite(train_feat["resid"])
        X_tr = train_feat.loc[mask, feature_cols]
        y_tr = train_feat.loc[mask, "resid"]

        if len(X_tr) < 50:
            t += step_months
            continue

        model = XGBRegressor(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=2,
        )
        model.fit(X_tr, y_tr, verbose=False)

        # Forecast for each horizon
        for h in horizons:
            target_idx = t + h
            if target_idx >= len(df):
                continue
            # Features row at target (already built with past-only windows)
            X_te = df.loc[[target_idx], feature_cols]
            if not X_te.notna().all(axis=1).iloc[0]:
                continue

            # Residual forecast
            r_hat = float(model.predict(X_te)[0])

            # Trend + season forecasts
            t_future = float(t_idx[-1] + h)  # continue the time index
            T_hat = _trend_forecast(a, b, t_future)
            m = int(pd.to_datetime(dates.iloc[target_idx]).month)
            S_hat = season_map.get(m, 0.0)

            y_pred = T_hat + S_hat + r_hat
            y_true = float(y.iloc[target_idx])

            rows.append({
                "date": dates.iloc[target_idx],
                "horizon": h,
                "model": "stl+xgb_resid",
                "y_true": y_true,
                "y_pred": y_pred,
            })

        t += step_months

    preds = pd.DataFrame(rows).sort_values(["date", "horizon"]).reset_index(drop=True)

    # Metrics per horizon
    metrics = []
    for h in sorted(set(horizons)):
        p = preds[preds["horizon"] == h]
        if p.empty:
            continue
        y_true = p["y_true"].to_numpy()
        y_pred = p["y_pred"].to_numpy()
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        smape = _smape(y_true, y_pred)
        metrics.append({
            "horizon": h,
            "model": "stl+xgb_resid",
            "MAE": mae,
            "RMSE": rmse,
            "sMAPE": smape,
            "MASE": np.nan,               # compare externally if desired
            "Skill_vs_Climo_%": np.nan,   # compare externally if desired
        })
    metrics = pd.DataFrame(metrics).sort_values(["horizon"]).reset_index(drop=True)

    return preds, metrics


# ---------------------------
# Main
# ---------------------------

def run(cfg_path: str) -> Tuple[str, str]:
    cfg = _load_config(cfg_path)
    project = cfg.get("project", {}).get("name", "project")
    processed_dir = cfg.get("data", {}).get("storage", {}).get("path_processed", "data/processed")

    horizons = cfg.get("forecast", {}).get("horizons", [1, 3, 6])
    cv = cfg.get("evaluation", {}).get("cv", {})
    initial_years = int(cv.get("initial_train_years", 20))
    step_months = int(cv.get("step_months", 1))

    df = _load_features(processed_dir, project)
    preds, metrics = backtest_stl_xgb(df, horizons, initial_years, step_months)

    preds_path = os.path.join(processed_dir, f"preds_stl_xgb_{project}.parquet")
    metrics_path = os.path.join(processed_dir, f"metrics_stl_xgb_{project}.csv")

    preds.to_parquet(preds_path, index=False)
    metrics.to_csv(metrics_path, index=False)

    print(f"Wrote STL+XGB predictions: {preds.shape} -> {preds_path}")
    print(f"Wrote STL+XGB metrics: {metrics.shape} -> {metrics_path}")
    return preds_path, metrics_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config file")
    args = ap.parse_args()
    run(args.config)
