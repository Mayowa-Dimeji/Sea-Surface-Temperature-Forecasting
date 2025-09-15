"""
Step 4 — Baselines + Rolling-Origin Backtesting (1, 3, 6 months)

This single script trains/evaluates three baselines:
  • climatology-by-month
  • seasonal naive (12-month seasonal)
  • SARIMA (SARIMAX without exog): order=(1,0,0), seasonal_order=(1,0,0,12)

It performs expanding-window, rolling-origin validation and writes:
  • data/processed/preds_<project>.parquet     (per-horizon forecasts)
  • data/processed/metrics_<project>.csv       (MAE, RMSE, sMAPE, MASE, skill)

Usage:
    python -u src/models/baselines_and_backtest.py --config src/config/default.yaml

Config keys used:
  project.name
  data.storage.path_processed
  evaluation.cv.initial_train_years (e.g., 20)
  evaluation.cv.step_months (e.g., 1)
  forecast.horizons (e.g., [1,3,6])

Note: We model anomalies, which are usually stationary, so SARIMA uses d=0, D=0.
"""
from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np # type: ignore
import pandas as pd # type: ignore
import yaml # type: ignore
from statsmodels.tsa.statespace.sarimax import SARIMAX # type: ignore


# ---------------------------
# Utilities
# ---------------------------

def _load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _load_features(processed_dir: str, project: str) -> pd.DataFrame:
    """Load features_<project>.parquet and return a tidy frame with ['date','ssta', ...features]."""
    path = os.path.join(processed_dir, f"features_{project}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing features file: {path}")
    df = pd.read_parquet(path)
    # ensure date ordering
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    # some earlier steps may not carry ssta; if not present, try rebuild from columns
    if "ssta" not in df.columns:
        raise ValueError("features file must include 'ssta' column (target anomalies).")
    return df


def _calendar_month_climatology(train: pd.DataFrame) -> np.ndarray:
    """Return an array[12] of monthly means from the training subset."""
    tmp = train.copy()
    tmp["m"] = tmp["date"].dt.month
    climo = tmp.groupby("m")["ssta"].mean()
    out = np.zeros(12, dtype=float)
    for m in range(1, 13):
        out[m-1] = float(climo.get(m, 0.0))
    return out


def _seasonal_naive_forecast(history: pd.Series, horizon: int) -> float:
    """Seasonal naive with period 12; assumes horizon <= 12."""
    # y_{t+h} = y_{t+h-12}; equivalently, copy value from 12 months ago
    idx = len(history) - (12 - horizon)
    if idx <= 0:
        return float(history.iloc[-1])  # fallback to last value
    return float(history.iloc[idx-1])


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred))
    # avoid division by zero
    sm = np.where(denom == 0, 0.0, np.abs(y_pred - y_true) / denom)
    return float(np.mean(sm) * 200)


def _mase(y_true: np.ndarray, y_pred: np.ndarray, y_naive: np.ndarray) -> float:
    mae_model = np.mean(np.abs(y_true - y_pred))
    mae_naive = np.mean(np.abs(y_true - y_naive))
    return float(mae_model / mae_naive) if mae_naive > 0 else math.inf


# ---------------------------
# Backtesting
# ---------------------------

def backtest_baselines(df: pd.DataFrame, horizons: List[int], initial_years: int, step_months: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Expanding-window backtest for the specified horizons.

    Returns (preds, metrics):
      preds: [date, horizon, model, y_true, y_pred]
      metrics: aggregated by [horizon, model]
    """
    df = df.copy().reset_index(drop=True)
    y = df["ssta"].astype(float).reset_index(drop=True)
    dates = pd.to_datetime(df["date"]).reset_index(drop=True)

    start_idx = initial_years * 12
    if start_idx >= len(df) - max(horizons):
        raise ValueError("Not enough data for the requested initial training window.")

    rows = []

    t = start_idx
    while t + max(horizons) < len(df):
        train = df.iloc[:t].copy()
        y_hist = train["ssta"].astype(float)
        climo = _calendar_month_climatology(train)

        # SARIMA fit once per cut
        try:
            sar_model = SARIMAX(y_hist, order=(1,0,0), seasonal_order=(1,0,0,12), trend='n', enforce_stationarity=False, enforce_invertibility=False)
            sar_res = sar_model.fit(disp=False)
            sar_fc = sar_res.get_forecast(steps=max(horizons)).predicted_mean.to_numpy()
        except Exception:
            sar_fc = np.array([y_hist.iloc[-1]] * max(horizons), dtype=float)

        for h in horizons:
            target_idx = t + h
            target_date = dates.iloc[target_idx]
            y_true = float(y.iloc[target_idx])

            # climatology by calendar month (use month of target)
            m = int(target_date.month)
            y_pred_climo = float(climo[m-1])

            # seasonal naive
            y_pred_seas = _seasonal_naive_forecast(y_hist, h)

            # sarima
            y_pred_sar = float(sar_fc[h-1])

            rows.append({
                "date": target_date,
                "horizon": h,
                "model": "climatology",
                "y_true": y_true,
                "y_pred": y_pred_climo,
            })
            rows.append({
                "date": target_date,
                "horizon": h,
                "model": "seasonal_naive",
                "y_true": y_true,
                "y_pred": y_pred_seas,
            })
            rows.append({
                "date": target_date,
                "horizon": h,
                "model": "sarima(1,0,0)(1,0,0,12)",
                "y_true": y_true,
                "y_pred": y_pred_sar,
            })

        t += step_months

    preds = pd.DataFrame(rows).sort_values(["date", "horizon", "model"]).reset_index(drop=True)

    # Build metrics
    metrics_rows = []
    for h in sorted(set(horizons)):
        p_h = preds[preds["horizon"] == h]
        # seasonal naive baseline for MASE
        base = p_h[p_h["model"] == "seasonal_naive"]["y_pred"].to_numpy()
        yv = p_h[p_h["model"] == "seasonal_naive"]["y_true"].to_numpy()
        for model in p_h["model"].unique():
            phm = p_h[p_h["model"] == model]
            y_true = phm["y_true"].to_numpy()
            y_pred = phm["y_pred"].to_numpy()
            mae = float(np.mean(np.abs(y_true - y_pred)))
            rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
            smape = _smape(y_true, y_pred)
            if model == "seasonal_naive":
                mase = 1.0
            else:
                mase = _mase(y_true, y_pred, base)
            # skill vs climatology
            mae_climo = float(np.mean(np.abs(p_h[p_h["model"] == "climatology"]["y_true"].to_numpy() -
                                              p_h[p_h["model"] == "climatology"]["y_pred"].to_numpy())))
            skill = 100.0 * (1.0 - (mae / mae_climo)) if mae_climo > 0 else 0.0
            metrics_rows.append({
                "horizon": h,
                "model": model,
                "MAE": mae,
                "RMSE": rmse,
                "sMAPE": smape,
                "MASE": mase,
                "Skill_vs_Climo_%": skill,
            })
    metrics = pd.DataFrame(metrics_rows).sort_values(["horizon", "model"]).reset_index(drop=True)

    return preds, metrics


# ---------------------------
# Main
# ---------------------------

def run(cfg_path: str) -> Tuple[str, str]:
    cfg = _load_config(cfg_path)
    project = cfg.get("project", {}).get("name", "project")
    processed_dir = cfg.get("data", {}).get("storage", {}).get("path_processed", "data/processed")

    horizons = cfg.get("forecast", {}).get("horizons", [1,3,6])
    cv = cfg.get("evaluation", {}).get("cv", {})
    initial_years = int(cv.get("initial_train_years", 20))
    step_months = int(cv.get("step_months", 1))

    df = _load_features(processed_dir, project)

    preds, metrics = backtest_baselines(df, horizons, initial_years, step_months)

    # Save artifacts
    preds_path = os.path.join(processed_dir, f"preds_{project}.parquet")
    metrics_path = os.path.join(processed_dir, f"metrics_{project}.csv")
    preds.to_parquet(preds_path, index=False)
    metrics.to_csv(metrics_path, index=False)

    print(f"Wrote predictions: {preds.shape} -> {preds_path}")
    print(f"Wrote metrics: {metrics.shape} -> {metrics_path}")

    # Small preview
    print("\nMetrics preview:\n", metrics.groupby(["horizon","model"]).head(1).to_string(index=False))

    return preds_path, metrics_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config file")
    args = ap.parse_args()
    run(args.config)
