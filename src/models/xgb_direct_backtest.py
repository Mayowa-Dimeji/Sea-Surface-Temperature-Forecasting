"""
Step 4.5 — Gradient Boosting (XGBoost) with rolling-origin CV

This model predicts SSTA directly from engineered features (lags, rolling means,
Fourier terms, optional indices). It mirrors the baselines' backtest: expanding
window and horizons [1,3,6] (configurable).

Artifacts written:
  • data/processed/preds_xgb_<project>.parquet
  • data/processed/metrics_xgb_<project>.csv

Usage:
    python -u src/models/xgb_direct_backtest.py --config src/config/default.yaml

Requirements: xgboost, pandas, numpy, yaml
"""
from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import numpy as np # type: ignore
import pandas as pd # type: ignore
import yaml # type: ignore
from xgboost import XGBRegressor # type: ignore


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
    return df


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred))
    sm = np.where(denom == 0, 0.0, np.abs(y_pred - y_true) / denom)
    return float(np.mean(sm) * 200)


def _backtest_xgb(df: pd.DataFrame, horizons: List[int], initial_years: int, step_months: int,
                  random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy().reset_index(drop=True)
    dates = pd.to_datetime(df["date"]).reset_index(drop=True)

    # Target and features
    if "ssta" not in df.columns:
        raise ValueError("features file must include 'ssta' as target.")
    y = df["ssta"].astype(float)
    feature_cols = [c for c in df.columns if c not in {"date", "ssta"}]

    start_idx = initial_years * 12
    if start_idx >= len(df) - max(horizons):
        raise ValueError("Not enough data for the requested initial training window.")

    rows = []

    t = start_idx
    while t + max(horizons) < len(df):
        train_idx = np.arange(0, t)
        # small validation tail for early stopping (last 12 months of train if available)
        val_len = min(12, len(train_idx) // 5) or 1
        tr_end = t - val_len
        tr_idx = np.arange(0, tr_end)
        va_idx = np.arange(tr_end, t)

        X_tr = df.loc[tr_idx, feature_cols]
        y_tr = y.loc[tr_idx]
        X_va = df.loc[va_idx, feature_cols]
        y_va = y.loc[va_idx]

        # Drop rows with NaNs in features
        tr_mask = X_tr.notna().all(axis=1)
        va_mask = X_va.notna().all(axis=1)
        X_tr, y_tr = X_tr[tr_mask], y_tr[tr_mask]
        X_va, y_va = X_va[va_mask], y_va[va_mask]

        # If too few rows, skip this cut
        if len(X_tr) < 50 or len(X_va) < 3:
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

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )

        # Forecast each horizon using the features row at target date (built with past-only lags)
        for h in horizons:
            target_idx = t + h
            if target_idx >= len(df):
                continue
            X_te = df.loc[[target_idx], feature_cols]
            if not X_te.notna().all(axis=1).iloc[0]:
                # if features contain NaN (e.g., due to windowing), skip
                continue
            y_true = float(y.iloc[target_idx])
            y_pred = float(model.predict(X_te)[0])
            rows.append({
                "date": dates.iloc[target_idx],
                "horizon": h,
                "model": "xgb_direct",
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
        # MASE relative to seasonal naive, we need that baseline; estimate quickly:
        # use y_{t} vs y_{t-12} on the same dates as preds (rough proxy)
        naive = []
        for d in p["date"]:
            # find index in df for that date
            idx = int(np.where(dates == d)[0][0])
            if idx - 12 >= 0:
                naive.append(float(y.iloc[idx - 12]))
            else:
                naive.append(float(y.iloc[max(0, idx - 1)]))
        naive = np.array(naive)
        mase = float(np.mean(np.abs(y_true - y_pred)) / (np.mean(np.abs(y_true - naive)) + 1e-12))
        metrics.append({
            "horizon": h,
            "model": "xgb_direct",
            "MAE": mae,
            "RMSE": rmse,
            "sMAPE": smape,
            "MASE": mase,
            "Skill_vs_Climo_%": np.nan,  # optional; compare offline to climo if desired
        })

    metrics = pd.DataFrame(metrics).sort_values(["horizon"]).reset_index(drop=True)
    return preds, metrics


def run(cfg_path: str) -> Tuple[str, str]:
    cfg = _load_config(cfg_path)
    project = cfg.get("project", {}).get("name", "project")
    processed_dir = cfg.get("data", {}).get("storage", {}).get("path_processed", "data/processed")

    horizons = cfg.get("forecast", {}).get("horizons", [1, 3, 6])
    cv = cfg.get("evaluation", {}).get("cv", {})
    initial_years = int(cv.get("initial_train_years", 20))
    step_months = int(cv.get("step_months", 1))

    df = _load_features(processed_dir, project)
    preds, metrics = _backtest_xgb(df, horizons, initial_years, step_months)

    preds_path = os.path.join(processed_dir, f"preds_xgb_{project}.parquet")
    metrics_path = os.path.join(processed_dir, f"metrics_xgb_{project}.csv")

    preds.to_parquet(preds_path, index=False)
    metrics.to_csv(metrics_path, index=False)

    print(f"Wrote XGB predictions: {preds.shape} -> {preds_path}")
    print(f"Wrote XGB metrics: {metrics.shape} -> {metrics_path}")
    return preds_path, metrics_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config file")
    args = ap.parse_args()
    run(args.config)
