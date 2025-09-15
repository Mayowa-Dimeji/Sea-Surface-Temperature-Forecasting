"""
Step 3 — Build features from monthly SSTA (and optional climate indices).

- Loads tiny monthly SSTA series written in Step 2
- Engineers lag features, rolling means (past-only), Fourier seasonal terms,
  simple time trend, and (optionally) lagged climate indices (e.g., ONI/MEI).
- Writes a compact features table to data/processed/features_<project>.parquet

Usage:
    python -u src/features/build_features.py --config src/config/default.yaml

Config (YAML) keys used:
    project.name
    data.storage.path_processed
    features.lags                # e.g., [1,2,3,4,5,6,12,18]
    features.rolling_means       # e.g., [3,6,12]
    features.fourier_k           # e.g., 3
    features.indices.use         # e.g., ["oni", "mei_v2"] (optional)
    features.indices.lags        # e.g., [0,1,2,3,4,5,6]
"""
from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, List, Optional, Tuple

import numpy as np # type: ignore
import pandas as pd # type: ignore
import yaml # type: ignore


# ---------------------------
# Helpers
# ---------------------------

def _load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _detect_ssta_path(processed_dir: str, project_name: str) -> str:
    """Be forgiving about the Step 2 filename (handles accidental double 'ssta_')."""
    candidates = [
        os.path.join(processed_dir, f"ssta_{project_name}_monthly.parquet"),
        os.path.join(processed_dir, f"ssta_{project_name}_monthly.csv"),
        os.path.join(processed_dir, f"ssta_ssta_{project_name}_monthly.parquet"),
        os.path.join(processed_dir, f"ssta_ssta_{project_name}_monthly.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    # fallback: glob any ssta_*_monthly.* and take the newest
    matches = sorted(glob.glob(os.path.join(processed_dir, "ssta_*_monthly.*")))
    if not matches:
        raise FileNotFoundError(
            f"Could not find SSTA file in {processed_dir}. Expected one of: {candidates}"
        )
    return matches[-1]


def _read_monthly_series(path: str) -> pd.DataFrame:
    import os
    import pandas as pd
    import numpy as np

    # 1) Load
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    # 2) Normalize column names
    cols_lower = {c: c.lower() for c in df.columns}
    df = df.rename(columns=cols_lower)

    # find/rename expected columns
    if "date" not in df.columns:
        for c in df.columns:
            if "date" in c:
                df = df.rename(columns={c: "date"})
                break
    if "ssta" not in df.columns:
        for c in df.columns:
            if c.startswith("ssta"):
                df = df.rename(columns={c: "ssta"})
                break

    if not {"date", "ssta"}.issubset(df.columns):
        raise ValueError(f"Input does not have ['date','ssta'] columns: {df.columns.tolist()}")

    # 3) Parse types
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # drop bad dates and strip tz if present
    df = df.dropna(subset=["date"]).copy()
    if getattr(df["date"].dt, "tz", None) is not None:
        df["date"] = df["date"].dt.tz_localize(None)

    df["ssta"] = pd.to_numeric(df["ssta"], errors="coerce")
    df = df.dropna(subset=["ssta"]).copy()

    # 4) Normalize to monthly and collapse duplicates safely
    # Convert to monthly Period, group by month, and take mean (handles any duplicate stamps)
    month_period = df["date"].dt.to_period("M")
    df = (
        df.assign(period=month_period)
          .groupby("period", as_index=False, sort=True)["ssta"]
          .mean()
    )
    # Back to Timestamp at month start (avoid pandas version-specific 'MS' arg)
    df["date"] = df["period"].dt.to_timestamp()  # month start by default
    df = df[["date", "ssta"]].sort_values("date").reset_index(drop=True)

    # 5) Compact dtype
    df["ssta"] = df["ssta"].astype("float32")

    return df



def _add_lag_features(df: pd.DataFrame, col: str, lags: List[int]) -> pd.DataFrame:
    out = df.copy()
    for L in sorted(set(int(x) for x in lags if int(x) > 0)):
        out[f"{col}_lag{L}"] = out[col].shift(L)
    return out


def _add_rolling_means(df: pd.DataFrame, col: str, windows: List[int]) -> pd.DataFrame:
    out = df.copy()
    # Past-only: shift by 1 before rolling
    for w in sorted(set(int(x) for x in windows if int(x) > 1)):
        out[f"{col}_rollmean{w}"] = out[col].shift(1).rolling(w, min_periods=w).mean()
    return out


def _add_fourier_terms(df: pd.DataFrame, k: int, period: int = 12) -> pd.DataFrame:
    out = df.copy()
    month = out["date"].dt.month.astype(int)
    for i in range(1, int(k) + 1):
        out[f"sin_{i}"] = np.sin(2 * np.pi * i * month / period)
        out[f"cos_{i}"] = np.cos(2 * np.pi * i * month / period)
    return out


def _add_trend(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["t"] = np.arange(len(out), dtype=np.float32)
    # Mild scaling helps some models; keep it simple here.
    out["t"] = out["t"] / max(1, out["t"].max())
    return out


def _load_all_indices(indices_dir: str) -> pd.DataFrame:
    """Load any CSV/Parquet in data/external into a single monthly frame.
    Each file should contain a date column and one numeric column; we infer the
    series name from the filename (e.g., oni.csv -> 'oni').
    """
    if not os.path.isdir(indices_dir):
        return pd.DataFrame()

    frames = []
    for path in glob.glob(os.path.join(indices_dir, "*.parquet")) + glob.glob(os.path.join(indices_dir, "*.csv")):
        name = os.path.splitext(os.path.basename(path))[0]
        # Read
        if path.endswith(".parquet"):
            tmp = pd.read_parquet(path)
        else:
            tmp = pd.read_csv(path)
        # Find date column
        date_col = None
        for c in tmp.columns:
            if "date" in c.lower() or c.lower() in {"time", "month"}:
                date_col = c
                break
        if date_col is None:
            continue
        tmp["date"] = pd.to_datetime(tmp[date_col], errors="coerce")
        tmp = tmp.dropna(subset=["date"]).copy()
        tmp["date"] = tmp["date"].dt.to_period("M").dt.to_timestamp("MS")
        # Pick a numeric col (first non-date numeric)
        num_cols = [c for c in tmp.columns if c != "date" and pd.api.types.is_numeric_dtype(tmp[c])]
        if not num_cols:
            # try coercing the first non-date column
            for c in tmp.columns:
                if c != "date":
                    tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
            num_cols = [c for c in tmp.columns if c != "date" and pd.api.types.is_numeric_dtype(tmp[c])]
        if not num_cols:
            continue
        val_col = num_cols[0]
        tmp = tmp[["date", val_col]].rename(columns={val_col: name})
        frames.append(tmp)

    if not frames:
        return pd.DataFrame()

    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="date", how="outer")
    out = out.sort_values("date").reset_index(drop=True)
    return out


def _keep_requested_indices(idx_df: pd.DataFrame, wanted: List[str]) -> pd.DataFrame:
    if idx_df.empty:
        return idx_df
    wanted = [w for w in wanted if w in idx_df.columns]
    cols = ["date"] + wanted
    return idx_df[cols].copy()


def _add_index_lags(df: pd.DataFrame, idx_df: pd.DataFrame, lags: List[int]) -> pd.DataFrame:
    if idx_df.empty:
        return df
    out = df.merge(idx_df, on="date", how="left")
    for col in [c for c in idx_df.columns if c != "date"]:
        for L in sorted(set(int(x) for x in lags if int(x) >= 0)):
            out[f"{col}_lag{L}"] = out[col].shift(L)
        # keep the contemporaneous value too (lag0); already present as col
    return out


# ---------------------------
# Main
# ---------------------------

def run(cfg_path: str) -> str:
    cfg = _load_config(cfg_path)

    project = cfg.get("project", {}).get("name", "project")
    processed_dir = cfg.get("data", {}).get("storage", {}).get("path_processed", "data/processed")
    indices_dir = cfg.get("data", {}).get("storage", {}).get("path_external", "data/external")

    # Find SSTA file
    ssta_path = _detect_ssta_path(processed_dir, project)

    # Read
    ssta = _read_monthly_series(ssta_path)  # columns: date, ssta

    # Feature params
    lags = cfg.get("features", {}).get("lags", [1,2,3,4,5,6,12,18])
    rolls = cfg.get("features", {}).get("rolling_means", [3,6,12])
    k = cfg.get("features", {}).get("fourier_k", 3)

    # Base frame
    df = ssta.copy()

    # Lags and rolling (no leakage: rolling uses shift(1))
    df = _add_lag_features(df, "ssta", lags)
    df = _add_rolling_means(df, "ssta", rolls)

    # Fourier seasonality + simple trend
    df = _add_fourier_terms(df, k)
    df = _add_trend(df)

    # Optional indices
    idx_cfg = cfg.get("features", {}).get("indices", {})
    use_indices = idx_cfg.get("use", []) or []
    idx_lags = idx_cfg.get("lags", [0,1,2,3,4,5,6])

    if use_indices:
        all_idx = _load_all_indices(indices_dir)
        idx_keep = _keep_requested_indices(all_idx, use_indices)
        df = _add_index_lags(df, idx_keep, idx_lags)

    # Drop rows with any NA created by lags/rolling (training-ready matrix)
    feature_cols = [c for c in df.columns if c != "date"]
    df_clean = df.dropna(subset=feature_cols).reset_index(drop=True)

    # Save compact artifact
    out_path = os.path.join(processed_dir, f"features_{project}.parquet")
    os.makedirs(processed_dir, exist_ok=True)
    df_clean.to_parquet(out_path, index=False)

    # Small log
    print(f"Loaded SSTA: {len(ssta)} rows from {os.path.basename(ssta_path)}")
    print(f"Wrote features: {df_clean.shape} -> {out_path}")

    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config file")
    args = ap.parse_args()
    run(args.config)
