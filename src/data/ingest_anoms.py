"""
Step 2 — Ingest monthly OISST subset for a small coastal box and
compute monthly SST anomalies (SSTA) vs a fixed 1991–2020 climatology.

Disk-light: downloads a tiny lat–lon subset via ERDDAP (CSV),
averages to a single box-mean time series, subtracts monthly climatology,
and writes one small Parquet/CSV file.

Usage:
    python -u src/data/ingest_anoms.py --config src/config/default.yaml

Config keys (YAML):
  data.location.lat_min/lat_max/lon_min/lon_max
  data.time.start ("1982-01-01"), data.time.end ("auto" for latest)
  data.anomaly.baseline_start=1991, baseline_end=2020
  data.storage.format=parquet|csv|csv.gz, data.storage.path_processed
  project.name (used for output filename)
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import requests
import yaml

# You can override the ERDDAP base via environment variable ERDDAP_BASE
ERDDAP_BASE = os.getenv("ERDDAP_BASE", "https://comet.nefsc.noaa.gov/erddap/griddap")

# OISST monthly SST (0.25°, ~1981–present)
DATASET_SST = "noaa_psl_2d74_d418_a6fb"
VAR_SST = "sst"

# Optional remote climatology (1991–2020 monthly means). We can also compute locally.
DATASET_CLIM = "noaa_psl_55a2_880b_1f29"
VAR_CLIM = "sst"


@dataclass
class Box:
    lat_min: float
    lat_max: float
    lon_min: float  # degrees East (can be negative for West)
    lon_max: float

    def to_0_360(self) -> "Box":
        """Convert lon range to [0, 360) convention used by this ERDDAP."""
        def conv(x: float) -> float:
            return x if x >= 0 else x + 360.0
        return Box(self.lat_min, self.lat_max, conv(self.lon_min), conv(self.lon_max))


def _build_time_selector(start_iso: str, end_iso: Optional[str]) -> str:
    """Build ERDDAP griddap time selector. If end is None or 'auto', use 'last'."""
    if end_iso and str(end_iso).lower() != "auto":
        return f"({start_iso}T00:00:00Z):1:({end_iso}T00:00:00Z)"
    return f"({start_iso}T00:00:00Z):1:last"


def _request_csv(url: str) -> pd.DataFrame:
    """Read ERDDAP CSV, skipping the units row (line 2)."""
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    # ERDDAP CSV has header row then a 'units' row; skip that
    df = pd.read_csv(io.StringIO(r.text), skiprows=[1])
    # Fallback if column names got odd
    if "time" not in df.columns:
        df = pd.read_csv(io.StringIO(r.text))
        # If the first data row is the 'UTC' units row, drop it
        if "time" in df.columns and isinstance(df["time"].iloc[0], str) and df["time"].iloc[0].upper() == "UTC":
            df = df.iloc[1:].reset_index(drop=True)
    return df


def _find_sst_col(df: pd.DataFrame) -> str:
    """Find the SST column (handles variations like 'sst (degree_C)')"""
    for c in df.columns:
        if str(c).lower().startswith("sst"):
            return c
    return df.columns[-1]


def fetch_monthly_box_mean(box: Box, start: str, end: Optional[str]) -> pd.DataFrame:
    """Fetch monthly OISST subset and return box-mean series with columns [date, sst_mean]."""
    box_e = box.to_0_360()
    time_sel = _build_time_selector(start, end)
    url = (
        f"{ERDDAP_BASE}/{DATASET_SST}.csv?{VAR_SST}"
        f"[{time_sel}]"
        f"[({box_e.lat_min}):1:({box_e.lat_max})]"
        f"[({box_e.lon_min}):1:({box_e.lon_max})]"
    )
    df = _request_csv(url)

    # Parse time and clean
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.tz_localize(None)
    df = df[df["time"].notna()]

    # Numeric SST
    sst_col = _find_sst_col(df)
    df[sst_col] = pd.to_numeric(df[sst_col], errors="coerce")
    df = df.dropna(subset=[sst_col])

    # Box mean per timestamp
    gb = df.groupby("time")[sst_col].mean().rename("sst_mean").reset_index()
    gb = gb.rename(columns={"time": "date"}).sort_values("date").reset_index(drop=True)
    return gb


def fetch_climatology_box_mean(box: Box) -> pd.DataFrame:
    """Fetch 12-month 1991–2020 climatology for the same box (may fail if ERDDAP hiccups)."""
    box_e = box.to_0_360()
    url = (
        f"{ERDDAP_BASE}/{DATASET_CLIM}.csv?{VAR_CLIM}%2Cvalid_yr_count"
        f"[(0001-01-01T00:00:00Z):1:(0001-12-01T00:00:00Z)]"
        f"[({box_e.lat_min}):1:({box_e.lat_max})]"
        f"[({box_e.lon_min}):1:({box_e.lon_max})]"
    )
    df = _request_csv(url)

    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.tz_localize(None)
    df = df[df["time"].notna()]
    df["month"] = df["time"].dt.month

    sst_col = _find_sst_col(df)
    df[sst_col] = pd.to_numeric(df[sst_col], errors="coerce")

    clim = df.groupby("month")[sst_col].mean().rename("clim1991_2020").reset_index()
    if "valid_yr_count" in df.columns:
        vyc = df.groupby("month")["valid_yr_count"].mean().rename("valid_yr_count").reset_index()
        clim = clim.merge(vyc, on="month", how="left")
    return clim


def compute_climatology_from_series(series: pd.DataFrame, start_year: int, end_year: int,
                                    sst_col: str = "sst_mean") -> pd.DataFrame:
    """Build a 12-month climatology from the box-mean monthly series for start_year..end_year."""
    s = series.copy()
    s["date"] = pd.to_datetime(s["date"], errors="coerce")
    s = s.dropna(subset=["date"])
    s["year"] = s["date"].dt.year
    s["month"] = s["date"].dt.month

    mask = (s["year"] >= start_year) & (s["year"] <= end_year)
    base = s.loc[mask, ["month", sst_col]].copy()

    base[sst_col] = pd.to_numeric(base[sst_col], errors="coerce")
    base = base.dropna(subset=[sst_col])

    clim = (
        base.groupby("month")[sst_col]
        .mean()
        .rename(f"clim{start_year}_{end_year}")
        .reset_index()
    )
    return clim


def compute_anomalies(series: pd.DataFrame, clim: pd.DataFrame, clim_col: str = "clim1991_2020") -> pd.DataFrame:
    """Return monthly SST anomalies vs a fixed climatology.
    series: ['date','sst_mean']; clim: ['month', clim_col] with 12 rows.
    """
    s = series.copy()
    s["date"] = pd.to_datetime(s["date"], errors="coerce")
    s["sst_mean"] = pd.to_numeric(s.get("sst_mean"), errors="coerce")
    s = s.dropna(subset=["date", "sst_mean"])
    s["month"] = s["date"].dt.month

    c = clim[["month", clim_col]].copy()
    c[clim_col] = pd.to_numeric(c[clim_col], errors="coerce")

    out = s.merge(c, on="month", how="left", validate="many_to_one")
    out["ssta"] = out["sst_mean"].astype("float32") - out[clim_col].astype("float32")

    out = out[["date", "sst_mean", clim_col, "ssta"]].sort_values("date").reset_index(drop=True)
    return out


def save_output(df: pd.DataFrame, out_path: str, fmt: str = "parquet") -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if fmt.lower() == "parquet":
        df.to_parquet(out_path, index=False)
    elif fmt.lower() in {"csv", "csv.gz"}:
        compression = "gzip" if fmt.lower().endswith(".gz") else None
        df.to_csv(out_path, index=False, compression=compression)
    else:
        df.to_parquet(out_path, index=False)


def run(cfg_path: str) -> str:
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    loc = cfg["data"]["location"]
    start = cfg["data"]["time"].get("start", "1982-01-01")
    end = cfg["data"]["time"].get("end", "auto")
    b_start = int(cfg["data"]["anomaly"].get("baseline_start", 1991))
    b_end = int(cfg["data"]["anomaly"].get("baseline_end", 2020))

    box = Box(
        lat_min=float(loc["lat_min"]),
        lat_max=float(loc["lat_max"]),
        lon_min=float(loc["lon_min"]),
        lon_max=float(loc["lon_max"]),
    )

    print("Downloading monthly OISST subset (box-mean)…", file=sys.stderr)
    series = fetch_monthly_box_mean(box, start, None if end == "auto" else end)

    print("Getting climatology…", file=sys.stderr)
    try:
        clim = fetch_climatology_box_mean(box)
        clim_col = "clim1991_2020"
    except Exception as e:
        print(f"Remote climatology failed ({e}); building {b_start}–{b_end} locally.", file=sys.stderr)
        clim = compute_climatology_from_series(series, b_start, b_end)
        clim = clim.rename(columns={f"clim{b_start}_{b_end}": "clim1991_2020"})
        clim_col = "clim1991_2020"

    print("Computing anomalies…", file=sys.stderr)
    ssta_full = compute_anomalies(series, clim, clim_col=clim_col)

    # Persist only the tiny artifact
    name = cfg.get("project", {}).get("name", "ssta_box")
    processed_dir = cfg["data"]["storage"].get("path_processed", "data/processed")
    fmt = cfg["data"]["storage"].get("format", "parquet")
    out_path = (
        f"{processed_dir}/ssta_{name}_monthly.parquet"
        if fmt == "parquet"
        else f"{processed_dir}/ssta_{name}_monthly.csv"
    )

    ssta_out = ssta_full[["date", "ssta"]].copy()
    ssta_out["ssta"] = ssta_out["ssta"].astype("float32")

    print(f"Writing {out_path}…", file=sys.stderr)
    save_output(ssta_out, out_path, fmt=fmt)

    # Small status preview
    with pd.option_context("display.max_rows", 5, "display.width", 120):
        print("\nFirst rows:\n", ssta_out.head().to_string(index=False), file=sys.stderr)
        print("\nLast rows:\n", ssta_out.tail().to_string(index=False), file=sys.stderr)

    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config file")
    args = ap.parse_args()
    try:
        outp = run(args.config)
        print(f"OK: wrote {outp}")
    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
        sys.exit(1)
