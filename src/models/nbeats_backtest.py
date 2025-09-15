"""
Step 4.7 — N-BEATS (PyTorch) with Rolling-Origin CV (1, 3, 6 months)

This trains a compact, generic N-BEATS model per horizon using a fixed
backcast window of past anomalies. It mirrors the baselines' backtest:
expanding window, horizon set from config, and outputs comparable artifacts.

Artifacts:
  • data/processed/preds_nbeats_<project>.parquet
  • data/processed/metrics_nbeats_<project>.csv

Usage:
    python -u src/models/nbeats_backtest.py --config src/config/default.yaml

Requirements:
    pip install torch pandas numpy pyyaml pyarrow

Notes:
- We forecast anomalies, which are usually near-stationary. No differencing.
- We train a *separate* N-BEATS model per horizon to keep it simple and fast.
- Backcast window (W) defaults to 36 months; tweak in CONFIG section below via YAML.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np # type: ignore
import pandas as pd # type: ignore
import yaml # type: ignore


try:
    import torch # type: ignore
    from torch import nn # type: ignore
    from torch.utils.data import Dataset, DataLoader # type: ignore
except Exception as e:
    raise SystemExit("PyTorch is required. Install with: pip install torch") from e


# ---------------------------
# Config helpers
# ---------------------------

def _load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _load_series_from_features(processed_dir: str, project: str) -> pd.DataFrame:
    """Load features_<project>.parquet and return ['date','ssta'] only."""
    path = os.path.join(processed_dir, f"features_{project}.parquet")
    if not os.path.exists(path):
        # try common variant
        alt = os.path.join(processed_dir, f"features_ssta_{project}.parquet")
        if not os.path.exists(alt):
            raise FileNotFoundError(f"Missing features file: {path} (or {alt})")
        path = alt
    df = pd.read_parquet(path)
    if "date" not in df.columns or "ssta" not in df.columns:
        raise ValueError("features file must include ['date','ssta'] columns.")
    df = df[["date","ssta"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    df["ssta"] = pd.to_numeric(df["ssta"], errors="coerce").astype(float)
    return df


# ---------------------------
# Data: window maker
# ---------------------------

def make_windows(y: np.ndarray, backcast: int, horizon: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X, Y, end_idx)
    X: [N, backcast] windows ending at index i-1 (covers y[i-backcast:i])
    Y: target at i + horizon (1-step index), scalar per row
    end_idx: the index i corresponding to the window end + 0 (i.e., next available t)
    """
    y = np.asarray(y, dtype=float)
    N = len(y)
    Xs, Ys, idxs = [], [], []
    # i is the first index AFTER the window (i.e., history length)
    for i in range(backcast, N - horizon):
        Xs.append(y[i - backcast : i])
        Ys.append(y[i + horizon])
        idxs.append(i)
    if not Xs:
        return np.empty((0, backcast), float), np.empty((0,), float), np.empty((0,), int)
    return np.stack(Xs), np.array(Ys), np.array(idxs)


class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------------------
# N-BEATS (compact generic)
# ---------------------------

class NBeatsBlock(nn.Module):
    def __init__(self, backcast: int, forecast: int, hidden: int = 128, num_layers: int = 4):
        super().__init__()
        layers = []
        in_features = backcast
        for _ in range(num_layers):
            layers += [nn.Linear(in_features, hidden), nn.ReLU()]
            in_features = hidden
        self.mlp = nn.Sequential(*layers)
        self.backcast_lin = nn.Linear(hidden, backcast)
        self.forecast_lin = nn.Linear(hidden, forecast)
    def forward(self, x):
        h = self.mlp(x)
        back = self.backcast_lin(h)
        fore = self.forecast_lin(h)
        return back, fore

class NBeats(nn.Module):
    def __init__(self, backcast: int, forecast: int, hidden: int = 128, blocks: int = 3, layers_per_block: int = 4):
        super().__init__()
        self.blocks = nn.ModuleList([
            NBeatsBlock(backcast, forecast, hidden=hidden, num_layers=layers_per_block)
            for _ in range(blocks)
        ])
    def forward(self, x):
        residual = x
        forecast = torch.zeros(x.size(0), 1 if isinstance(self.blocks[0].forecast_lin, nn.Linear) else x.size(1), device=x.device)
        for b in self.blocks:
            back, fore = b(residual)
            residual = residual - back
            forecast = forecast + fore
        return forecast  # shape [B, forecast]


# ---------------------------
# Training helpers
# ---------------------------

def train_model(X_tr, y_tr, X_va, y_va, backcast: int, forecast: int, device: str = "cpu",
                epochs: int = 300, lr: float = 1e-3, batch_size: int = 64,
                hidden: int = 128, blocks: int = 3, layers_per_block: int = 4,
                patience: int = 20) -> NBeats:
    model = NBeats(backcast, forecast, hidden=hidden, blocks=blocks, layers_per_block=layers_per_block).to(device)
    train_ds = WindowDataset(X_tr, y_tr)
    val_ds = WindowDataset(X_va, y_va)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()  # MAE is robust for anomalies

    best_state = None
    best_val = float("inf")
    bad = 0

    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            yhat = model(xb)
            loss = loss_fn(yhat, yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(xb)
        train_loss /= max(1, len(train_ds))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                yhat = model(xb)
                val_loss += loss_fn(yhat, yb).item() * len(xb)
        val_loss /= max(1, len(val_ds))

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    val = np.where(denom == 0, 0.0, np.abs(y_pred - y_true) / denom)
    return float(np.mean(val) * 200)


# ---------------------------
# Backtest
# ---------------------------

def backtest_nbeats(df: pd.DataFrame, horizons: List[int], initial_years: int, step_months: int,
                    backcast: int = 36, device: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy().reset_index(drop=True)
    dates = pd.to_datetime(df["date"]).reset_index(drop=True)
    y = df["ssta"].astype(float).to_numpy()

    start_idx = initial_years * 12
    if start_idx + max(horizons) >= len(df):
        raise ValueError("Not enough data for the requested initial training window.")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = []

    # Precompute windows (global), later filter by index relative to t
    X_all = {}
    Y_all = {}
    IDX_all = {}
    for h in horizons:
        Xh, Yh, Ih = make_windows(y, backcast=backcast, horizon=h)
        X_all[h], Y_all[h], IDX_all[h] = Xh, Yh, Ih

    t = start_idx
    while t + max(horizons) < len(df):
        # For each horizon, fit a small N-BEATS on train slice and predict target
        for h in horizons:
            Xh, Yh, Ih = X_all[h], Y_all[h], IDX_all[h]
            # we can only use windows whose end index i < t
            mask_tr_all = Ih < t
            if mask_tr_all.sum() < 100:  # need enough samples
                continue
            # validation = last 12 months of available train windows
            Ih_tr_all = Ih[mask_tr_all]
            split_idx = max(0, len(Ih_tr_all) - 12)
            tr_mask = np.zeros(len(Ih_tr_all), dtype=bool)
            tr_mask[:split_idx] = True
            va_mask = ~tr_mask

            X_tr = Xh[mask_tr_all][tr_mask]
            y_tr = Yh[mask_tr_all][tr_mask]
            X_va = Xh[mask_tr_all][va_mask]
            y_va = Yh[mask_tr_all][va_mask]

            if len(X_va) < 5:
                # fall back: small random split
                rs = np.random.RandomState(42)
                idxs = rs.permutation(np.where(mask_tr_all)[0])
                cut = int(0.9 * len(idxs))
                tr_idx, va_idx = idxs[:cut], idxs[cut:]
                X_tr, y_tr = Xh[tr_idx], Yh[tr_idx]
                X_va, y_va = Xh[va_idx], Yh[va_idx]

            model = train_model(
                X_tr, y_tr, X_va, y_va,
                backcast=backcast, forecast=1,
                device=device,
                epochs=300, lr=1e-3, batch_size=64,
                hidden=128, blocks=3, layers_per_block=4,
                patience=20,
            )

            # Predict the target at date index t+h
            target_idx = t + h
            x_te_start = target_idx - backcast
            if x_te_start < 0:
                continue
            x_te = y[x_te_start:target_idx].reshape(1, -1)
            if x_te.shape[1] != backcast:
                continue
            with torch.no_grad():
                y_hat = model(torch.tensor(x_te, dtype=torch.float32, device=device)).cpu().numpy().ravel()[0]
            y_true = float(y[target_idx])

            rows.append({
                "date": dates.iloc[target_idx],
                "horizon": h,
                "model": f"nbeats(W={backcast})",
                "y_true": y_true,
                "y_pred": float(y_hat),
            })

        t += step_months

    preds = pd.DataFrame(rows).sort_values(["date","horizon"]).reset_index(drop=True)

    # Metrics per horizon
    metrics = []
    for h in sorted(set(horizons)):
        p = preds[preds["horizon"] == h]
        if p.empty:
            continue
        y_true = p["y_true"].to_numpy()
        y_pred = p["y_pred"].to_numpy()
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
        sm = smape(y_true, y_pred)
        metrics.append({
            "horizon": h,
            "model": f"nbeats(W={backcast})",
            "MAE": mae,
            "RMSE": rmse,
            "sMAPE": sm,
            "MASE": np.nan,
            "Skill_vs_Climo_%": np.nan,
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

    horizons = cfg.get("forecast", {}).get("horizons", [1,3,6])
    cv = cfg.get("evaluation", {}).get("cv", {})
    initial_years = int(cv.get("initial_train_years", 20))
    step_months = int(cv.get("step_months", 1))

    # optional nbeats params in YAML
    nbeats_cfg = cfg.get("models", {}).get("nbeats", {})
    backcast = int(nbeats_cfg.get("backcast_window", 36))

    # Load series
    df = _load_series_from_features(processed_dir, project)

    preds, metrics = backtest_nbeats(df, horizons, initial_years, step_months, backcast=backcast)

    preds_path = os.path.join(processed_dir, f"preds_nbeats_{project}.parquet")
    metrics_path = os.path.join(processed_dir, f"metrics_nbeats_{project}.csv")
    preds.to_parquet(preds_path, index=False)
    metrics.to_csv(metrics_path, index=False)

    print(f"Wrote N-BEATS predictions: {preds.shape} -> {preds_path}")
    print(f"Wrote N-BEATS metrics: {metrics.shape} -> {metrics_path}")

    return preds_path, metrics_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config file")
    args = ap.parse_args()
    run(args.config)
