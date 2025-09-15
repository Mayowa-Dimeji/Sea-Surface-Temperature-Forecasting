# Sea Surface Temperature Anomaly Forecasting (SSTA)

> Predict monthly sea surface temperature anomalies (SSTA) at (or near) UK ports using disk‑light, reproducible pipelines.

---

## Why this project?

Sea surface temperatures shape marine ecosystems, fisheries, shipping, and coastal risk. Forecasting **anomalies** (deviations from a fixed climatology) helps quantify warming trends and seasonal cycles and anticipate events linked to climate modes like **El Niño/La Niña**.

This repo is designed to be:

- **Informative:** clear EDA, diagnostics, and model comparisons.
- **Lean on storage:** subsetting at source, saving only tiny monthly series.
- **Reproducible:** deterministic backtesting and versioned configs.

---

## Data sources (small, reliable, global)

- **NOAA OISST v2.1 (primary)** — Global daily & monthly SST on a 0.25° grid (1981–present). We subset a small lat–lon box around a chosen UK port and use **monthly** fields to avoid storing large daily data.
- **NOAA ERSSTv5 (optional, trend check)** — Global monthly SST on a 2° grid (1854–present) for long‑term context.
- **Climate indices (optional exogenous features)** — ENSO/ONI, MEI v2, PDO, AMO, IOD (DMI), etc.

> **UK coverage:** Both OISST and ERSST are global and include UK waters; we simply subset the desired coastal box.

---

## What we model

- **Target:** Monthly **SSTA** for a specific location or small coastal box (e.g., 1°×1° around a port).
- **Climatology baseline:** Default **1991–2020** monthly means (configurable, e.g., 1981–2010).
- **Forecast horizons:** +1, +3, +6 months (configurable).

---

## Methods (in layers)

1. **Baselines:**

   - Monthly climatology (by calendar month)
   - Seasonal naïve (value from t−12)
   - SARIMA / SARIMAX (seasonal period = 12; optional exogenous indices)

2. **Hybrid ML:**

   - STL decomposition: $y_t = T_t + S_t + R_t$
   - Model residuals $R_t$ with Gradient Boosting (e.g., XGBoost/LightGBM) using lagged features.

3. **Neural (optional):**

   - Feed-forward MLP on lagged features, or N‑BEATS for univariate series.

**Features:** lags (1–18 with explicit 12), rolling means (3/6/12), Fourier seasonal terms, lagged climate indices (0–6 months) to capture teleconnections.

---

## Storage‑friendly design

- Subset **monthly** OISST directly around the port box (no daily → monthly step).
- Compute anomalies vs the fixed baseline.
- **Persist only the tiny SSTA series** (`date, ssta`), optionally plus indices → usually **<100 KB**.
- If caching, prefer **Parquet** (float32) or compressed CSV.

---

## Repository layout

```
.
├─ README.md
├─ pyproject.toml / requirements.txt
├─ .env.example
├─ data/
│  ├─ external/         # climate indices (CSV/Parquet, tiny)
│  ├─ interim/          # temporary artefacts (ignored in git)
│  └─ processed/        # ssta_<location>_monthly.parquet (tiny)
├─ notebooks/
│  ├─ 01_eda.ipynb
│  ├─ 02_features.ipynb
│  ├─ 03_baselines.ipynb
│  ├─ 04_models.ipynb
│  └─ 05_backtests.ipynb
├─ src/
│  ├─ config/
│  │  └─ default.yaml   # location, baseline, horizons, features
│  ├─ data/
│  │  ├─ fetch_oisst.py # ERDDAP subset → monthly box
│  │  ├─ make_anoms.py  # compute anomalies vs baseline
│  │  └─ fetch_indices.py
│  ├─ features/
│  │  └─ build_features.py
│  ├─ models/
│  │  ├─ baselines.py   # climatology, seasonal-naive, SARIMA(X)
│  │  ├─ gradient_boost.py
│  │  └─ nbeats.py      # optional
│  ├─ eval/
│  │  ├─ backtest.py    # rolling-origin CV
│  │  └─ metrics.py     # MAE, RMSE, sMAPE, MASE, skill vs baseline
│  └─ plots/
│     └─ figures.py     # STL, seasonal subseries, ACF, forecast plots
├─ scripts/
│  ├─ run_pipeline.sh   # convenient end-to-end runner
│  └─ make_report.py    # export plots + metrics table
└─ tests/
   └─ test_leakage_and_splits.py
```

> You can keep this structure even if some modules start as stubs—grow them as you go.

---

## Quick start

### 1) Environment

- **Python:** 3.10+
- Create and activate env (choose one):

```bash
# uv + pip (fast)
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt

# or: conda
conda create -n ssta python=3.10 -y
conda activate ssta
pip install -r requirements.txt
```

**Key deps:** `xarray`, `pandas`, `numpy`, `netCDF4`, `cfgrib` (if needed), `statsmodels`, `scikit-learn`, `xgboost`, `matplotlib`, `pyyaml`.

### 2) Configure location & baseline

Edit `src/config/default.yaml` (example):

```yaml
location:
  name: plymouth_box
  lat_min: 50.0
  lat_max: 51.0
  lon_min: -5.0
  lon_max: -4.0
baseline:
  period: [1991, 2020]
forecast:
  horizons: [1, 3, 6]
features:
  lags: [1, 2, 3, 4, 5, 6, 12, 18]
  rolling_means: [3, 6, 12]
  fourier_k: 3
indices:
  use: ["oni", "mei_v2"]
  lags: [0, 1, 2, 3, 4, 5, 6]
```

### 3) Minimal pipeline (tiny artefacts)

```bash
# Fetch monthly OISST subset for the configured box
python -m src.data.fetch_oisst --config src/config/default.yaml

# Compute anomalies vs baseline (writes processed/ssta_<name>_monthly.parquet)
python -m src.data.make_anoms --config src/config/default.yaml

# Optional: fetch climate indices (ENSO/MEI/etc.)
python -m src.data.fetch_indices --config src/config/default.yaml

# Build features & run baselines + models with rolling-origin CV
python -m src.features.build_features --config src/config/default.yaml
python -m src.models.baselines --config src/config/default.yaml
python -m src.models.gradient_boost --config src/config/default.yaml

# Export figures & metrics table
python -m src.plots.figures --config src/config/default.yaml
python scripts/make_report.py --config src/config/default.yaml
```

> By default, only tiny Parquet/CSV files (monthly SSTA + indices) are written to `data/processed/`.

---

## Evaluation

- **Backtesting:** expanding-window, rolling‑origin CV.
- **Horizons:** +1, +3, +6 months.
- **Metrics:** MAE, RMSE, sMAPE, **MASE** (relative to seasonal‑naïve), and **skill** vs climatology.
- **Uncertainty:** SARIMAX intervals or conformal prediction wrapper for ML models.

### Expected plots

- STL decomposition (trend/season/residual)
- Seasonal subseries (box/line by month)
- ACF/PACF of anomalies
- Feature importance (SHAP) for ML
- Forecast vs actuals with intervals (per horizon)

---

## Reproducibility & hygiene

- Fixed **climatology baseline** across train/test.
- Rolling features computed with **past‑only** windows (no leakage).
- Version‑pinned deps and random seeds.
- Optional: DVC/MLflow for data & experiment tracking.

---

## Results (template)

| Horizon | Model          | MAE | RMSE | sMAPE | MASE | Skill vs Climo |
| ------: | -------------- | --: | ---: | ----: | ---: | -------------: |
|   +1 mo | Seasonal naïve |     |      |       | 1.00 |           0.0% |
|   +1 mo | SARIMAX(+ENSO) |     |      |       |      |                |
|   +1 mo | STL + XGBoost  |     |      |       |      |                |
|   +3 mo | …              |     |      |       |      |                |
|   +6 mo | …              |     |      |       |      |                |

Place rendered plots in `reports/figures/` and export a short summary in `reports/README.md`.

---

## Frequently asked

**Q: Can I use a single buoy or port station?**
_A:_ Yes, if you have an in‑situ series (e.g., Cefas SmartBuoys). You can still use the same pipeline—just skip gridded subsetting.

**Q: Will daily data blow up storage?**
_A:_ We avoid it by using monthly OISST directly. If you must use daily, aggregate to monthly in memory and only persist the monthly series.

**Q: Which baseline is best?**
_A:_ 1991–2020 is standard for modern anomaly work; 1981–2010 is also common. Be consistent across train/test and when comparing sources.

---

## Contributing

- Use feature branches and conventional commit messages.
- Add or update unit tests in `tests/` for new feature engineering or split logic.
- Run linters/formatters (`ruff`, `black`) before PRs.

---

## Licence

Choose one (e.g., MIT). Add `LICENCE` file at the project root.

---

## Acknowledgements

This repository builds on open marine climate datasets provided by NOAA and related climate index providers. Thanks to the researchers, engineers, and maintainers who make these data publicly available.
