#!/usr/bin/env bash
set -euo pipefail

CFG=${1:-src/config/default.yaml}

python -m src.data.fetch_oisst --config "$CFG" || true
python -m src.data.make_anoms --config "$CFG" || true
python -m src.data.fetch_indices --config "$CFG" || true

python -m src.features.build_features --config "$CFG" || true
python -m src.models.baselines --config "$CFG" || true
python -m src.models.gradient_boost --config "$CFG" || true

python -m src.plots.figures --config "$CFG" || true
