#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
pytest -q
space-mission-vv --config configs/demo.json
printf '\nValidation complete.\nPlot: %s/results/reference/space_mission_pipeline_vv_summary.png\n' "$(pwd)"
