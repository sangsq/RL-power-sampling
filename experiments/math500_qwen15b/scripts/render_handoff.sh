#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
export JUPYTER_CONFIG_DIR="${JUPYTER_CONFIG_DIR:-/tmp/math500-jupyter}"
export IPYTHONDIR="${IPYTHONDIR:-/tmp/math500-ipython}"

cd "$ROOT"
mkdir -p reports
export CUDA_VISIBLE_DEVICES=""
export MPLBACKEND=Agg
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/math500-matplotlib}" "$PYTHON" analyze.py >/dev/null
"$PYTHON" scripts/build_notebook.py >/dev/null
"$PYTHON" -m jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=180 notebooks/results.ipynb
"$PYTHON" -m jupyter nbconvert --to html --no-input --output results_handoff.html --output-dir reports notebooks/results.ipynb

if command -v google-chrome >/dev/null; then
  google-chrome --headless --disable-gpu --no-sandbox --no-pdf-header-footer \
    --print-to-pdf="$ROOT/reports/results_handoff.pdf" \
    "file://$ROOT/reports/results_handoff.html"
fi
