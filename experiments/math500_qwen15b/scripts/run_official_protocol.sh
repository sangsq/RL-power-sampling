#!/usr/bin/env bash
set -euo pipefail

if [[ "${CUDA_VISIBLE_DEVICES:-}" != "1" ]]; then
  echo "Refusing to run: set CUDA_VISIBLE_DEVICES=1 so physical GPU 0 remains untouched." >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
SEED="${1:-0}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd "$ROOT"

"$PYTHON" run.py --condition base --name "base_seed${SEED}" --seed "$SEED"
"$PYTHON" run.py --condition low_temp --name "low_temp_seed${SEED}" --seed "$SEED"

for shard in 0 1 2 3 4; do
  offset=$((100 * shard))
  "$PYTHON" run.py \
    --condition power \
    --name "power_seed${SEED}_shard${shard}" \
    --offset "$offset" \
    --limit 100 \
    --seed "$SEED"
done
