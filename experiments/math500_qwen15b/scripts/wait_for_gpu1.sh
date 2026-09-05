#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-3600}"

cd "$ROOT"
while true; do
  status="$(nvidia-smi -i 1 --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits)"
  utilization="${status%%,*}"
  memory="${status##*, }"
  now="$(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "- ${now} — GPU1 monitor: utilization=${utilization}%, memory=${memory} MiB." | tee -a notes/ACTIVITY_LOG.md
  if (( utilization < 10 && memory < 1000 )); then
    echo "- ${now} — GPU1 is free; starting validation and seed-0 official protocol." | tee -a notes/ACTIVITY_LOG.md
    break
  fi
  sleep "$INTERVAL_SECONDS"
done

export CUDA_VISIBLE_DEVICES=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True

"$PYTHON" -m pytest -q -c "$ROOT/../../pyproject.toml" "$ROOT/../../tests/test_math500.py"
"$PYTHON" run.py \
  --condition power --name validation_identity --offset 0 --limit 8 --seed 31415 \
  --alpha 1 --proposal-temperature 1 --mcmc-steps 2 --max-tokens 192 --block-size 192
"$PYTHON" -c 'import json; x=json.load(open("results/validation_identity.json")); assert x["summary"]["mean_acceptance_rate"] == 1.0'
bash scripts/run_official_protocol.sh 0
"$PYTHON" analyze.py
