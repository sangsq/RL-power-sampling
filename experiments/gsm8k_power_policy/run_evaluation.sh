#!/usr/bin/env bash
set -euo pipefail

if [[ "${CUDA_VISIBLE_DEVICES:-}" != "1" ]]; then
  echo "Set CUDA_VISIBLE_DEVICES=1" >&2
  exit 2
fi
EXPERIMENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$EXPERIMENT_ROOT"

# Evaluation is separate from the training time budget. Keep batch size fixed
# across conditions so the per-batch seed protocol remains comparable.
BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
FINAL_ADAPTER="checkpoints/alpha2_seed0/final"
"$PYTHON" -c 'import json; s=json.load(open("results/alpha2_seed0/training_summary.json")); assert not s["reason"].startswith("error:"), s'

"$PYTHON" evaluate.py --name dev_initial --split train --limit 256 --batch-size "$BATCH_SIZE"
"$PYTHON" evaluate.py --name dev_policy --adapter "$FINAL_ADAPTER" --split train --limit 256 --batch-size "$BATCH_SIZE"

for checkpoint in step_0000 step_0032 step_0064 step_0128 final; do
  if [[ -d "checkpoints/alpha2_seed0/$checkpoint" ]]; then
    "$PYTHON" evaluate.py --name "diagnostic_$checkpoint" --adapter "checkpoints/alpha2_seed0/$checkpoint" \
      --split train --limit 64 --samples 4 --score --batch-size "$BATCH_SIZE"
  fi
done

"$PYTHON" evaluate.py --name test_base --temperature 1 --batch-size "$BATCH_SIZE"
"$PYTHON" evaluate.py --name test_t05 --temperature 0.5 --batch-size "$BATCH_SIZE"
"$PYTHON" evaluate.py --name test_t02 --temperature 0.2 --batch-size "$BATCH_SIZE"
"$PYTHON" evaluate.py --name test_policy --adapter "$FINAL_ADAPTER" --batch-size "$BATCH_SIZE"
"$PYTHON" analyze.py
