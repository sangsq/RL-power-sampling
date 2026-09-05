#!/usr/bin/env bash
set -uo pipefail

EXPERIMENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$EXPERIMENT_ROOT/../.."
export CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export MPLCONFIGDIR=/tmp/power-policy-matplotlib
export PP_RUN_NAME=alpha4_box128_seed0
export PP_STARTED_UNIX="$(date +%s)"
JOB_DIR="$EXPERIMENT_ROOT/results/jobs/$PP_RUN_NAME"
mkdir -p "$JOB_DIR"
.venv/bin/python - <<'PY' > "$JOB_DIR/submission.json"
from datetime import datetime, timezone
import json, os
started = int(os.environ['PP_STARTED_UNIX'])
print(json.dumps({'run': os.environ['PP_RUN_NAME'], 'submitted_unix': started,
                  'submitted_at_utc': datetime.fromtimestamp(started, timezone.utc).isoformat(),
                  'check_after_unix': started + 1800,
                  'check_at_utc': datetime.fromtimestamp(started + 1800, timezone.utc).isoformat()}, indent=2))
PY

nvidia-smi -i 1 --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv -l 2 > "$JOB_DIR/gpu.csv" &
GPU_MONITOR_PID=$!
timeout --signal=TERM --kill-after=5s 1790s \
  .venv/bin/python -m experiments.gsm8k_power_policy.train \
  --name "$PP_RUN_NAME" --alpha 4 --max-tokens 128 --steps 10000 \
  --max-seconds 1740 --lr 1e-5 --seed 0 \
  --prompts-per-step 16 --generation-batch 64 --microbatch 8 \
  --prompt-file experiments/gsm8k_power_policy/prompts/solve_box.prompt \
  > "$JOB_DIR/training.log" 2>&1
TRAIN_EXIT=$?
printf '%s\n' "$TRAIN_EXIT" > "$JOB_DIR/exit_code.txt"
kill "$GPU_MONITOR_PID" 2>/dev/null || true
wait "$GPU_MONITOR_PID" 2>/dev/null || true

# The detached job performs the requested 30-minute check even if the client disconnects.
.venv/bin/python - <<'PY'
import os, time
time.sleep(max(0, int(os.environ['PP_STARTED_UNIX']) + 1800 - time.time()))
PY
.venv/bin/python -m experiments.gsm8k_power_policy.inspect_training --name "$PP_RUN_NAME" \
  > "$JOB_DIR/check.log" 2>&1
exit "$TRAIN_EXIT"
