#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
export CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

# Every training invocation, including startup and saving, has a 1,790-second bound.
exec timeout --signal=TERM --kill-after=5s 1785s \
  "$PYTHON" "$EXPERIMENT_ROOT/scripts/train.py" "$@"
