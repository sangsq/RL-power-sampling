# Setup and Reproducibility

## Setup

Use Python 3.12. From this directory, install the project:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -c requirements-tested.txt -e .
source .venv/bin/activate
```

With pip, use `python3.12 -m venv .venv` followed by
`.venv/bin/python -m pip install -c requirements-tested.txt -e .`.
An existing `.venv` is available on this machine.

## Model weights and reproducibility

Download model weights once before running offline experiments:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen2.5-0.5B", revision="060db6499f32faf8b98477b0a26969ef7d8b9987")
snapshot_download("Qwen/Qwen2.5-Math-1.5B", revision="4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2")
PY
```

Weights use the standard Hugging Face cache. For a project-local cache, set
`HF_HUB_CACHE="$PWD/models/hub"` from the repository root before downloading
and running. GPU experiments require a compatible NVIDIA driver and
`CUDA_VISIBLE_DEVICES=1`; MATH500 uses offline mode.

`requirements-tested.txt` records the tested Python 3.12 environment from
2026-09-04, including PyTorch 2.10.0/CUDA 12.8. Earlier sampling runs used
2.10.0+cu129; numerical equivalence between those environments was not tested.
Recreate the virtual environment after moving the repository. Per-run configs
and source snapshots provide additional provenance.

## Rebuild the public reports

These commands use CPU only and need no model weights or ignored scratch files:

```bash
python -m pytest -q
python experiments/gsm8k_power_policy/scripts/analyze.py
python experiments/rloo_adapt_len/scripts/analyze.py
```

The baseline notebook documents its protocol; the adaptive-length notebook contains
one combined comparison table and two training-diagnostic panels. Computation lives
in scripts; notebooks contain Markdown and relative figure links.

Tracked per-question correctness counts and response lengths support exact
regeneration of pass@N, paired-question intervals and token totals. Saved sequence
log probabilities reproduce the NLL/CE/true-loss decomposition. Compact files do
not support regrading response text. Original outputs and checkpoints remain local;
`results/provenance/manifest.json` records their hashes. Frozen source snapshots
and historical configs may contain old paths; report generation never resolves them.

## Fresh experiments

Use a new run name and physical GPU 1. The bounded training launchers enforce an
external limit of 1,790 seconds including shutdown; they refuse to overwrite a
saved run. For example:

```bash
bash experiments/gsm8k_power_policy/scripts/run_train.sh \
  --name new_alpha2 --alpha 2 --max-seconds 1680 --max-tokens 512 \
  --prompts-per-step 16 --generation-batch 64 --microbatch 4 \
  --prompt-file experiments/gsm8k_power_policy/scripts/prompts/paper_math.prompt \
  --track-accuracy

bash experiments/rloo_adapt_len/scripts/run_train.sh \
  --name new_adapt_alpha2 --alpha 2 --max-seconds 1680
```

The published runs took about one hour under their original budget. The adaptive
script preserves the historical 38-updates-per-stage quota: a new 28-minute run
may stop with an incomplete schedule and cannot be treated as its reproduction.
The published reports are regenerated from the completed historical measurements.

For baseline decoding, choose a separate output directory under this repository:

```bash
CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python experiments/gsm8k_power_policy/scripts/evaluate.py \
  --condition mcmc_alpha2 \
  --output experiments/gsm8k_power_policy/results/new_evaluation
```

Other conditions are `base_t1`, `base_t0.25`, `policy_alpha2` (requires `--adapter`),
and `select_temperature` (the recorded development grid). The evaluator resumes
only matching configurations and writes compact measurements after completion.
Saved historical adapters are local and are not distributed with the public reports.
