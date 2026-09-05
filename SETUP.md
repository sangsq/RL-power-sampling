# Setup and Reproducibility

## Setup

Use Python 3.12. From this directory, install both experiments together:

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

## Verify and run

```bash
python -m pytest -q
python scripts/verify_portability.py
```

The verifier uses CPU only and preserves saved results and notebook outputs.
It also regenerates analysis in temporary directories and compares summaries.

Generation commands are in the [GSM8K README](experiments/gsm8k_qwen05b/README.md)
and [MATH500 README](experiments/math500_qwen15b/README.md). They require NVIDIA
CUDA and `CUDA_VISIBLE_DEVICES=1`. See
[model weights and reproducibility](#model-weights-and-reproducibility) for
downloads and environment details.

