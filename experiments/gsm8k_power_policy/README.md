# E003–E005 · Power policy · GSM8K / Qwen2.5-0.5B

**E003** — First run complete: **158 updates in 27m 35s**. On all 1,319 test questions,
post-hoc explicit numeric accuracy was 34.34%, below base T=0.2 at 41.70%;
the primary boxed metric fell to 0.30% as the output format changed.
Read the [findings](reports/FINDINGS.md) and [full report](reports/REPORT.md).

**E005** — New run: [alpha 4, instruction prompt, 128-token horizon](reports/ALPHA4_BOX128.md),
submitted through tmux with a 30-minute training cap and delayed inspection.

Train ordinary all-linear rank-8 LoRA to approximate the complete-response
power distribution `p0(response | question)^2`, using HF Transformers and PEFT.
Only question prompts and model probabilities enter training. The raw-question
prompt and strict boxed-answer grader match the original GSM8K experiment.

- Power alpha: 2; LoRA alpha: 16; dropout: 0; frozen base and output embedding/head.
- Each update: four prompts, four responses each, one on-policy RLOO update.
- Response probabilities are summed through genuine EOS or the 512-token cap.
- No correctness reward, group standardization, length normalization, or replay.
- Training: 1,024 fixed questions outside the first 256 train examples (development).
- One seed, learning rate 1e-5, at most 256 updates; wall time takes precedence.

## Run

Install from the [setup instructions](../../SETUP.md). Always use physical GPU 1.
The explicit outer timeout bounds even a stalled CUDA call; the trainer normally
stops earlier and saves `checkpoints/<name>/final/`.

From the repository root:

```bash
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  timeout --signal=TERM --kill-after=5s 1790s \
  .venv/bin/python experiments/gsm8k_power_policy/train.py \
  --name alpha2_seed0 --max-seconds 1680 --generation-batch 16
```

The 28-minute internal budget includes model loading, rollouts, scoring,
backpropagation and checkpoints. A daemon watchdog exits 30 seconds after that
budget. The external timeout terminates at 1,790 seconds and kills after another
5 seconds, below the user's 30-minute maximum. Time-interrupted rollouts are
never used as complete responses in the training objective. Each run needs a
new name; old runs are never overwritten.

Evaluation runs separately on saved checkpoints, so its time does not reduce
the bounded training budget. Use the final completed checkpoint, not a checkpoint
selected for its correctness score.

Reproduce the complete evaluation matrix (batch size 64 for every condition):

```bash
CUDA_VISIBLE_DEVICES=1 PYTHON="$PWD/.venv/bin/python" \
  bash experiments/gsm8k_power_policy/run_evaluation.sh
```

For an individual condition, use the same batch size:

```bash
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python experiments/gsm8k_power_policy/evaluate.py \
  --name test_policy --batch-size 64 --adapter experiments/gsm8k_power_policy/checkpoints/alpha2_seed0/final

CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python experiments/gsm8k_power_policy/evaluate.py \
  --name test_base --batch-size 64 --temperature 1
```

Also evaluate the base at `--temperature 0.5 --name test_t05` and
`--temperature 0.2 --name test_t02`. All conditions use HF, one decoding seed,
and the same generation settings except temperature/adapter. `--split train
--limit 256` selects the development set. `--limit 64 --samples 4 --score` with
an adapter at T=1 measures held-out objective, diversity and pass@4.

```bash
.venv/bin/python experiments/gsm8k_power_policy/analyze.py
.venv/bin/python -m pytest -q
```

## Outputs

- `results/<run>/`: configuration, data indices/hashes, per-update metrics,
  rollout token IDs/log-probabilities and training termination summary.
- `checkpoints/<run>/`: initial/intermediate/final adapters, optimizer state,
  data ordering/position, Python/Torch/CUDA RNG states.
- `results/evaluation/`: resumable evaluation responses and summary statistics.
- `figures/`: training curves; `reports/`: interpretation and limitations.

## E004 · Pass@N comparison

The [fixed protocol](reports/PASS_AT_N_PROTOCOL.md) compares base T=1, base T=0.3
and policy T=1 on 500 randomly selected test questions, with 32 responses each.
It retains the raw training prompt and 512-token/EOS termination, using vLLM
for all three conditions. It reports both boxed and explicit numeric scoring.

```bash
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  .venv/bin/python -m experiments.gsm8k_power_policy.pass_at_n
.venv/bin/python -m experiments.gsm8k_power_policy.analyze_pass_at_n
```

Run these commands from the repository root. Saved JSONL groups allow resuming
with an identical configuration; `--name` creates a separate evaluation.
Results are under `results/pass_at_n/raw_500_n32/`.

Completed results: numeric pass@1 / pass@32 were **20.86% / 80.00%** for base
T=1, **39.81% / 83.20%** for base T=0.3, and **34.24% / 83.40%** for policy.
Policy's pass@32 difference from T=0.3 was +0.20 points (95% paired interval
[-3.20, +3.60]); this run shows no clear extra coverage at N=32.
See the [full curves, both graders and intervals](reports/PASS_AT_N_raw_500_n32.md).

The earlier single-answer report also includes an
[archived MCMC comparison](reports/MCMC_COMPARISON.md), regraded under the same
answer rules with explicit backend, alpha and cost differences. To regenerate
that appendix after `analyze.py`, run:

```bash
.venv/bin/python -m experiments.gsm8k_power_policy.compare_mcmc
```

Implementation reference: `assignment5-alignment`'s HF/PEFT helpers; no runtime
imports from that repository. Differences include exact generated-token scoring,
EOS masks, leave-one-out baselines and sequence-sum losses.

See the [training protocol and findings](reports/REPORT.md) and the
[pass@N protocol](reports/PASS_AT_N_PROTOCOL.md). Alpha conditioning and X-LoRA
remain deferred research ideas.
