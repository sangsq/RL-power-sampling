# E003 · Objective improves, but low-temperature sampling remains stronger

This run trained a fixed power-alpha-2 policy on Qwen2.5-0.5B / GSM8K using
HF Transformers and PEFT, all-linear LoRA rank 8, and LoRA alpha 16. Training
used only questions and model probabilities, with no answer supervision. It
completed **158 updates in 27m 35s** and saved the final checkpoint when the
time budget was reached. No checkpoint was selected by evaluation accuracy.

## Full test set

All four conditions cover 1,319 questions with one response per question,
using the same HF backend, raw-question prompt, 512-token cap, and decoding seed.

| Condition | Strict boxed accuracy (original primary metric) | Explicit numeric accuracy (post-hoc supplement) | Mean tokens |
|---|---:|---:|---:|
| Base, T=1 | 9.93% | 19.56% | 268.6 |
| Base, T=0.5 | 24.03% | 37.45% | 262.7 |
| Base, T=0.2 | 27.45% | 41.70% | 270.1 |
| Power policy, T=1 | 0.30% | 34.34% | 134.1 |

The policy largely switched to `#### <number>` and rarely produced boxed
answers. The strict-score decline therefore cannot be read as mathematical
ability falling to zero. After viewing development outputs and before viewing
test results, we recorded [uniform supplementary extraction rules](GRADING_SENSITIVITY.md)
and regraded the saved responses for every condition. No responses were
resampled and the primary metric stayed fixed. The supplement checks the
final number, not the validity of the reasoning.

Under the supplementary metric, the policy improves over T=1 by **14.78
percentage points**, with a question-paired bootstrap 95% interval of
**[12.05, 17.59]**. Its difference from T=0.5 is **−3.11 [−5.91, −0.30]**;
from T=0.2, **−7.35 [−10.16, −4.47]**. These intervals reflect question
resampling only, omit training/decoding-seed uncertainty, and have no
multiple-comparison adjustment.

## Observed behavior

- On the fixed 64-question development diagnostic with four samples per
  question, J improved from **−116.22 to −31.44**, while estimated
  KL(policy || base) rose from 0 to 3.67. Optimization changed behavior
  substantially, but this does not establish closeness to the exact power distribution.
- Responses became shorter and adopted the familiar GSM8K `####` format.
  The final diagnostic had no first-token EOS or responses shorter than
  16 tokens; test EOS rate was 99.24%. No empty-response collapse was observed.
- Estimated sequence entropy fell from 116.22 to 24.09, while nearly all
  four responses per question still had distinct text. Supplementary numeric
  pass@4 was **70.31%** both before and after training, giving no evidence
  of covering more questions with a correct candidate.
- Manual inspection found both improvements and arithmetic or relationship
  errors in shorter answers; see the [paired examples](QUALITATIVE_EXAMPLES.md).
  For example, the policy evaluated `40 / 2 + 5` as 20. Format changes do
  not account for every error.
- Mean output length roughly halved, but observed batch-64 inference time
  was 116.7 seconds for the policy and 116.6 seconds for T=0.2. The current
  implementation did not show corresponding wall-clock savings. These
  single-run timings are not a benchmark with controlled system load.

## Conclusion and follow-up

**Under this budget and configuration, the power policy did not outperform
direct low-temperature sampling.** It improved explicit numeric accuracy
over T=1 and shortened responses, while changing format and remaining weaker
than both low-temperature baselines. This does not establish that the method
is ineffective in general: there was one training seed, only 632 training
questions were encountered, and every update triggered gradient norm clipping.

A follow-up should first fix a GSM8K final-answer scoring rule in advance,
then repeat training and decoding seeds to check these observations, retaining
the 30-minute limit per training run. This experiment did not add tuning runs.
Alpha conditioning and X-LoRA remain deferred research ideas.

Full settings, diagnostics, confidence intervals, and costs are in the
[technical report](REPORT.md); the [summary](../results/summary.json) is
machine-readable. All 18 CPU tests passed. Real-model alpha-1 identity,
alpha-2 smoke, frozen-reference, and checkpoint checks were completed.

<!-- MCMC_COMPARISON -->
## MCMC power sampling comparison (supplement)

Archived vLLM results cover all 1,319 GSM8K test questions with three decoding seeds. The original responses were regraded using the policy experiment's strict boxed and explicit numeric answer rules. No responses were resampled or seeds selected by their outcomes.

| Condition | Backend / seeds | Strict accuracy | Numeric accuracy | Output tokens/question | Generated tokens/question |
|---|---|---:|---:|---:|---:|
| Base T=1 | HF / 1 | 9.93% | 19.56% | 268.6 | 268.6 |
| Base T=0.5 | HF / 1 | 24.03% | 37.45% | 262.7 | 262.7 |
| Base T=0.2 | HF / 1 | 27.45% | 41.70% | 270.1 | 270.1 |
| Power policy α=2, T=1 | HF / 1 | 0.30% | 34.34% | 134.1 | 134.1 |
| Archived base T=1 | vLLM / mean of 3 | 12.03% | 21.13% | 272.1 | 272.1 |
| Archived base T=0.2 | vLLM / mean of 3 | 29.09% | 41.95% | 274.9 | 274.9 |
| MCMC power alpha=4 | vLLM / mean of 3 | 32.95% | 46.50% | 250.9 | 1635.9 |

MCMC uses the frozen base, power α=4, proposal T=0.25, block size 32, two Metropolis–Hastings updates per block, a 512-token cap, and the `full` suffix probability ratio. Each block first extends the sequence from the proposal, then selects a random suffix boundary and resamples that suffix. Acceptance uses the base and proposal sequence probabilities: `min(1, exp(α·Δlog p0 − Δlog q))`. This finite-step autoregressive MCMC implementation does not produce independent samples from the exact power distribution.

MCMC requires multiple proposal and reference-scoring passes per question at inference time, without training. The policy first trains an adapter and then uses ordinary sampling. MCMC generated-token counts include extensions and rejected proposals, but exclude reference-scoring forward passes. Policy costs do not amortize the earlier 27m 35s training run.

**This is a historical algorithm reference; its differences from the policy do not isolate method quality.** Power α differs (4 versus 2), as do the backend (vLLM versus HF), random seeds, and number of seeds. MCMC settings were previously selected on development data. Archived vLLM base and low-temperature results are included to expose differences between backends and experiment batches. The archived MCMC files do not record an explicit `seed_policy` field; original configs and file hashes are retained in the supplement.

[Machine-readable summary and individual seeds](../results/mcmc_comparison.json) · [Original MCMC report](../../gsm8k_qwen05b/reports/REPORT.md) · [Sampler implementation](../../../src/power_sampling/sampler.py)
