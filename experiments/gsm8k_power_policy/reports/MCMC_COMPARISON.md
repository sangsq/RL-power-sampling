# Algorithm reference for the single-answer experiment

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
