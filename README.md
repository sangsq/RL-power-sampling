# Reasoning by power distribution sampling

## Introduction

Experiments in reasoning by sequence-level power-shapening a transformer language model:
$$
p(\mathrm{response} \mid \mathrm{prompt}) \;\longrightarrow\; \frac{p(\mathrm{response} \mid \mathrm{prompt})^{\alpha}}{Z_{\alpha}}
$$

Power distribution is obtained via: 1. Markov Chain Monte Carlo (MCMC) power sampling, or 2. training an RL policy to variationally approximate the power distribution, without access to any true label.

**Models used:**
- Qwen2.5-0.5B
- Qwen2.5-Math-1.5B

**Datasets used:**
- GSM8K
- MATH500

## Current Results

Strict boxed-answer accuracy, averaged across decoding seeds:

| Dataset / model | Base | Low temperature | MCMC power |
|---|---:|---:|---:|
| GSM8K / Qwen2.5-0.5B | 12.03% | 29.09% | 32.95% |
| MATH500 / Qwen2.5-Math-1.5B | 46.4% | 60.2% | 64.9% |

GSM8K six-sample voting reaches **39.98%** at a similar proposal-token budget
to MCMC. The alpha-2 LoRA policy improves numeric accuracy over base T=1 but
remains below low-temperature sampling. On a fixed 500-question subset, numeric
pass@32 is **83.40%** for the policy versus **83.20%** for base T=0.3, with no
clear difference. The alpha-4, 128-token policy has finished training;
answer accuracy is not yet evaluated.

Protocols, uncertainty, and limitations are in the [E001–E005 experiment index](experiments/README.md).

## Project Structure

```text
src/           # Sampling, grading, and policy updates
experiments/   # Indexed experiments, data, configs, results, and reports
tests/         # CPU correctness checks
scripts/       # Repository verification utilities
```

See [SETUP.md](SETUP.md) for installation, model downloads, and verification.
Model checkpoints and private agent notes are excluded from Git.
