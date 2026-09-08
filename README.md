# LLM reasoning via distribution sharpening

Experiments in reasoning by sequence-level power-sharpening a transformer language model:
$$
p(\mathrm{response} \mid \mathrm{prompt}) \;\longrightarrow\; \frac{p(\mathrm{response} \mid \mathrm{prompt})^{\alpha}}{Z_{\alpha}}
$$

Power distribution is obtained via: 
1. Markov Chain Monte Carlo (MCMC) power sampling, or 
2. Training an RL policy to variationally approximate the power distribution, without having access to any true label. REINFORCE with Leave-One-Out (RLOO) is used to train the policy.

**Model(s) used:**
- Qwen2.5-0.5B

**Dataset(s) used:**
- GSM8K

## Results

Answer pass@N on 500 GSM8K questions with Qwen2.5-0.5B, 32 samples per question and a 512-token cap.

| Method | Pass@1 | Pass@4 | Pass@32 |
|---|---:|---:|---:|
| Base (T=1) | 16.71% | 41.48% | 76.40% |
| Best low T (T=0.25) | 34.03% | 58.98% | 82.40% |
| MCMC (α=2) | 38.93% | 65.82% | 86.80% |
| RLOO | 38.56% | 59.25% | 85.80% |

RLOO improves pass@1 over low-temperature sampling and approaches MCMC's point estimate, but remains below MCMC at pass@4

See the [report](experiments/rloo_adapt_len/report.ipynb) for full results.

## Project Structure

```text
src/           # Sampling, grading, and policy updates
experiments/   # Indexed experiments, data, configs, results, and reports
tests/         # CPU correctness checks
scripts/       # Repository verification utilities
```

See [SETUP.md](SETUP.md) for installation, model downloads, and verification.
Model checkpoints and private agent notes are excluded from Git.
