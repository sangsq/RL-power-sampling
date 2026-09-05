# Reproduction plan

## Objective

Reproduce the central training-free algorithm in *Reasoning with Sampling* at a scale suitable for one RTX 4090: Qwen2.5-0.5B, GSM8K, the exact raw question as the prompt, and vLLM on physical GPU 1.

## Protocol

1. Derive the sequence-level Metropolis–Hastings ratio and cross-check it against the paper and official code.
2. Validate grading, variable-length suffix accounting, and the identity kernel with unit and GPU tests.
3. Tune only on the first 256 GSM8K train questions.
4. Ablate proposal temperature, power, MCMC steps, block size, maximum length, and EOS suffix accounting.
5. Evaluate the paper-aligned near-best setting (`alpha=4`, temperature `0.25`, 16 blocks, 2 MH steps/block, 512 output tokens) on all 1,319 GSM8K test questions across three independent seeds.
6. Compare against temperature-1 sampling, selected low-temperature sampling, and six-sample self-consistency at approximately the same proposal-generated-token budget.
7. Retain every response, report mean and standard deviation across seeds, and use paired question-clustered bootstrap intervals for method differences.

The reproduction intentionally reduces the paper's model size and MCMC count. It tests the algorithmic idea; it does not attempt a numerical reproduction of the paper's 7B-model results.
