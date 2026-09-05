# Reproduction plan

**Status:** Complete for two full decoding seeds. The exact-protocol second seed
was prioritized over a small-shard hyperparameter ablation.

## Objective

Measure whether training-free power sampling improves single-shot MATH500 accuracy over matched ordinary and low-temperature decoding for frozen `Qwen/Qwen2.5-Math-1.5B`.

## Protocol

1. Use the exact 500 examples and prompt shipped by official commit `720a8e9d084c87a630595e316f5260f1d7c3446c`.
2. Validate the MH ratio, nested-box extraction, symbolic grading, prompt string, data hash, and checkpoint round trip on CPU.
3. Once physical GPU 1 is free, run a short identity-kernel validation (`alpha=1`, proposal `T=1`) without touching GPU 0.
4. Run full MATH500 ordinary sampling (`T=1`) and low-temperature sampling (`T=0.25`).
5. Run the paper-aligned power sampler: `alpha=4`, proposal `T=0.25`, `max_tokens=3072`, `block_size=192`, `mcmc_steps=10`. Match the official five 100-question shards.
6. Repeat seed 1 to check whether the power-over-low-temperature ordering survives a second decoding trajectory. Prioritize this exact replication over a small-shard MCMC ablation.
7. Report accuracy, parse rate, response length, proposal-token cost, acceptance rate, paired differences, uncertainty, and representative changed answers.

## Comparability

The prompt, target distribution, proposal, block schedule, maximum length,
benchmark, answer parser, and primary grader match the paper. Model size differs
(1.5B versus 7B), vLLM replaces sequential Transformers generation, and seeds
are made reproducible. A second `math_verify` score tests grader sensitivity.
These differences remain explicit in every report.
