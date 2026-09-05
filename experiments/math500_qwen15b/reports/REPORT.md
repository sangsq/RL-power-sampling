# Power Sampling on MATH500 with Qwen2.5-Math-1.5B

## Executive summary

This study reproduces the MATH500 experiment from *Reasoning with Sampling* on
a smaller frozen model. Across two complete decoding seeds, power sampling
reaches **64.9%**, compared with **60.2%** for low-temperature sampling and
**46.4%** for ordinary sampling. The paired power-over-low-temperature effect is
**+4.7 percentage points** (question-bootstrap 95% CI: **+1.8 to +7.7**).

The method ordering reported for Qwen2.5-Math-7B is therefore reproduced on
Qwen2.5-Math-1.5B. The gain is not free: power sampling generates about **15.7×**
as many candidate tokens as appear in its returned response.

## Research question

Does sequence-level sharpening—sampling from a distribution proportional to
`p(response)^alpha`—improve single-response MATH500 accuracy without training or
modifying Qwen2.5-Math-1.5B?

## Protocol

- Data: the official fixed MATH500 JSON at public commit `720a8e9d…`, verified
  by SHA-256 `838cd5ff…e1e06`.
- Prompt: the official Qwen Math chain-of-thought prompt with a required final
  `\boxed{}` answer.
- Baselines: ordinary sampling at temperature 1.0 and low-temperature sampling
  at temperature 0.25.
- Power sampling: alpha 4, a temperature-0.25 proposal, 16 blocks of 192 tokens,
  10 suffix-resampling Metropolis–Hastings updates per block, and a 3,072-token
  completion limit.
- Engine: vLLM 0.19.1 in BF16. Every model process ran with
  `CUDA_VISIBLE_DEVICES=1`; GPU0 was never visible to the experiment.
- Evaluation: a self-contained reimplementation of the paper's MATH grader is
  primary. `math_verify` is retained as a sensitivity analysis.

## Results

| Method | Seed 0 | Seed 1 | Mean | `math_verify` mean | Parse rate | Final tokens | Sampled tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base, T=1 | 45.0% | 47.8% | 46.4% | 47.0% | 84.9% | 664 | 664 |
| Low temperature, T=0.25 | 60.6% | 59.8% | 60.2% | 61.4% | 89.1% | 653 | 653 |
| Power sampling, alpha=4 | 65.2% | 64.6% | **64.9%** | 66.1% | 88.9% | 628 | 9,861 |

Paired question-bootstrap effects after averaging the two seeds are:

- Low temperature minus base: **+13.8 pp** (95% CI: +10.1 to +17.4).
- Power minus base: **+18.5 pp** (95% CI: +14.7 to +22.3).
- Power minus low temperature: **+4.7 pp** (95% CI: +1.8 to +7.7).

The paper's corresponding 7B scores are 49.6%, 69.0%, and 74.8%. Absolute
scores differ as expected after reducing the model to 1.5B, while the ordering
and incremental benefit over low temperature remain.

## What changed in the outputs?

Power sampling produces 67 gains and 44 losses against low temperature on seed
0, then 60 gains and 36 losses on seed 1. Paired inspection shows two effects.

First, some trajectories contain different mathematical decisions. Power
sampling repairs an off-by-one boundary count and replaces a false algebraic
identity in representative cases. Second, it changes termination and format
behavior. Across both seeds, 45 gains begin from an unparseable low-temperature
response, while 37 losses end in an unparseable power response. Consequently,
power accuracy rises even though its mean boxed-answer parse rate (88.9%) is
essentially the same as—and slightly below—low temperature (89.1%).

These observations support the limited claim that the sampler changes both
reasoning content and format behavior. They do not establish a uniform local
correction mechanism. Full paired examples appear in `QUALITATIVE_EXAMPLES.md`.

## Validation and evaluation correction

Tests cover the exact prompt, official dataset hash, all 500 target answers,
Metropolis–Hastings ratio identities, checkpoint/RNG restoration, and result
aggregation. A GPU identity-kernel validation with alpha 1 and temperature 1
accepted every proposal, as required.

The initial analysis used `math_verify`. It disagreed with the public grader on
22 of 1,500 seed-0 outputs, mainly because of units, percentages, tuple ordering,
and strict integer rules. The final local paper-compatible grader matches the
official implementation on all **3,008** saved outputs, including the validation
set and both full seeds. The qualitative ranking is unchanged under
`math_verify`, but absolute accuracies shift by up to 1.2 points.

## Limitations

- This is a directional 1.5B reproduction of a headline 7B experiment.
- The comparison holds the number of returned answers fixed, not total inference
  compute. Power sampling uses approximately 15.7× sampled-token overhead.
- Two decoding seeds are more informative than one but do not reproduce the
  public repository's full eight-run protocol or support a reliable seed-level
  confidence interval.
- vLLM batching and explicit per-question seeds replace sequential Transformers
  generation. They preserve the target and proposal distributions but not exact
  floating-point or random-number trajectories.

## Conclusion

The study reproduces **power sampling > low temperature > base** on frozen
Qwen2.5-Math-1.5B. Sequence-level sharpening improves observed single-response
accuracy beyond conditional temperature scaling, at a substantial inference
cost. This is evidence for a decoding improvement—not equivalence to RL, a
larger model, or an equal-compute alternative.
