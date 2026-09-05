# E004 · Raw-prompt pass@N comparison

All three conditions use the same 500 GSM8K test questions, selected without replacement using Python random seed 0; sorted original indices are saved in config.json. Each question has 32 samples per condition (48,000 responses total).

The prompt is exactly the raw question used during policy training, with no chat template, few-shot demonstrations or R1 tags. Generation ends at EOS or 512 tokens. All conditions use vLLM, BF16, top-p=1, top-k disabled, no penalties or stop strings. Per-question seeds are 42000 + original test index, paired across conditions. The final 158-update policy adapter is loaded directly through vLLM LoRA (adapter weights cast to BF16 for inference); no merging or further training. This is a new matched-vLLM evaluation, so differences from earlier HF results also reflect the subset, decoder and random draws.

For c successes among m samples, pass@N = 1 − C(m−c,N)/C(m,N), averaged over questions. It measures whether an oracle could find a correct candidate, not majority-vote accuracy. At N=m it is the fraction of questions with at least one observed success. The numeric extraction rule was fixed in the previous experiment and is applied unchanged; strict boxed scoring remains separate. Numeric correctness does not validate the reasoning.

## Strict boxed

| Condition | Pass@1 | Pass@2 | Pass@4 | Pass@8 | Pass@16 | Pass@32 |
|---|---:|---:|---:|---:|---:|---:|
| Base T=1 | 11.54% | 20.33% | 32.76% | 47.12% | 60.91% | 73.40% |
| Base T=0.3 | 27.23% | 38.53% | 49.98% | 60.59% | 69.59% | 76.40% |
| Policy T=1 | 0.61% | 1.14% | 2.00% | 3.34% | 5.45% | 8.80% |

Policy minus baseline, percentage points with paired question-bootstrap 95% intervals:

- vs Base T=1, N=1: -10.93 [-12.06, -9.86].
- vs Base T=1, N=32: -64.60 [-69.00, -60.20].
- vs Base T=0.3, N=1: -26.62 [-29.25, -24.14].
- vs Base T=0.3, N=32: -67.60 [-71.80, -63.40].

## Explicit numeric

| Condition | Pass@1 | Pass@2 | Pass@4 | Pass@8 | Pass@16 | Pass@32 |
|---|---:|---:|---:|---:|---:|---:|
| Base T=1 | 20.86% | 32.93% | 46.36% | 58.92% | 69.97% | 80.00% |
| Base T=0.3 | 39.81% | 51.36% | 61.66% | 70.43% | 77.65% | 83.20% |
| Policy T=1 | 34.24% | 46.00% | 57.24% | 67.66% | 76.50% | 83.40% |

Policy minus baseline, percentage points with paired question-bootstrap 95% intervals:

- vs Base T=1, N=1: +13.38 [+11.82, +14.96].
- vs Base T=1, N=32: +3.40 [-0.20, +7.20].
- vs Base T=0.3, N=1: -5.57 [-7.36, -3.84].
- vs Base T=0.3, N=32: +0.20 [-3.20, +3.60].


![Pass@N curves](../figures/pass_at_n_raw_500_n32.png)

## Behavior and cost

| Condition | Boxed parse | Numeric parse | Mean tokens | Truncated | Unique texts/question | Generation + scoring minutes |
|---|---:|---:|---:|---:|---:|---:|
| Base T=1 | 41.04% | 73.65% | 272.7 | 14.73% | 31.99 | 4.70 |
| Base T=0.3 | 64.89% | 89.19% | 267.9 | 5.84% | 30.04 | 4.69 |
| Policy T=1 | 2.02% | 96.47% | 132.8 | 0.60% | 31.42 | 7.09 |

## Limits

One trained checkpoint and one decoding seed protocol; no seed uncertainty is included. Intervals are pointwise, not simultaneous bands or multiplicity-adjusted claims. Previously viewed test results informed the decision to run this evaluation, so this is exploratory, not a pristine held-out confirmation. Finite sample pass@N does not measure the full answer support. No checkpoint or subset was selected by these outcomes.

Full N=1…32 arrays, intervals, indices and settings: [summary.json](../results/pass_at_n/raw_500_n32/summary.json). All raw responses, generated token IDs and per-sample grades are retained in the adjacent JSONL files.
