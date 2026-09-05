# Research notes

## Question

Does sequence-level power sampling improve a frozen Qwen2.5-0.5B model on GSM8K when the model receives only the raw question? If it does, is the gain better than simpler uses of the same sampling budget?

## Algorithm

The target is the sequence distribution

\[
\pi(x) \propto p(x)^\alpha,
\]

not ordinary tokenwise temperature sampling. Starting from a low-temperature extension, each MCMC update samples a response-token cut, redraws the complete suffix from proposal `q`, and accepts candidate `x'` with

\[
\log r = \alpha[\log p(x')-\log p(x)] + \log q(x)-\log q(x').
\]

Only suffix terms after the shared prefix are needed. The implementation stores raw base-model and normalized proposal log probabilities separately. vLLM accelerates all extension, suffix-proposal, and scoring calls in batches.

## Protocol decisions

- Model: `Qwen/Qwen2.5-0.5B`, BF16.
- Prompt: exact GSM8K `question` string; no instruction, chat template, or answer-format hint.
- Grading: the last `\boxed{...}` value must numerically match the value after `####` in GSM8K. The standalone grader reproduces all 267 positive judgments from the parent project's earlier question-only outputs.
- Development: first 256 train questions, seed 0.
- Test: all 1,319 questions, seeds 0–2, with a deterministic independent RNG stream derived for every example.
- Main power setting: `alpha=4`, proposal temperature `0.25`, block size `32`, two MH updates per block, and 512 output tokens. This preserves the paper's 16-block structure while reducing its ten updates per block.

## Main result

| Method | Accuracy, mean ± SD | Proposal tokens/question |
|---|---:|---:|
| Temperature 1 | 12.03 ± 0.57% | 272 |
| Temperature 0.2 | 29.09 ± 1.40% | 275 |
| Power sampling, alpha 4 | 32.95 ± 0.43% | 1,636 |
| Six-sample vote, temperature 0.2 | 39.98 ± 1.21% | 1,638 |

Power improved over one low-temperature sample by 3.87 percentage points; a paired bootstrap that resamples questions after averaging their three seed outcomes gave a 95% interval of `[2.40, 5.31]`. Six-sample voting improved over power by 7.03 points, interval `[5.48, 8.59]`.

The fairest interpretation is therefore conditional. The power sampler successfully moved a 0.5B frozen model toward more correct outputs and was notably stable across seeds. It did not beat a much simpler self-consistency baseline when both were allowed about 1,638 proposal-generated tokens per problem. Power also requires extra full-sequence scoring calls that the token count omits, so voting is cheaper than this nominal match suggests.

## Ablation observations

- Power mattered: alpha 2, 4, and 8 produced 31.25%, 37.11%, and 37.50% on development. Alpha 4 was retained as the paper-aligned near-best setting; alpha 8 then scored 32.07% in its single test run versus 32.60% for alpha 4 seed 0.
- Two MH updates per block were enough: one, two, and four steps gave 33.59%, 37.11%, and 36.72%. More transitions increased cost without improving this small run.
- The paper-default ten updates gave 37.89% on development but required 5,858 proposal tokens per question. Its full-test seed-0 result was 32.52% at 6,119 tokens, slightly below two-step power's 32.60% at 1,645 tokens.
- Block size 32 was best among 16, 32, and 64. It also retains the paper's 16-block schedule at a 512-token budget.
- Output length was essential. Reducing 512 tokens to 256 lowered low-temperature accuracy from 34.77% to 15.62% and power from 37.11% to 19.14%, mainly because boxed-answer parse rates collapsed.
- Proposal temperature 0.25 was much better than 0.5 for alpha 4 (37.11% versus 29.69%). Independently, temperature 0.2 was the strongest ordinary-sampling value among 0.1, 0.2, 0.25, and 0.3.
- The official repository truncates the current suffix to the candidate length if the candidate emits EOS early, whereas the paper's ratio compares complete suffixes. Reproducing this `official_overlap` behavior changed development accuracy only from 37.11% to 37.50%. The main result uses the mathematically complete ratio.

## Diagnostics and limitations

The identity control exposed two implementation issues. First, variable-length candidates must include all probability terms from each complete suffix. Second, vLLM's BF16 generation-logprob and prompt-scoring paths are not numerically identical; over 64 tokens their accumulated difference reached 0.682. The exact alpha=temperature=1 identity is therefore enforced analytically. Other settings retain vLLM's measured proposal correction, so small numerical bias remains possible.

A second diagnostic explained why the parent repository once reported a 20.24% question-only baseline. Giving every request the same seed creates correlated vLLM streams: batch-seed test accuracy was 19.86%, 10.54%, and 9.86% across seeds. Per-example streams gave the far more stable 11.37%, 12.36%, and 12.36% used here. The old 20.24% result is reproducible as a favorable single batch stream, but it should not be compared with independently seeded power chains.

Finally, these results cover one small base model, strict boxed-answer grading, one benchmark, and a modest development set. The test set was evaluated for alpha 4 and alpha 8, so the alpha comparison is not a fully untouched confirmatory study. Wall times were also affected by an unrelated GPU1 process; proposal-token counts are the cleaner cost measure.

Repeated sampling continued to scale through the explored range. Plurality voting rose from 28.33% with one sample to 39.98% with six samples across three seeds. A ten-sample seed-0 exploration reached 44.66% at 2,727 proposal tokens; its pass@10 oracle ceiling was 59.59%. This is the best observed accuracy, but it is intentionally not mixed into the three-seed main table.
