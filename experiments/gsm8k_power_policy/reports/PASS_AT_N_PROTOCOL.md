# Raw-prompt pass@32 protocol

Fixed before the 500-question run, 2026-09-04, following the user's revised request.

- Compare only base T=1, base T=0.3, and the existing final power-policy adapter at T=1.
- Sample 500 of the 1,319 GSM8K test questions without replacement using Python `random.Random(0)`; sort and save original indices. Selection does not use answers or previous correctness.
- Generate 32 responses per condition per question (48,000 total). Compute all N=1…32 with `1 - C(32-c,N)/C(32,N)`, averaged over questions. Display N=1,2,4,8,16,32 and full curves.
- Use the exact raw-question training prompt, no added special tokens or few-shot demonstrations, EOS or 512-token limit. The proposed R1 three-shot prompt and tag-based decoder were cancelled before any such sampling ran.
- Use one vLLM engine with BF16 base weights and native LoRA loading, not a merged checkpoint. Adapter weights are cast from stored FP32 to BF16 by the inference backend. This differs numerically from the previous HF implementation; all three new conditions share the same backend.
- Disable top-k filtering, use top-p=1, no repetition/presence/frequency penalties, no stop strings or forced EOS. Pair per-question seeds as `42000 + original test index` across conditions. Do not force samples to be distinct or replace failed answers.
- Keep both existing strict boxed grading and the previously documented explicit numeric-answer grading, unchanged. No R1 grader and no new answer extraction rules.
- Retain all response text, token IDs, grades, completion reasons, subset indices, source/model/adapter hashes and versions. Resume only with an identical configuration.
- Report pointwise question-paired bootstrap 95% intervals using 10,000 resamples (seed 0). These do not include train/decoding-seed variability or multiple-comparison correction.
- This is inference only; no new training or checkpoint selection. Prior full-test observations make the experiment exploratory rather than an untouched test confirmation.

Implementation follows assignment5-alignment's per-question vLLM sampling and combinatorial pass@N estimator, while retaining this experiment's raw prompt, EOS termination, 512-token horizon and graders.
