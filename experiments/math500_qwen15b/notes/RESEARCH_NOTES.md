# Research notes

## 2026-09-01 — Protocol reconstruction

The paper targets the sequence-level distribution `p(x)^alpha`, which is not equivalent to lowering the temperature of every conditional next-token distribution. Its sampler grows a response in 16 blocks and applies suffix-resampling Metropolis–Hastings updates after every block. For MATH500, the official settings are `alpha=4`, proposal temperature `0.25`, 10 MH updates per block, and a 3,072-token cap.

The official code evaluates Qwen2.5-Math-7B and reports 49.6% base, 69.0% low-temperature, and 74.8% power-sampling accuracy. This study substitutes the cached 1.5B checkpoint, so the scientific target is a directional reproduction rather than equality of absolute scores.

The official runner is sequential and writes only after a complete 100-question shard. The local implementation batches requests with vLLM and checkpoints after each block. It preserves the MH target and proposal but makes seeds explicit and resumable.

## 2026-09-01 — Seed-0 result

Using the grader compatible with the paper, ordinary temperature-1 sampling
scored 45.0%, low-temperature sampling scored 60.6%, and power sampling scored
65.2%. Thus the smaller model reproduces the paper's ordering. The paired
power-minus-low-temperature difference is +4.6 percentage points; a
question-level paired bootstrap gives a 95% interval of +0.6 to +8.8 points for
this decoding seed. This interval does not cover decoding-seed uncertainty.

The comparison is one returned answer per question, not equal inference compute.
Power sampling generated 9,843 candidate tokens per question on average for a
627-token final response, or roughly 15.7 times the final response length. Mean
MH acceptance was 60.8%.

## 2026-09-01 — Evaluation correction

The first analysis used `math_verify`. Cross-checking against the public
repository grader found 22 disagreements among the 1,500 seed-0 outputs. These
mostly involved units, percentages, equivalent expressions, tuple ordering, or
strict integer rules. A compact local implementation of the paper's grader now
serves as the primary metric and exactly matches the public grader on every
saved output checked so far. `math_verify` remains visible as a sensitivity
analysis; it gives 45.2%, 61.6%, and 66.4% on seed 0, so the ranking is stable.

## 2026-09-01 — Paired behavior

Relative to low-temperature sampling, seed-0 power sampling creates 67 gains
and 44 losses. Some gains are genuine reasoning changes: it fixes an off-by-one
boundary count and rejects a false algebraic identity in inspected examples.
Other changes concern answer termination. Twenty gains turn an unparseable
low-temperature response into a correct boxed answer, while 19 losses replace a
correct low-temperature answer with an unboxed power response. This explains
how power accuracy can rise even though its boxed-answer parse rate falls from
90.4% to 88.4%. The safe conclusion is that power sampling changes both
mathematical trajectories and format behavior.

## 2026-09-01 — Two-seed conclusion

Seed 1 independently gives 47.8% base, 59.8% low temperature, and 64.6% power
accuracy with the paper-compatible grader. Across seeds 0 and 1, the respective
means are 46.4%, 60.2%, and 64.9%. Power exceeds low temperature by 4.7 points;
paired question resampling after averaging the two seeds gives a 95% interval
of +1.8 to +7.7 points. Seed-level standard deviations are 2.0, 0.6, and 0.4
points, though two seeds are too few for a reliable seed-variance interval.

The paired direction is consistent in both runs. Seed 0 has 67 power gains and
44 losses relative to low temperature; seed 1 has 60 gains and 36 losses. Across
1,000 paired outputs, 45 gains begin from an unparseable low-temperature answer,
while 37 losses end in an unparseable power answer. The main conclusion remains
directional and compute-aware: sequence-level power sampling improves one-answer
accuracy on this frozen 1.5B model, but requires about 15.7 times the final
response token count in proposal sampling.
