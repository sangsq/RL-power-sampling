# E005 · Alpha 4 / instruction prompt / 128-token training

Run: `alpha4_box128_seed0`. Submitted through tmux session `power_alpha4_box128`
at 2026-09-04 20:59:25 PDT. The requested 30-minute check completed at
21:29:25 PDT (2026-09-05 04:29:25 UTC).

## Completed run and 30-minute check

Training exited normally (exit code 0) after **221 updates in 28m 33.26s**,
stopping on the time budget. The final adapter and optimizer/RNG state were
saved; source snapshots passed their hash checks. All logged training metrics
were finite. The run generated 1,794,093 training tokens, with 11.09 GB peak
allocated memory and 72.08% mean sampled GPU utilization. Every update
triggered gradient clipping.

| Training window | Mean objective J | Estimated KL | Mean response tokens | EOS rate | Truncated |
|---|---:|---:|---:|---:|---:|
| First 16 updates | -273.68 | -0.002 | 124.12 | 8.01% | 91.99% |
| Last 16 updates | -84.10 | 12.98 | 127.08 | 4.30% | 95.70% |

These windows contain different training questions and are not held-out
comparisons. The final window has no responses shorter than 16 tokens, but
most responses still hit the 128-token cap. This run has not yet been evaluated
for answer accuracy; objective improvement alone does not establish it.

[Saved 30-minute inspection](../results/jobs/alpha4_box128_seed0/inspection.json)

## Training protocol

The exact prompt, including the literal `\box{}`, is:

```text
Solve the following question step by step and place final result in \box{}: {question}
```

- Fresh Qwen2.5-0.5B base and adapter; fixed model revision and seed 0.
- Power alpha 4; all-linear LoRA rank 8, LoRA alpha 16, dropout 0.
- HF Transformers/PEFT, BF16/SDPA, physical GPU 1 only.
- T=1, top-p=1, no top-k filtering; genuine EOS or 128 new tokens.
- 16 questions per update, K=4 responses per question, 64 fresh rollouts;
  generation batch 64, scoring/backward microbatch 8.
- AdamW learning rate 1e-5, weight decay 0, gradient norm clip 1.
- Same 1,024 training questions and 256-question development holdout as the
  original run. Repeat epochs in deterministic shuffled order until the time
  budget; no training answers or correctness rewards.
- Internal budget 1,740 seconds (29 minutes), including loading and saving;
  daemon watchdog at 1,770 seconds; outer timeout at 1,790 seconds with a
  further five-second kill grace. The 10,000-step ceiling leaves time as the
  practical stopping condition. No evaluation runs during training.

The three-update probe averaged approximately 11.5 seconds per update, with
9.30 GB peak allocated GPU memory. Initial truncation was 93.75% under the
requested 128-token horizon; this is recorded as behavior, not modified away.
All 20 CPU tests passed, including literal prompt substitution and epoch ordering.

The original alpha-2 trainer source was preserved at
`results/alpha2_seed0/source/train.py` before extending the trainer. Every new
run also snapshots its trainer and policy code and records their hashes.

## Files and inspection

- [Submitted configuration](../results/alpha4_box128_seed0/config.json)
- [Live training log](../results/jobs/alpha4_box128_seed0/training.log)
- [Training metrics](../results/alpha4_box128_seed0/metrics.jsonl)
- [Submission and check times](../results/jobs/alpha4_box128_seed0/submission.json)
- Check result: `results/jobs/alpha4_box128_seed0/inspection.json`, written at
  the 30-minute mark by the detached job.
- Final adapter and optimizer/RNG state: `checkpoints/alpha4_box128_seed0/final/`.

Attach with `tmux attach -t power_alpha4_box128`. The launch script is
[`run_alpha4_tmux.sh`](../run_alpha4_tmux.sh). The saved configuration specifies
the prompt and horizon for any later evaluation of this checkpoint.
