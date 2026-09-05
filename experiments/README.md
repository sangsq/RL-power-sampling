# Experiments in chronological order

IDs are permanent and ordered by experiment start date. Seeds, baselines,
ablations, smoke tests, diagnostics, and resumed jobs share their parent
experiment's ID. Historical files retain their original paths; the mapping is
stored in [registry.json](registry.json).

| ID | Start date (Los Angeles time) | Experiment | Results / report |
|---|---|---|---|
| E001 | 2026-08-30 | GSM8K / Qwen2.5-0.5B: base, low temperature, MCMC, and voting | [Experiment](gsm8k_qwen05b/README.md) |
| E002 | 2026-09-01 | MATH500 / Qwen2.5-Math-1.5B: sampling reproduction | [Experiment](math500_qwen15b/README.md) |
| E003 | 2026-09-04 | GSM8K power policy: alpha 2, raw prompt, 512 tokens | [Report](gsm8k_power_policy/reports/FINDINGS.md) |
| E004 | 2026-09-04 | Fixed 500-question pass@1…32: base T=1, T=0.3, and E003 policy | [Report](gsm8k_power_policy/reports/PASS_AT_N_raw_500_n32.md) |
| E005 | 2026-09-04 20:59:25 | GSM8K power policy: alpha 4, solve-box prompt, 128 tokens | [Training record](gsm8k_power_policy/reports/ALPHA4_BOX128.md) |

Dates come from existing activity logs, session records, and job submission
metadata. Historical experiments without reliable timestamps list only the
date. The archived MCMC comparison added to E003 references E001 and does not
receive another ID. The cancelled R1 three-shot / N=128 proposal did not run
as a formal experiment and has no separate ID.

The next experiment is **E006**. Use `E006_<description>` for new directories
or run names, and update this table and the registry before launching it.
Existing results and checkpoints are not moved when IDs are assigned.
