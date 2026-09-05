"""Regrade archived full-test MCMC runs using the policy experiment's metrics."""

import hashlib
import json

import numpy as np

from power_sampling.grading import grade, ground_truth, load_jsonl
from experiments.gsm8k_power_policy.analyze import ROOT, extract_explicit_answer


def main():
    archive = ROOT.parent / "gsm8k_qwen05b/results"
    dataset = load_jsonl(ROOT.parent / "gsm8k_qwen05b/data/gsm8k/test.jsonl")
    output = {"description": "Archived vLLM runs; three decoding seeds per condition; regraded without resampling",
              "conditions": {}, "sources": {}}
    for name, prefix in [("Archived base T=1", "test_base_perexample_seed"),
                         ("Archived base T=0.2", "test_lowtemp020_perexample_seed"),
                         ("MCMC power alpha=4", "test_power_a4_n2_b32_seed")]:
        seeds = []
        for seed in range(3):
            path = archive / f"{prefix}{seed}.json"
            run = json.loads(path.read_text())
            rows = run["rows"]
            assert [r["question"] for r in rows] == [r["question"] for r in dataset]
            assert [r["ground_truth"] for r in rows] == [ground_truth(r) for r in dataset]
            correct, numeric, boxed, parsed = [], [], [], []
            for row in rows:
                ok, box = grade(row["response"], row["ground_truth"])
                assert bool(ok) == row["correct"]
                answer = extract_explicit_answer(row["response"])
                correct.append(ok)
                boxed.append(box is not None)
                parsed.append(answer is not None)
                numeric.append(answer is not None and grade("\\boxed{" + answer + "}", row["ground_truth"])[0])
            seeds.append({"seed": seed, "questions": len(rows), "accuracy": float(np.mean(correct)),
                          "answer_accuracy": float(np.mean(numeric)), "parse_rate": float(np.mean(boxed)),
                          "answer_parse_rate": float(np.mean(parsed)),
                          "mean_response_tokens": run["summary"]["mean_response_tokens"],
                          "mean_sampled_tokens": run["summary"]["mean_sampled_tokens"],
                          "mean_acceptance_rate": run["summary"]["mean_acceptance_rate"],
                          "elapsed_seconds": run["summary"]["elapsed_seconds"]})
            output["sources"][path.name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                            "config": run["config"]}
        output["conditions"][name] = {
            "seeds": seeds,
            "mean": {key: float(np.mean([s[key] for s in seeds])) for key in seeds[0] if key not in ("seed", "questions")}}
    path = ROOT / "results/mcmc_comparison.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    lines = ["## MCMC power sampling comparison (supplement)", "",
             "Archived vLLM results cover all 1,319 GSM8K test questions with three decoding seeds. "
             "The original responses were regraded using the policy experiment's strict boxed and explicit "
             "numeric answer rules. No responses were resampled or seeds selected by their outcomes.", "",
             "| Condition | Backend / seeds | Strict accuracy | Numeric accuracy | Output tokens/question | Generated tokens/question |",
             "|---|---|---:|---:|---:|---:|"]
    previous = json.loads((ROOT / "results/summary.json").read_text())["evaluations"]
    for name, label in [("test_base", "Base T=1"), ("test_t05", "Base T=0.5"),
                        ("test_t02", "Base T=0.2"), ("test_policy", "Power policy α=2, T=1")]:
        s = previous[name]
        lines.append(f"| {label} | HF / 1 | {s['accuracy']:.2%} | {s['answer_accuracy']:.2%} | "
                     f"{s['mean_length']:.1f} | {s['mean_length']:.1f} |")
    for name, condition in output["conditions"].items():
        s = condition["mean"]
        lines.append(f"| {name} | vLLM / mean of 3 | {s['accuracy']:.2%} | {s['answer_accuracy']:.2%} | "
                     f"{s['mean_response_tokens']:.1f} | {s['mean_sampled_tokens']:.1f} |")
    lines += ["", "MCMC uses the frozen base, power α=4, proposal T=0.25, block size 32, two "
              "Metropolis–Hastings updates per block, a 512-token cap, and the `full` suffix probability ratio. "
              "Each block first extends the sequence from the proposal, then selects a random suffix boundary "
              "and resamples that suffix. Acceptance uses the base and proposal sequence probabilities: "
              "`min(1, exp(α·Δlog p0 − Δlog q))`. This finite-step autoregressive MCMC implementation "
              "does not produce independent samples from the exact power distribution.", "",
              "MCMC requires multiple proposal and reference-scoring passes per question at inference time, "
              "without training. The policy first trains an adapter and then uses ordinary sampling. "
              "MCMC generated-token counts include extensions and rejected proposals, but exclude "
              "reference-scoring forward passes. Policy costs do not amortize the earlier 27m 35s training run.", "",
              "**This is a historical algorithm reference; its differences from the policy do not isolate "
              "method quality.** Power α differs (4 versus 2), as do the backend (vLLM versus HF), random "
              "seeds, and number of seeds. MCMC settings were previously selected on development data. "
              "Archived vLLM base and low-temperature results are included to expose differences between "
              "backends and experiment batches. The archived MCMC files do not record an explicit "
              "`seed_policy` field; original configs and file hashes are retained in the supplement.", "",
              "[Machine-readable summary and individual seeds](../results/mcmc_comparison.json) · "
              "[Original MCMC report](../../gsm8k_qwen05b/reports/REPORT.md) · "
              "[Sampler implementation](../../../src/power_sampling/sampler.py)", ""]
    appendix = "\n".join(lines)
    (ROOT / "reports/MCMC_COMPARISON.md").write_text("# Algorithm reference for the single-answer experiment\n\n" + appendix)
    marker = "<!-- MCMC_COMPARISON -->"
    for name in ("REPORT.md", "FINDINGS.md"):
        report = ROOT / "reports" / name
        original = report.read_text().split(marker, 1)[0].rstrip()
        report.write_text(original + "\n\n" + marker + "\n" + appendix)
    print(json.dumps({name: s["mean"] for name, s in output["conditions"].items()}, indent=2))


if __name__ == "__main__":
    main()
