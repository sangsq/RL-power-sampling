"""Summarize HF evaluations, paired uncertainty, and the training trajectory."""

import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from power_sampling.grading import grade, last_boxed

ROOT = Path(__file__).resolve().parent


def extract_explicit_answer(response):
    """Post-hoc format sensitivity; do not infer an answer from arbitrary numbers."""
    boxed = last_boxed(response)
    if boxed is not None:
        return boxed
    if "####" in response:
        tail = response.rsplit("####", 1)[1]
    else:
        matches = list(re.finditer(r"\b(?:the\s+)?(?:final\s+)?answer\s*(?:is\b|:|=)", response, re.I))
        if not matches:
            return None
        tail = response[matches[-1].end():]
    match = re.match(r"\s*[:=$*\s]*([-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:/\d+(?:\.\d+)?)?)", tail)
    return match.group(1).rstrip(",") if match else None


def relaxed_correct(row):
    return [answer is not None and grade("\\boxed{" + answer + "}", row["ground_truth"])[0]
            for answer in map(extract_explicit_answer, row["responses"])]


def summarize(rows):
    correctness = np.array([row["correct"] for row in rows], dtype=float)
    lengths = np.array([row["lengths"] for row in rows])
    result = {
        "questions": len(rows), "samples_per_question": correctness.shape[1],
        "accuracy": float(correctness.mean()), "pass_at_k": float(correctness.max(axis=1).mean()),
        "parse_rate": float(np.mean([x is not None for row in rows for x in row["parsed_answers"]])),
        "mean_length": float(lengths.mean()), "median_length": float(np.median(lengths)),
        "eos_rate": float(np.mean([row["eos"] for row in rows])),
        "first_token_eos_rate": float(np.mean([row["first_token_eos"] for row in rows])),
        "short_rate": float((lengths < 16).mean()),
        "mean_unique_responses": float(np.mean([len(set(row["responses"])) for row in rows])),
        "mean_unique_parsed_answers": float(np.mean([
            len({x for x in row["canonical_answers"] if x is not None}) for row in rows])),
    }
    relaxed = np.array([relaxed_correct(row) for row in rows], dtype=float)
    result.update(answer_accuracy=float(relaxed.mean()), answer_pass_at_k=float(relaxed.max(axis=1).mean()),
                  answer_parse_rate=float(np.mean([extract_explicit_answer(x) is not None
                                                   for row in rows for x in row["responses"]])),
                  mean_unique_explicit_answers=float(np.mean([
                      len({extract_explicit_answer(x) for x in row["responses"]} - {None}) for row in rows])))
    if "policy_logprobs" in rows[0]:
        lp, lp0 = (np.array([row[key] for row in rows]) for key in ["policy_logprobs", "base_logprobs"])
        result.update(objective=float((2 * lp0 - lp).mean()), kl_estimate=float((lp - lp0).mean()),
                      sequence_entropy=float(-lp.mean()), base_logprob=float(lp0.mean()))
    return result


def paired(left, right, relaxed=False):
    assert [r["question"] for r in left] == [r["question"] for r in right]
    correct = relaxed_correct if relaxed else lambda row: row["correct"]
    difference = np.array([np.mean(correct(a)) - np.mean(correct(b)) for a, b in zip(left, right)])
    rng = np.random.default_rng(0)
    samples = difference[rng.integers(0, len(difference), size=(10000, len(difference)))].mean(axis=1)
    return {"difference": float(difference.mean()), "ci95": np.quantile(samples, [.025, .975]).tolist()}


def write_report(summaries, comparisons, evaluations, relaxed_comparisons):
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    train_path = ROOT / "results/alpha2_seed0/training_summary.json"
    if not train_path.exists():
        return
    training = json.loads(train_path.read_text())
    log = ROOT / "results/alpha2_seed0/metrics.jsonl"
    metrics = [json.loads(x) for x in log.read_text().splitlines()]
    lines = ["# Fixed-alpha power policy on GSM8K", "", "## Protocol", "",
             "Qwen2.5-0.5B; frozen base; all-linear LoRA rank 8, LoRA scaling alpha 16; power alpha 2. "
             "HF Transformers/PEFT, BF16/SDPA, raw questions, 512-token cap, strict boxed-answer grading. "
             "Each fresh batch contains four questions and four responses per question. "
             "Sequence log-probabilities include EOS; the update uses a leave-one-out baseline, "
             "without correctness rewards or length/advantage normalization.", "",
             f"Training completed **{training['completed_steps']} updates in {training['wall_seconds']/60:.2f} minutes** "
             f"(termination: `{training['reason']}`). The final completed checkpoint was used; "
             "no checkpoint was selected by evaluation accuracy. Development and test evaluation ran separately.", "",
             f"Training used seed 0 and processed {4 * training['completed_steps']} question presentations "
             "from the 1,024 selected training questions. "
             "Evaluation used seed 2026 (deterministic per-batch offsets), generation batch size 64, "
             "and all 1,319 test questions. The four test conditions differ only in temperature or adapter.", "",
             f"The run sampled {sum(x['sampled_tokens'] for x in metrics):,} training tokens; "
             f"peak allocated GPU memory was {max(x['peak_memory_gb'] for x in metrics):.2f} GB. "
             f"Gradient clipping triggered on {np.mean([x['gradient_clipped'] for x in metrics]):.1%} of updates.", "",
             "## Evaluation", "",
             "| Condition | Questions | Samples/question | Accuracy | Pass@k | Parse rate | Mean tokens | EOS rate |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, s in summaries.items():
        lines.append(f"| {name} | {s['questions']} | {s['samples_per_question']} | {s['accuracy']:.2%} | "
                     f"{s['pass_at_k']:.2%} | {s['parse_rate']:.2%} | {s['mean_length']:.1f} | {s['eos_rate']:.2%} |")
    lines += ["", "## Paired test comparisons", ""]
    for name, comparison in comparisons.items():
        low, high = comparison['ci95']
        lines.append(f"- Policy minus {name}: {100*comparison['difference']:+.2f} percentage points "
                     f"(question-bootstrap 95% CI [{100*low:+.2f}, {100*high:+.2f}]).")
    lines += ["", "## Post-hoc format sensitivity", "",
              "The original boxed-answer metric remains primary. After inspecting development outputs, "
              "we additionally scored explicit `####` and `answer is/:/=` markers for every condition. "
              "See [the frozen extraction rules](GRADING_SENSITIVITY.md). This measures final-answer agreement, "
              "not validity of the derivation.", "",
              "| Condition | Numeric answer accuracy | Parse rate | Numeric pass@k |",
              "|---|---:|---:|---:|"]
    for name, s in summaries.items():
        lines.append(f"| {name} | {s['answer_accuracy']:.2%} | {s['answer_parse_rate']:.2%} | {s['answer_pass_at_k']:.2%} |")
    lines.append("")
    for name, comparison in relaxed_comparisons.items():
        low, high = comparison['ci95']
        lines.append(f"- Numeric policy minus {name}: {100*comparison['difference']:+.2f} percentage points "
                     f"(question-bootstrap 95% CI [{100*low:+.2f}, {100*high:+.2f}]).")
    lines += ["", "## Held-out objective and behavior", ""]
    for name, s in summaries.items():
        if "objective" in s:
            lines.append(f"- {name}: J={s['objective']:.3f}, estimated KL={s['kl_estimate']:.3f}, "
                         f"sequence entropy={s['sequence_entropy']:.3f}, first-token EOS={s['first_token_eos_rate']:.2%}, "
                         f"short (<16-token) responses={s['short_rate']:.2%}, "
                         f"mean unique responses={s['mean_unique_responses']:.2f}, "
                         f"mean unique explicit answers={s['mean_unique_explicit_answers']:.2f}.")
    timing_path = ROOT / "results/evaluation_timing.json"
    if timing_path.exists():
        timings = json.loads(timing_path.read_text())
        lines += ["", "## Observed inference cost", "",
                  "Single-run batched generation timings on GPU 1, batch size 64, excluding model loading. "
                  "These include decoding and writing outputs and are not a controlled latency benchmark.", "",
                  "| Condition | Total seconds | Amortized ms/question |", "|---|---:|---:|"]
        for name in ("test_base", "test_t05", "test_t02", "test_policy"):
            if name in timings:
                seconds = timings[name]["session_seconds"]
                lines.append(f"| {name} | {seconds:.1f} | {1000 * seconds / timings[name]['total']:.1f} |")
    lines += ["", "![Training trajectory](../figures/alpha2_seed0_training.png)", "",
              "![Test comparisons](../figures/test_comparison.png)", "", "## Limits", "",
              "This is a single training seed and a single decoding seed. Question-bootstrap intervals "
              "do not include seed uncertainty. The wall-clock budget takes precedence over the planned "
              "256 updates. Training-batch objective values mix different questions; use held-out "
              "diagnostics for before/after comparisons. An improving objective alone does not prove "
              "closeness to the exact power distribution or improved reasoning. "
              "Historical vLLM/MCMC results are not used as matched-backend baselines.", "",
              "## Validation", "",
              "The CPU suite passed all 18 tests, including an enumerated RLOO gradient check, "
              "EOS/padding masks, microbatch gradient equivalence, frozen-reference invariance, "
              "adapter/optimizer/RNG restoration and explicit-answer extraction. A real-model "
              "alpha=1 identity smoke run produced zero gradient and byte-identical adapter weights. "
              "Five short alpha=2 smoke updates and a two-update 512-token probe completed before "
              "the formal run. Training/development question separation and the unchanged formal "
              "training source hashes were verified.", "",
              "Raw responses, token IDs, configs, training scores and checkpoints are saved alongside this report."]
    (reports / "REPORT.md").write_text("\n".join(lines) + "\n")
    if "test_policy" in evaluations and "test_base" in evaluations:
        examples = ["# Paired test examples", "",
                    "First three numeric-answer flips in each direction under the post-hoc extraction rule. "
                    "Correctness here refers to the extracted final answer, not the derivation.", ""]
        for desired, title in [(True, "Base incorrect → policy correct"), (False, "Base correct → policy incorrect")]:
            examples += [f"## {title}", ""]
            selected = 0
            for base, policy in zip(evaluations['test_base'], evaluations['test_policy']):
                if relaxed_correct(policy)[0] != desired or relaxed_correct(base)[0] == desired:
                    continue
                examples += [f"### Question {policy['index']}", "", policy['question'], "",
                             f"Target: {policy['ground_truth']}", "", "Base:", "````text", base['responses'][0],
                             "````", "", "Policy:", "````text", policy['responses'][0], "````", ""]
                selected += 1
                if selected == 3:
                    break
        (reports / "QUALITATIVE_EXAMPLES.md").write_text("\n".join(examples))


def main():
    summaries, evaluations = {}, {}
    for folder in sorted((ROOT / "results/evaluation").glob("*")):
        if not (folder / "summary.json").exists():
            continue
        rows = [json.loads(x) for x in (folder / "responses.jsonl").read_text().splitlines()]
        summaries[folder.name] = summarize(rows)
        (folder / "summary.json").write_text(json.dumps(summaries[folder.name], indent=2))
        evaluations[folder.name] = rows
    comparisons, relaxed_comparisons = {}, {}
    if "test_policy" in evaluations:
        for baseline in ("test_base", "test_t05", "test_t02"):
            if baseline in evaluations:
                comparisons[baseline] = paired(evaluations["test_policy"], evaluations[baseline])
                relaxed_comparisons[baseline] = paired(evaluations["test_policy"], evaluations[baseline], relaxed=True)
    output = {"evaluations": summaries, "policy_minus_baseline": comparisons,
              "posthoc_numeric_policy_minus_baseline": relaxed_comparisons}
    (ROOT / "results/summary.json").write_text(json.dumps(output, indent=2))
    figures = ROOT / "figures"
    figures.mkdir(exist_ok=True)
    conditions = [name for name in ("test_base", "test_t05", "test_t02", "test_policy") if name in summaries]
    if len(conditions) == 4:
        x = np.arange(4)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for offset, key, label in [(-.2, "accuracy", "Strict boxed (primary)"),
                                   (.2, "answer_accuracy", "Explicit numeric (post-hoc)")]:
            bars = axes[0].bar(x + offset, [100 * summaries[n][key] for n in conditions], .4, label=label)
            axes[0].bar_label(bars, fmt="%.1f", fontsize=8)
        axes[0].set(ylabel="Test accuracy (%)", ylim=(0, 65))
        axes[0].legend(fontsize=8)
        bars = axes[1].bar(x, [summaries[n]["mean_length"] for n in conditions], color="#64879a")
        axes[1].bar_label(bars, fmt="%.0f")
        axes[1].set(ylabel="Mean response tokens", ylim=(0, 320))
        for ax in axes:
            ax.set_xticks(x, ["Base T=1", "Base T=.5", "Base T=.2", "Policy T=1"], rotation=15)
        fig.tight_layout()
        fig.savefig(figures / "test_comparison.png", dpi=160)
        plt.close(fig)
    for log in (ROOT / "results").glob("*/metrics.jsonl"):
        rows = [json.loads(x) for x in log.read_text().splitlines()]
        if not rows:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(10, 6))
        for ax, key in zip(axes.flat, ["objective", "mean_length", "kl_estimate", "eos_rate"]):
            ax.plot([x["step"] for x in rows], [x[key] for x in rows], alpha=.7)
            ax.set(xlabel="Update", ylabel=key)
            ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(figures / f"{log.parent.name}_training.png", dpi=160)
        plt.close(fig)
    write_report(summaries, comparisons, evaluations, relaxed_comparisons)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
