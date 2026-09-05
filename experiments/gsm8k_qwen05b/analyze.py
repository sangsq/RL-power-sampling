"""Summarize formal runs, bootstrap paired differences, and make figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from power_sampling.grading import grade


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())


def runs(prefix: str) -> list[dict]:
    return [load(f"{prefix}{seed}") for seed in range(3)]


def aggregate(group: list[dict]) -> dict:
    accuracy = np.array([run["summary"]["accuracy"] for run in group])
    tokens = np.array([run["summary"]["mean_sampled_tokens"] for run in group])
    return {
        "seeds": accuracy.tolist(),
        "mean_accuracy": float(accuracy.mean()),
        "sd_accuracy": float(accuracy.std(ddof=1)),
        "mean_sampled_tokens": float(tokens.mean()),
    }


def paired_ci(left: list[dict], right: list[dict], seed: int = 0) -> dict:
    # Average over seeds within each question, then resample questions.
    a = np.array([[row["correct"] for row in run["rows"]] for run in left]).mean(0)
    b = np.array([[row["correct"] for row in run["rows"]] for run in right]).mean(0)
    differences = a - b
    rng = np.random.default_rng(seed)
    draws = differences[rng.integers(0, len(differences), size=(10_000, len(differences)))].mean(1)
    return {
        "difference": float(differences.mean()),
        "cluster_bootstrap_95_ci": np.quantile(draws, [0.025, 0.975]).tolist(),
    }


def accuracy(name: str) -> float:
    return load(name)["summary"]["accuracy"]


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    formal = {
        "Base, T=1": runs("test_base_perexample_seed"),
        "Low temp, T=0.2": runs("test_lowtemp020_perexample_seed"),
        "Power, alpha=4": runs("test_power_a4_n2_b32_seed"),
        "Vote, 6 samples": runs("test_vote6_t020_seed"),
    }
    summary = {name: aggregate(group) for name, group in formal.items()}
    summary["paired_differences"] = {
        "power_minus_low_temp": paired_ci(formal["Power, alpha=4"], formal["Low temp, T=0.2"]),
        "vote_minus_power": paired_ci(formal["Vote, 6 samples"], formal["Power, alpha=4"]),
    }

    vote_scaling = {}
    for k in range(1, 7):
        plurality, oracle = [], []
        for run in formal["Vote, 6 samples"]:
            plurality_correct, oracle_correct = [], []
            for row in run["rows"]:
                answers = row["parsed_answers"][:k]
                counts = {answer: answers.count(answer) for answer in answers if answer is not None}
                winner = max(counts, key=lambda answer: (counts[answer], -answers.index(answer))) if counts else None
                selected = answers.index(winner) if winner is not None else 0
                plurality_correct.append(grade(row["responses"][selected], row["ground_truth"])[0])
                oracle_correct.append(any(grade(response, row["ground_truth"])[0] for response in row["responses"][:k]))
            plurality.append(np.mean(plurality_correct))
            oracle.append(np.mean(oracle_correct))
        vote_scaling[str(k)] = {
            "plurality_mean": float(np.mean(plurality)),
            "plurality_sd": float(np.std(plurality, ddof=1)),
            "pass_at_k_mean": float(np.mean(oracle)),
        }
    summary["vote_scaling"] = vote_scaling
    vote10 = load("test_vote10_t020_seed0")
    power10 = load("test_power_a4_n10_b32_seed0")
    pass10 = np.mean([
        any(grade(response, row["ground_truth"])[0] for response in row["responses"])
        for row in vote10["rows"]
    ])
    summary["exploratory"] = {
        "vote10_seed0_accuracy": vote10["summary"]["accuracy"],
        "vote10_seed0_pass_at_10": float(pass10),
        "vote10_seed0_sampled_tokens": vote10["summary"]["mean_sampled_tokens"],
        "power10_seed0_accuracy": power10["summary"]["accuracy"],
        "power10_seed0_sampled_tokens": power10["summary"]["mean_sampled_tokens"],
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))

    labels = list(formal)
    means = [summary[label]["mean_accuracy"] * 100 for label in labels]
    errors = [summary[label]["sd_accuracy"] * 100 for label in labels]
    costs = [summary[label]["mean_sampled_tokens"] for label in labels]
    colors = ["#7A869A", "#4C78A8", "#F58518", "#54A24B"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(labels, means, yerr=errors, capsize=4, color=colors)
    axes[0].set(ylabel="GSM8K accuracy (%)", title="Test accuracy, mean ± SD across 3 seeds", ylim=(0, 45))
    axes[0].tick_params(axis="x", rotation=18)
    for index, mean in enumerate(means):
        axes[0].text(index, mean + errors[index] + 0.8, f"{mean:.1f}", ha="center", fontsize=9)
    axes[1].scatter(costs, means, s=100, c=colors)
    for x, y, label in zip(costs, means, labels):
        axes[1].annotate(label, (x, y), xytext=(6, 4), textcoords="offset points", fontsize=9)
    axes[1].set(xlabel="Proposal-generated tokens / question", ylabel="Accuracy (%)", title="Accuracy–compute tradeoff")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "main_results.png", dpi=180)
    plt.close(fig)

    panels = [
        ("Proposal temperature", [0.10, 0.20, 0.25, 0.30], [
            accuracy("dev_lowtemp010_perexample_seed0"), accuracy("dev_lowtemp020_perexample_seed0"),
            accuracy("dev_lowtemp025_perexample_seed0"), accuracy("dev_lowtemp030_perexample_seed0")]),
        ("Power alpha", [2, 4, 8], [accuracy("dev_power_a2_n2_b32_seed0"),
            accuracy("dev_power_a4_n2_b32_seed0"), accuracy("dev_power_a8_n2_b32_seed0")]),
        ("MH steps / block", [1, 2, 4, 10], [accuracy("dev_power_a4_n1_b32_seed0"),
            accuracy("dev_power_a4_n2_b32_seed0"), accuracy("dev_power_a4_n4_b32_seed0"),
            accuracy("dev_power_a4_n10_b32_seed0")]),
        ("Block size", [16, 32, 64], [accuracy("dev_power_a4_n2_b16_seed0"),
            accuracy("dev_power_a4_n2_b32_seed0"), accuracy("dev_power_a4_n2_b64_seed0")]),
        ("Max generation length", [256, 512], [accuracy("dev_power_a4_n2_b32_len256_seed0"),
            accuracy("dev_power_a4_n2_b32_seed0")]),
        ("Suffix accounting", ["Full", "Official\noverlap"], [accuracy("dev_power_a4_n2_b32_seed0"),
            accuracy("dev_power_a4_n2_b32_official_overlap_seed0")]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for axis, (title, x, y) in zip(axes.flat, panels):
        axis.plot(x, np.array(y) * 100, marker="o", color="#F58518", linewidth=2)
        axis.set(title=title, ylabel="Dev accuracy (%)")
        axis.grid(alpha=0.25)
        axis.set_ylim(max(0, min(y) * 100 - 5), min(50, max(y) * 100 + 5))
    fig.suptitle("Development ablations (256 GSM8K train questions, seed 0)", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES / "ablations.png", dpi=180)
    plt.close(fig)

    k = np.arange(1, 7)
    plurality = np.array([vote_scaling[str(value)]["plurality_mean"] for value in k]) * 100
    plurality_sd = np.array([vote_scaling[str(value)]["plurality_sd"] for value in k]) * 100
    oracle = np.array([vote_scaling[str(value)]["pass_at_k_mean"] for value in k]) * 100
    fig, axis = plt.subplots(figsize=(6.5, 4.2))
    axis.errorbar(k, plurality, yerr=plurality_sd, marker="o", capsize=3, label="Plurality vote")
    axis.plot(k, oracle, marker="o", linestyle="--", label="Pass@k oracle ceiling")
    axis.scatter([10], [vote10["summary"]["accuracy"] * 100], marker="*", s=130, color="#1f77b4", label="Vote@10, seed 0")
    axis.scatter([10], [pass10 * 100], marker="*", s=130, color="#ff7f0e", label="Pass@10, seed 0")
    axis.set(xlabel="Samples per question", ylabel="GSM8K accuracy (%)", title="How repeated sampling scales")
    axis.set_xticks([1, 2, 3, 4, 5, 6, 10])
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "vote_scaling.png", dpi=180)
    plt.close(fig)

    parse_rates = [
        np.mean([sum(row["parsed_answer"] is not None for row in run["rows"]) / len(run["rows"]) for run in group]) * 100
        for group in formal.values()
    ]
    power_rows = [row for run in formal["Power, alpha=4"] for row in run["rows"]]
    acceptance = [
        np.mean([row["acceptance_rate"] for row in power_rows if row["correct"] == correct]) * 100
        for correct in (False, True)
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    axes[0].bar(labels, parse_rates, color=colors)
    axes[0].set(title="Parseable boxed answer", ylabel="Rate (%)", ylim=(0, 100))
    axes[0].tick_params(axis="x", rotation=25)
    low_lengths = [row["response_tokens"] for row in formal["Low temp, T=0.2"][0]["rows"]]
    power_lengths = [row["response_tokens"] for row in formal["Power, alpha=4"][0]["rows"]]
    axes[1].hist(low_lengths, bins=25, alpha=0.65, label="Low temp")
    axes[1].hist(power_lengths, bins=25, alpha=0.65, label="Power")
    axes[1].set(title="Response length, seed 0", xlabel="Tokens", ylabel="Questions")
    axes[1].legend(frameon=False)
    axes[2].bar(["Incorrect", "Correct"], acceptance, color=["#B0B8C4", "#F58518"])
    axes[2].set(title="Power MH acceptance", ylabel="Mean acceptance (%)", ylim=(0, 100))
    fig.tight_layout()
    fig.savefig(FIGURES / "behavior.png", dpi=180)
    plt.close(fig)

    low, power, vote = formal["Low temp, T=0.2"][0], formal["Power, alpha=4"][0], formal["Vote, 6 samples"][0]
    categories = {
        "Power fixes low-temperature": lambda a, b, c: not a and b,
        "Low-temperature fixes power": lambda a, b, c: a and not b,
        "Voting fixes power": lambda a, b, c: not b and c,
        "Power fixes voting": lambda a, b, c: b and not c,
    }
    lines = ["# Qualitative paired examples", "", "All examples use seed 0 and the identical raw question prompt.", ""]
    for title, predicate in categories.items():
        lines += [f"## {title}", ""]
        found = 0
        for low_row, power_row, vote_row in zip(low["rows"], power["rows"], vote["rows"]):
            if not predicate(low_row["correct"], power_row["correct"], vote_row["correct"]):
                continue
            chosen_vote = vote_row["responses"][vote_row["selected"]]
            lines += [
                f"### Example {found + 1}", "", f"**Question.** {low_row['question']}", "",
                f"**Ground truth:** `{low_row['ground_truth']}`", "",
                f"**Low-temperature ({'correct' if low_row['correct'] else 'wrong'}):**\n\n{low_row['response'][:900]}", "",
                f"**Power ({'correct' if power_row['correct'] else 'wrong'}):**\n\n{power_row['response'][:900]}", "",
                f"**Six-sample vote ({'correct' if vote_row['correct'] else 'wrong'}):**\n\n{chosen_vote[:900]}", "",
            ]
            found += 1
            if found == 3:
                break
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "QUALITATIVE_EXAMPLES.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
