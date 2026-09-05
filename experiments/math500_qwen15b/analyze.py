"""Aggregate complete MATH500 shards, quantify effects, and plot results."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from math500_sampling.paper_grading import paper_grade


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CONDITIONS = ("base", "low_temp", "power")
LABELS = {"base": "Base (T=1)", "low_temp": "Low temperature (T=0.25)", "power": "Power sampling"}
COLORS = {"base": "#4C78A8", "low_temp": "#F28E2B", "power": "#59A14F"}
PAPER_7B = {"base": 0.496, "low_temp": 0.690, "power": 0.748}


def load_groups() -> dict[tuple[str, int], dict[int, dict]]:
    groups = defaultdict(dict)
    for path in sorted(RESULTS.glob("*.json")):
        if path.name == "summary.json":
            continue
        payload = json.loads(path.read_text())
        config = payload.get("config", {})
        condition, seed = config.get("condition"), config.get("seed")
        if condition not in CONDITIONS or seed is None:
            continue
        for row in payload["rows"]:
            index = row["dataset_index"]
            if index in groups[condition, seed]:
                raise RuntimeError(f"Duplicate {condition=} {seed=} dataset_index={index}")
            groups[condition, seed][index] = {
                **row,
                "paper_correct": paper_grade(row["parsed_answer"], row["ground_truth"]),
            }
    return dict(groups)


def bootstrap_delta(left: np.ndarray, right: np.ndarray, draws: int = 10_000) -> tuple[float, float, float]:
    per_question = (left.astype(float) - right.astype(float)).mean(axis=0)
    rng = np.random.default_rng(0)
    indices = rng.integers(0, len(per_question), size=(draws, len(per_question)))
    samples = per_question[indices].mean(axis=1)
    return per_question.mean(), *np.quantile(samples, [0.025, 0.975])


def main() -> None:
    groups = load_groups()
    complete = {key: rows for key, rows in groups.items() if set(rows) == set(range(500))}
    per_seed = []
    for (condition, seed), rows in sorted(complete.items()):
        ordered = [rows[index] for index in range(500)]
        per_seed.append(
            {
                "condition": condition,
                "seed": seed,
                "accuracy": np.mean([row["paper_correct"] for row in ordered]),
                "math_verify_accuracy": np.mean([row["correct"] for row in ordered]),
                "grader_disagreements": sum(row["correct"] != row["paper_correct"] for row in ordered),
                "parse_rate": np.mean([row["parsed_answer"] is not None for row in ordered]),
                "response_tokens": np.mean([row["response_tokens"] for row in ordered]),
                "sampled_tokens": np.mean([row["sampled_tokens"] for row in ordered]),
                "acceptance_rate": np.mean([row["acceptance_rate"] for row in ordered]),
            }
        )

    aggregates = {}
    for condition in CONDITIONS:
        rows = [row for row in per_seed if row["condition"] == condition]
        if rows:
            aggregates[condition] = {
                "seeds": [row["seed"] for row in rows],
                **{
                    metric: float(np.mean([row[metric] for row in rows]))
                    for metric in (
                        "accuracy", "math_verify_accuracy", "parse_rate", "response_tokens",
                        "sampled_tokens", "acceptance_rate", "grader_disagreements",
                    )
                },
                "accuracy_std": float(np.std([row["accuracy"] for row in rows], ddof=1)) if len(rows) > 1 else None,
            }

    comparisons = {}
    for condition in ("low_temp", "power"):
        seeds = sorted(
            {seed for method, seed in complete if method == condition}
            & {seed for method, seed in complete if method == "base"}
        )
        if seeds:
            left = np.array(
                [[complete[condition, seed][index]["paper_correct"] for index in range(500)] for seed in seeds]
            )
            right = np.array(
                [[complete["base", seed][index]["paper_correct"] for index in range(500)] for seed in seeds]
            )
            estimate, low, high = bootstrap_delta(left, right)
            comparisons[f"{condition}_minus_base"] = {
                "seeds": seeds,
                "difference": float(estimate),
                "ci95": [float(low), float(high)],
            }

    paired_seeds = sorted(
        {seed for method, seed in complete if method == "power"}
        & {seed for method, seed in complete if method == "low_temp"}
    )
    if paired_seeds:
        power = np.array(
            [[complete["power", seed][index]["paper_correct"] for index in range(500)] for seed in paired_seeds]
        )
        low_temp = np.array(
            [[complete["low_temp", seed][index]["paper_correct"] for index in range(500)] for seed in paired_seeds]
        )
        estimate, low, high = bootstrap_delta(power, low_temp)
        comparisons["power_minus_low_temp"] = {
            "seeds": paired_seeds,
            "difference": float(estimate),
            "ci95": [float(low), float(high)],
        }

    transitions = {}
    for seed in paired_seeds:
        low_rows, power_rows = complete["low_temp", seed], complete["power", seed]
        counts = defaultdict(int)
        for index in range(500):
            low_correct = low_rows[index]["paper_correct"]
            power_correct = power_rows[index]["paper_correct"]
            label = {
                (True, True): "both_correct",
                (False, False): "both_wrong",
                (False, True): "power_gain",
                (True, False): "power_loss",
            }[low_correct, power_correct]
            counts[label] += 1
        transitions[f"seed{seed}"] = dict(counts)

    summary = {
        "complete_groups": len(complete),
        "partial_groups": {f"{condition}_seed{seed}": len(rows) for (condition, seed), rows in groups.items() if len(rows) < 500},
        "per_seed": per_seed,
        "aggregates": aggregates,
        "comparisons": comparisons,
        "low_temp_to_power_transitions": transitions,
        "paper_qwen_math_7b": PAPER_7B,
        "primary_grader": "paper-compatible MATH grader",
        "sensitivity_grader": "math_verify",
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    if all(condition in aggregates for condition in CONDITIONS):
        x = np.arange(len(CONDITIONS))
        values = [aggregates[condition]["accuracy"] for condition in CONDITIONS]
        fig, axis = plt.subplots(figsize=(7.2, 4.5))
        bars = axis.bar(x, values, color=[COLORS[condition] for condition in CONDITIONS], width=0.62)
        axis.scatter(x, [PAPER_7B[condition] for condition in CONDITIONS], marker="D", color="#222222", label="Paper: 7B")
        for index, bar in enumerate(bars):
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012, f"{values[index]:.1%}", ha="center")
        axis.set_xticks(x, [LABELS[condition] for condition in CONDITIONS])
        axis.set(ylabel="MATH500 accuracy", ylim=(0, 0.85), title="Power sampling: Qwen2.5-Math-1.5B reproduction")
        axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
        fig.tight_layout()
        (ROOT / "figures").mkdir(exist_ok=True)
        fig.savefig(ROOT / "figures" / "main_results.png", dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
