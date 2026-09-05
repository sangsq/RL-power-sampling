"""Plot complete pass@N curves and question-paired bootstrap comparisons."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from power_sampling.grading import load_jsonl
from experiments.gsm8k_power_policy.pass_at_n import CONDITIONS, ROOT, pass_at_k

LABELS = {"base_t1": "Base T=1", "base_t03": "Base T=0.3", "policy_t1": "Policy T=1"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="raw_500_n32")
    args = parser.parse_args()
    folder = ROOT / "results/pass_at_n" / args.name
    config = json.loads((folder / "config.json").read_text())
    records = {name: load_jsonl(folder / f"{name}.jsonl") for name in CONDITIONS}
    n = config["samples"]
    for rows in records.values():
        assert [r["index"] for r in rows] == config["indices"]
        assert all(len(r["samples"]) == n for r in rows)
    rng = np.random.default_rng(0)
    size = len(config["indices"])
    # Reuse question weights across all conditions and N for paired uncertainty.
    weights = rng.multinomial(size, np.full(size, 1 / size), size=10000) / size
    summary = {"config": config, "bootstrap": "10000 paired question resamples, seed 0; pointwise 95% intervals",
               "conditions": {}, "comparisons": {}}
    matrices = {}
    for name, rows in records.items():
        metrics = {}
        for metric, key in [("strict", "n_correct"), ("numeric", "n_answer_correct")]:
            matrix = np.array([[pass_at_k(n, r[key], k) for k in range(1, n + 1)] for r in rows])
            matrices[name, metric] = matrix
            ci = np.quantile(weights @ matrix, [.025, .975], axis=0)
            metrics[metric] = {"pass_at_n": matrix.mean(axis=0).tolist(), "ci95": ci.T.tolist()}
        samples = [s for r in rows for s in r["samples"]]
        metrics.update(questions=len(rows), samples=len(samples),
                       boxed_parse_rate=float(np.mean([s["boxed"] is not None for s in samples])),
                       numeric_parse_rate=float(np.mean([s["answer"] is not None for s in samples])),
                       mean_tokens=float(np.mean([len(s["token_ids"]) for s in samples])),
                       truncated_rate=float(np.mean([s["finish_reason"] == "length" for s in samples])),
                       mean_unique_responses=float(np.mean([len({s["text"] for s in r["samples"]}) for r in rows])),
                       elapsed_seconds=rows[-1]["elapsed_seconds"])
        summary["conditions"][name] = metrics
    for baseline in ("base_t1", "base_t03"):
        summary["comparisons"][baseline] = {}
        for metric in ("strict", "numeric"):
            diff = matrices["policy_t1", metric] - matrices[baseline, metric]
            summary["comparisons"][baseline][metric] = {
                "difference": diff.mean(axis=0).tolist(),
                "ci95": np.quantile(weights @ diff, [.025, .975], axis=0).T.tolist()}
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, metric, title in zip(axes, ("strict", "numeric"), ("Strict boxed answers", "Explicit numeric answers")):
        for name, values in summary["conditions"].items():
            s = values[metric]
            x = np.arange(1, n + 1)
            line, = ax.plot(x, 100 * np.array(s["pass_at_n"]), label=LABELS[name])
            ci = 100 * np.array(s["ci95"])
            ax.fill_between(x, ci[:, 0], ci[:, 1], color=line.get_color(), alpha=.12)
        ax.set(title=title, xlabel="N (samples per question)", ylabel="Pass@N (%)", ylim=(0, 100), xlim=(1, n))
        ax.set_xscale("log", base=2)
        ticks = [k for k in (1, 2, 4, 8, 16, 32) if k <= n]
        ax.set_xticks(ticks, [str(k) for k in ticks])
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
    fig.suptitle(f"GSM8K test: {size} fixed questions, raw prompt, {n} samples, 512-token cap")
    fig.tight_layout()
    figure = ROOT / "figures" / f"pass_at_n_{args.name}.png"
    fig.savefig(figure, dpi=180)
    plt.close(fig)

    ks = [k for k in (1, 2, 4, 8, 16, 32) if k <= n]
    lines = ["# E004 · Raw-prompt pass@N comparison", "",
             f"All three conditions use the same {size} GSM8K test questions, selected without replacement "
             f"using Python random seed {config['subset_seed']}; sorted original indices are saved in config.json. "
             f"Each question has {n} samples per condition ({size * n * 3:,} responses total).", "",
             "The prompt is exactly the raw question used during policy training, with no chat template, "
             "few-shot demonstrations or R1 tags. Generation ends at EOS or 512 tokens. All conditions use "
             "vLLM, BF16, top-p=1, top-k disabled, no penalties or stop strings. Per-question seeds are "
             "42000 + original test index, paired across conditions. The final 158-update policy adapter "
             "is loaded directly through vLLM LoRA (adapter weights cast to BF16 for inference); no merging or further training. "
             "This is a new matched-vLLM evaluation, so differences from earlier HF results also reflect "
             "the subset, decoder and random draws.", "",
             "For c successes among m samples, pass@N = 1 − C(m−c,N)/C(m,N), averaged over questions. "
             "It measures whether an oracle could find a correct candidate, not majority-vote accuracy. "
             "At N=m it is the fraction of questions with at least one observed success. "
             "The numeric extraction rule was fixed in the previous experiment and is applied unchanged; "
             "strict boxed scoring remains separate. Numeric correctness does not validate the reasoning.", ""]
    for metric, title in [("strict", "Strict boxed"), ("numeric", "Explicit numeric")]:
        lines += [f"## {title}", "", "| Condition | " + " | ".join(f"Pass@{k}" for k in ks) + " |",
                  "|---|" + "---:|" * len(ks)]
        for name, s in summary["conditions"].items():
            lines.append("| " + LABELS[name] + " | " + " | ".join(f"{s[metric]['pass_at_n'][k-1]:.2%}" for k in ks) + " |")
        lines += ["", "Policy minus baseline, percentage points with paired question-bootstrap 95% intervals:", ""]
        for baseline in ("base_t1", "base_t03"):
            s = summary["comparisons"][baseline][metric]
            for k in (1, n):
                lo, hi = s["ci95"][k-1]
                lines.append(f"- vs {LABELS[baseline]}, N={k}: {100*s['difference'][k-1]:+.2f} "
                             f"[{100*lo:+.2f}, {100*hi:+.2f}].")
        lines.append("")
    lines += ["", f"![Pass@N curves](../figures/{figure.name})", "", "## Behavior and cost", "",
              "| Condition | Boxed parse | Numeric parse | Mean tokens | Truncated | Unique texts/question | Generation + scoring minutes |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name, s in summary["conditions"].items():
        lines.append(f"| {LABELS[name]} | {s['boxed_parse_rate']:.2%} | {s['numeric_parse_rate']:.2%} | "
                     f"{s['mean_tokens']:.1f} | {s['truncated_rate']:.2%} | {s['mean_unique_responses']:.2f} | "
                     f"{s['elapsed_seconds']/60:.2f} |")
    lines += ["", "## Limits", "",
              "One trained checkpoint and one decoding seed protocol; no seed uncertainty is included. "
              "Intervals are pointwise, not simultaneous bands or multiplicity-adjusted claims. "
              "Previously viewed test results informed the decision to run this evaluation, so this is "
              "exploratory, not a pristine held-out confirmation. Finite sample pass@N does not measure "
              "the full answer support. No checkpoint or subset was selected by these outcomes.", "",
              f"Full N=1…{n} arrays, intervals, indices and settings: "
              f"[summary.json](../results/pass_at_n/{args.name}/summary.json). "
              "All raw responses, generated token IDs and per-sample grades are retained in the adjacent JSONL files."]
    (ROOT / "reports" / f"PASS_AT_N_{args.name}.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({name: {metric: [s[metric]["pass_at_n"][k-1] for k in ks] for metric in ("strict", "numeric")}
                      for name, s in summary["conditions"].items()}, indent=2))


if __name__ == "__main__":
    main()
