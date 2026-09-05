"""Build and execute the concise result-presentation notebook."""

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "results.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python", "version": "3.12"}
nb.cells = [
    markdown(r"""
[ask] [en] Last updated: 2026-09-04

**What it does**: Presents Qwen2.5-0.5B GSM8K question-only results for base, low-temperature, sequence-level power sampling, and repeated-sampling controls.

**What it saves**: none; all tables and figures are displayed from previously saved experiment artifacts.

**What it uses**: `../results/summary.json` and PNG files in `../figures/`, produced by `../analyze.py`.
"""),
    markdown(r"""
# Reasoning with Sampling: small-scale result notebook

The central question is whether blockwise MCMC sampling from a distribution proportional to
$p(x)^\alpha$ improves a frozen 0.5B model—and whether it beats simpler ways to spend the same inference budget.
"""),
    code("""
import json
from pathlib import Path
from IPython.display import Image, display

REPO = next(
    parent for parent in (Path.cwd(), *Path.cwd().parents)
    if (parent / "experiments" / "gsm8k_qwen05b" / "analyze.py").is_file()
)
ROOT = REPO / "experiments" / "gsm8k_qwen05b"
summary = json.loads((ROOT / "results" / "summary.json").read_text())
methods = ["Base, T=1", "Low temp, T=0.2", "Power, alpha=4", "Vote, 6 samples"]
for method in methods:
    row = summary[method]
    print(f"{method:22s} {100*row['mean_accuracy']:5.2f} ± {100*row['sd_accuracy']:.2f}%"
          f"   tokens={row['mean_sampled_tokens']:.0f}")
"""),
    markdown(r"""
## Formal three-seed comparison

Power sampling uses $\alpha=4$, proposal temperature 0.25, 32-token blocks, two MH updates per block,
and a 512-token response limit. Seeds are derived independently per question; bars show mean ± sample SD.
"""),
    code("display(Image(filename=ROOT / 'figures' / 'main_results.png', width=1000))"),
    markdown(r"""
Power improves over one selected low-temperature response by **3.87 percentage points**
(question-clustered 95% bootstrap CI **[2.40, 5.31]**). Six-sample voting improves over power by
**7.03 points** (**[5.48, 8.59]**) at nearly identical proposal-generated-token count.
"""),
    markdown("## Development ablations"),
    code("display(Image(filename=ROOT / 'figures' / 'ablations.png', width=1000))"),
    markdown(r"""
The useful region is $\alpha\in[4,8]$, block size 32, and at least 512 output tokens. Ten MH steps—the
paper default—do not improve the full-test seed despite 3.7× the proposal-token cost of two steps.
"""),
    markdown("## Repeated-sampling scaling and behavior"),
    code("""
display(Image(filename=ROOT / "figures" / "vote_scaling.png", width=720))
display(Image(filename=ROOT / "figures" / "behavior.png", width=1000))
"""),
    code("""
power_gain = summary["paired_differences"]["power_minus_low_temp"]
best = summary["exploratory"]
print("Power − low temp:", f"{100*power_gain['difference']:.2f} pp", power_gain['cluster_bootstrap_95_ci'])
print("Exploratory vote@10:", f"{100*best['vote10_seed0_accuracy']:.2f}%",
      f"pass@10={100*best['vote10_seed0_pass_at_10']:.2f}%")
print("Paper-default 10-step power:", f"{100*best['power10_seed0_accuracy']:.2f}%")
"""),
    markdown(r"""
## Takeaway

Sequence-level power sampling reliably exposes better responses than one low-temperature draw, but it is
not the best compute-aware method here. Six-sample voting reaches **39.98 ± 1.21%**, and exploratory
vote@10 reaches **44.66%** (one seed), whereas paper-default ten-step power reaches **32.52%**.

The result is limited to one small model, strict boxed-answer grading, and GSM8K. Proposal-token counts
also omit power sampling's additional likelihood-scoring passes.
"""),
]

OUTPUT.parent.mkdir(exist_ok=True)
NotebookClient(nb, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
nbf.write(nb, OUTPUT)
print(OUTPUT)
