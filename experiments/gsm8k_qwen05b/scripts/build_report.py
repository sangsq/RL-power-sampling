"""Build a readable PDF handoff using only matplotlib."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]


def page(pdf: PdfPages, title: str, paragraphs: list[str], *, footer: str = "") -> None:
    fig = plt.figure(figsize=(8.5, 11), facecolor="white")
    title_lines = textwrap.wrap(title, width=44, break_long_words=False)
    fig.text(0.08, 0.94, "\n".join(title_lines), fontsize=19, weight="bold", color="#243447", va="top")
    y = 0.89 - 0.045 * (len(title_lines) - 1)
    for paragraph in paragraphs:
        if paragraph.startswith("## "):
            y -= 0.015
            fig.text(0.08, y, paragraph[3:], fontsize=13, weight="bold", color="#F58518")
            y -= 0.042
            continue
        lines = textwrap.wrap(paragraph, width=88, break_long_words=False) or [""]
        fig.text(0.08, y, "\n".join(lines), fontsize=10.5, linespacing=1.45, va="top", color="#263238")
        y -= 0.027 * len(lines) + 0.022
    if footer:
        fig.text(0.08, 0.035, footer, fontsize=8, color="#607D8B")
    pdf.savefig(fig)
    plt.close(fig)


def image_page(pdf: PdfPages, title: str, image: str, caption: str) -> None:
    fig = plt.figure(figsize=(8.5, 11), facecolor="white")
    fig.text(0.08, 0.94, title, fontsize=20, weight="bold", color="#243447")
    axis = fig.add_axes([0.06, 0.20, 0.88, 0.68])
    axis.imshow(mpimg.imread(ROOT / image))
    axis.axis("off")
    fig.text(0.08, 0.14, "\n".join(textwrap.wrap(caption, 108)), fontsize=10.5, linespacing=1.4, va="top")
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    summary = json.loads((ROOT / "results" / "summary.json").read_text())
    (ROOT / "reports").mkdir(exist_ok=True)
    with PdfPages(ROOT / "reports" / "report.pdf") as pdf:
        page(
            pdf,
            "Reasoning with Sampling — Small-Scale Reproduction",
            [
                "Qwen2.5-0.5B · GSM8K · raw question-only prompting · vLLM on physical GPU 1",
                "30 August 2026",
                "## Executive finding",
                "Sequence-level power sampling improved full-test accuracy from 29.09% for one selected low-temperature sample to 32.95%. The paired improvement was 3.87 percentage points (question-clustered 95% bootstrap interval: 2.40 to 5.31).",
                "The result is positive but not dominant. Six low-temperature samples followed by numeric majority voting reached 39.98% at essentially the same proposal-generated-token budget. In this 0.5B setting, inference-time compute helps, but voting spends it more effectively than MCMC power sampling.",
                "A final single-seed exploration reached 44.66% with ten-sample voting. Paper-default ten-step power reached 32.52% while using more than twice as many proposal tokens.",
                "## Scope",
                "This is an algorithmic reproduction of Karan and Du (2025), not a direct numerical replication. It deliberately uses a small general model, two MCMC steps per block, and one benchmark so the full experiment fits on one RTX 4090.",
            ],
            footer="Paper: arxiv.org/abs/2510.14901 · Code: github.com/aakaran/reasoning-with-sampling",
        )
        page(
            pdf,
            "Method and evaluation protocol",
            [
                "## Target distribution",
                "The sampler targets a sequence distribution proportional to p(x)^alpha. This is different from lowering token temperature because future-path normalizers do not cancel at sequence level.",
                "## Batched autoregressive MCMC",
                "Responses are extended in blocks. Each update chooses a random response-token cut, redraws the complete suffix from a low-temperature proposal q, and accepts it with log r = alpha[log p(x') - log p(x)] + log q(x) - log q(x'). vLLM batches both proposals and full-response likelihood scoring.",
                "## Main configuration",
                "Alpha 4; proposal temperature 0.25; 32-token blocks; two MH updates per block; 512 response tokens. The 512-token budget yields 16 blocks, matching the paper's block count while reducing its ten MCMC updates.",
                "## Data and grading",
                "Development uses the first 256 GSM8K train questions. Final evaluation uses all 1,319 test questions and three seeds. Each example has an independent deterministic RNG stream. The prompt is exactly the question string, with no chat template, chain-of-thought cue, or boxed-answer instruction. The strict grader requires the last boxed value to match GSM8K's final numeric answer.",
            ],
        )
        image_page(
            pdf,
            "Full-test results",
            "figures/main_results.png",
            "Bars show mean ± sample SD across three seeds. Proposal-token counts omit power sampling's additional likelihood-scoring passes, so the six-sample vote is cheaper than the nominal token match suggests.",
        )
        page(
            pdf,
            "Numerical results",
            [
                "## Three-seed full test",
                "Temperature 1: 11.37%, 12.36%, 12.36% — mean 12.03 ± 0.57%; 272 proposal tokens/question.",
                "Temperature 0.2: 30.63%, 28.73%, 27.90% — mean 29.09 ± 1.40%; 275 proposal tokens/question.",
                "Power, alpha 4: 32.60%, 32.83%, 33.43% — mean 32.95 ± 0.43%; 1,636 proposal tokens/question; approximately 65.3% MH acceptance.",
                "Six-sample vote: 38.74%, 41.17%, 40.03% — mean 39.98 ± 1.21%; 1,638 proposal tokens/question.",
                "## Paired uncertainty",
                f"Power minus low temperature: {summary['paired_differences']['power_minus_low_temp']['difference'] * 100:.2f} points; 95% cluster-bootstrap interval [2.40, 5.31].",
                f"Six-sample vote minus power: {summary['paired_differences']['vote_minus_power']['difference'] * 100:.2f} points; interval [5.48, 8.59].",
                "The bootstrap averages each question's three seed outcomes, then resamples questions. It therefore preserves repeated outcomes for the same problem rather than treating 3,957 generations as independent data.",
                "Exploratory seed 0: vote@10 reached 44.66% at 2,727 proposal tokens/question; ten-step power reached 32.52% at 6,119 tokens. These points are excluded from the formal three-seed uncertainty estimates.",
            ],
        )
        image_page(
            pdf,
            "Development ablations",
            "figures/ablations.png",
            "Ablations use 256 train questions and one seed. Alpha 4–8, two MH updates, block size 32, and a 512-token response budget form the robust region. These small single-seed differences should not be interpreted as precise hyperparameter rankings.",
        )
        image_page(
            pdf,
            "Repeated-sampling scaling",
            "figures/vote_scaling.png",
            "Plurality voting rises monotonically through six samples. The pass@6 ceiling is 53.68%, leaving a 13.70-point aggregation gap. Stars show the exploratory seed-0 vote@10 result (44.66%) and pass@10 ceiling (59.59%).",
        )
        image_page(
            pdf,
            "Behavior diagnostics",
            "figures/behavior.png",
            "Power changes output length but only slightly changes parse rate relative to low temperature. Correct and incorrect power responses have almost identical mean MH acceptance, so acceptance is not a response-quality score.",
        )
        page(
            pdf,
            "Reliability, limitations, and conclusion",
            [
                "## Reliability checks",
                "Unit tests cover grading, MH identity algebra, and unequal suffix lengths. The official code truncates the current suffix when a candidate ends early; the paper's ratio requires complete suffixes. The main implementation follows the equation and exposes the official behavior only as a labeled compatibility ablation (37.11% versus 37.50% on development).",
                "BF16 generation and prompt-scoring log probabilities in vLLM can accumulate small differences. The exact alpha=temperature=1 identity is enforced analytically; other conditions retain the measured correction and may contain small numerical bias.",
                "A shared batch seed creates correlated vLLM random streams and produced unstable test accuracy from 9.86% to 19.86%. Formal runs derive an independent stream per question. This explains the earlier approximately 20% one-seed baseline without changing the grader.",
                "## Limitations",
                "One 0.5B model, one benchmark, strict boxed-answer parsing, a 256-question development set, and two MCMC steps cannot establish general behavior for larger math models. Alpha 8 also received a single test run, so that comparison is exploratory. Another GPU1 process affected wall time; proposal-token counts are more reproducible.",
                "## Conclusion",
                "Power sampling reliably improved a frozen Qwen2.5-0.5B model over one low-temperature response and substantially over temperature 1. It did not beat self-consistency at a matched proposal-token budget. The most useful lesson is methodological: sequence-level decoding can expose better responses, but it must be evaluated against strong, compute-aware repeated-sampling baselines.",
            ],
            footer="Complete outputs, code, tests, figures, research notes, and paired examples are included in this folder.",
        )
    print(ROOT / "reports" / "report.pdf")


if __name__ == "__main__":
    main()
