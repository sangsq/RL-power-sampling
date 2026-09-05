# E002 · MATH500 · Qwen2.5-Math-1.5B

Reproduction of Karan and Du's [*Reasoning with Sampling*](https://arxiv.org/abs/2510.14901),
cross-checked against the [official implementation](https://github.com/aakaran/reasoning-with-sampling/tree/main/llm_experiments).
Keeps the paper's prompt and sampling protocol while reducing the model from 7B
to `Qwen/Qwen2.5-Math-1.5B`.

## Results

| Method | 1.5B accuracy | Paper 7B accuracy | Parse rate | Sampled tokens / answer |
|---|---:|---:|---:|---:|
| Base, T=1 | 46.4% | 49.6% | 84.9% | 664 |
| Low temperature, T=0.25 | 60.2% | 69.0% | 89.1% | 653 |
| Power sampling, alpha=4 | **64.9%** | **74.8%** | 88.9% | 9,861 |

Across two decoding seeds, power improves over low temperature by 4.7 percentage
points (paired question-bootstrap 95% CI: +1.8 to +7.7). This interval does not
fully capture decoding-seed uncertainty; the public protocol uses eight runs.
Power samples about 15.7 times the number of tokens in its final answer. The
comparison matches one returned answer, not sampled-token compute.

[Full report](reports/REPORT.md) · [PDF](reports/results_handoff.pdf) ·
[Notebook](notebooks/results.ipynb) · [Paired examples](reports/QUALITATIVE_EXAMPLES.md)

## Run

Install once using the [repository setup](../../SETUP.md#setup). The following
commands run from this experiment directory with that environment activated:

```bash
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  bash scripts/run_official_protocol.sh 0
python analyze.py
bash scripts/render_handoff.sh
```

The protocol runs base, low temperature, and five power shards. Power uses
alpha=4, proposal T=0.25, 16 blocks of 192 tokens, ten MH updates per block, and
a 3,072-token response limit. `run.py --help` lists individual-run options.
Power checkpoints after every block and resumes interrupted shards.

`PYTHON=/path/to/python` selects a different interpreter for the shell scripts.
Download the model before running the offline protocol; see
[reproducibility](../../SETUP.md#model-weights-and-reproducibility). Generation requires physical
GPU 1 to be the only visible GPU.

`render_handoff.sh` rebuilds analysis, executes the notebook, and exports HTML
on CPU. It also exports PDF when `google-chrome` is available. Outputs go to
`reports/`; the existing PDF can be read without Chrome.

## Files

- `data/`: the fixed 500-question dataset and its provenance.
- `results/`: responses, seed/shard summaries, and resumable checkpoints.
- `notes/`: original plan, research notes, and chronological activity log;
  historical commands use the directory layout at the time they were recorded.
- [`../../src/math500_sampling/`](../../src/math500_sampling/): sampler,
  exact prompt, symbolic grader, and paper-compatible grader.
- [`../../tests/test_math500.py`](../../tests/test_math500.py): CPU tests.
- [Power-policy experiments](../gsm8k_power_policy/README.md): label-free
  LoRA training to approximate the power distribution.

From the repository root, `python scripts/verify_portability.py` checks both
experiments. The notebook runs from the repository root, this directory, or
`notebooks/`.
