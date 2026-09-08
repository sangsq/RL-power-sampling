"""Summarize the saved baseline evidence and build its protocol notebook."""

import json
from pathlib import Path
import nbformat
from power_sampling.evidence import evaluate_rows

HERE = Path(__file__).resolve().parents[1]


def main():
    folder = HERE/'results/evaluation'
    protocol = json.loads((folder/'protocol.json').read_text())
    results = {}
    for name in ['base_t1', 'base_t0.25', 'policy_alpha2', 'mcmc_alpha2']:
        rows = [json.loads(line) for line in (folder/'questions'/f'{name}.jsonl').open()]
        assert [r['index'] for r in rows] == protocol['test_indices']
        assert len(rows) == 500
        results[name], _ = evaluate_rows(rows)
    (folder/'summary.json').write_text(json.dumps(results, indent=2)+'\n')
    training = json.loads((HERE/'results/fixed512_alpha2/training_summary.json').read_text())
    selection = json.loads((folder/'temperature_selection.json').read_text())
    cells = [
        '# GSM8K power-policy baselines\n\n'
        'This experiment supplies base sampling, development-selected low temperature, fixed-length RLOO '
        'and MCMC for the [adaptive-length study](../rloo_adapt_len/report.ipynb). '
        'The combined comparison table and uncertainty estimates appear there once.',
        '## Protocol\n\n'
        'Qwen2.5-0.5B, revision `060db6499f32faf8b98477b0a26969ef7d8b9987`, '
        'on the same 500 GSM8K test questions, 32 responses per question and a 512-token cap. '
        'The exact prompt is:\n\n```text\n'+protocol['prompt_template']+'\n```\n\n'
        f"Low T={selection['selected_temperature']} was chosen on separate development questions by mean answer "
        'pass@1/2/4/8. [Temperature selection](results/evaluation/temperature_selection.json) records the complete grid. '
        'The original [protocol](results/evaluation/protocol.json) is retained verbatim for provenance; '
        'it also records historical conditions outside this curated comparison.\n\n'
        'MCMC uses target alpha 2, proposal T=0.5, block size 32 and two MH updates per block, '
        'with full-suffix acceptance and EOS retained. This is a finite-compute implementation, '
        'not evidence of convergence to the exact power distribution. '
        'All test conditions use the same model revision, prompt and question order.',
        '## Fixed-length training and limitations\n\n'
        'The historical alpha-2 run uses 1,024 GSM8K training questions, seed 0, all-linear LoRA rank 8 '
        'with LoRA scaling alpha 16, learning rate 1e-5 and a fixed 512-token cap. '
        f"It completed {training['completed_steps']} updates in {training['wall_seconds']/60:.1f} minutes. "
        'The objective and leave-one-out baseline are shared with adaptive RLOO; labels enter only monitoring. '
        '[Training config](results/fixed512_alpha2/config.json) and '
        '[metrics](results/fixed512_alpha2/metrics.jsonl) preserve the executed settings.\n\n'
        'Fixed-length RLOO improves answer pass@1 over base T=1 but performs below low temperature '
        'and largely loses the requested boxed format. The adaptive comparison also changes update count '
        'and diagnostic overhead, so it does not isolate response-length scheduling alone.',
        '## Reproduce the analysis\n\n'
        'From the repository root, after installation:\n\n```bash\n'
        'python experiments/gsm8k_power_policy/scripts/analyze.py\n'
        'python experiments/rloo_adapt_len/scripts/analyze.py\n```\n\n'
        '[Compact measurements](results/evaluation/summary.json) are recomputed from per-question correctness '
        'counts and token lengths. [Source hashes](results/provenance/manifest.json) link them to the original '
        'raw outputs, which remain local. Compact data supports pass@N, paired-question uncertainty and token '
        'cost analysis; it does not permit regrading response text. No model or GPU is needed to rebuild reports. '
        'See [SETUP](../../SETUP.md) for fresh-run entry points.'
    ]
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(s) for s in cells])
    nbformat.validate(notebook)
    nbformat.write(notebook, HERE/'report.ipynb')


if __name__ == '__main__':
    main()
