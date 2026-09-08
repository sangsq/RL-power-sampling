"""Compact saved evaluation evidence without changing grading or token counts."""

import json
from pathlib import Path

import numpy as np
from power_sampling.evaluation import pass_at_k


def compact_question(row):
    samples = row['samples']
    return {
        'index': row['index'],
        'samples': len(samples),
        'correct': {key: sum(s['correct'][key] for s in samples)
                    for key in ('answer', 'boxed', 'conclusion')},
        'lengths': [len(s['token_ids']) for s in samples],
        'generated_tokens': sum(s.get('generated_tokens', len(s['token_ids'])) for s in samples),
    }


def compact_file(source, destination):
    """Preserve question order; full responses remain in the source file."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp')
    with Path(source).open() as src, temporary.open('w') as dst:
        for line in src:
            dst.write(json.dumps(compact_question(json.loads(line))) + '\n')
    temporary.replace(destination)


def evaluate_rows(raw):
    assert raw and all(r['samples'] == 32 and len(r['lengths']) == 32 for r in raw)
    vectors = {
        grading: {
            str(k): np.array([pass_at_k(32, r['correct'][grading], k)
                              for r in raw])
            for k in [1, 2, 4, 8, 16, 32]
        } for grading in ['answer', 'boxed', 'conclusion']
    }
    lengths = [length for r in raw for length in r['lengths']]
    result = {'pass_at_k': {g: {k: float(v.mean()) for k, v in ks.items()}
                            for g, ks in vectors.items()},
              'mean_length': float(np.mean(lengths)), 'median_length': float(np.median(lengths)),
              'output_tokens': sum(lengths),
              'generated_tokens': sum(r['generated_tokens'] for r in raw)}
    return result, vectors

