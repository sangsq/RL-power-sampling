"""Within-question effective power and uncertainty; no correctness inputs."""

import numpy as np


def effective_alpha(base, policy, lengths, group_size, seed=0):
    values = np.column_stack([base, policy, lengths]).reshape(-1, group_size, 3)
    centered = values - values.mean(axis=1, keepdims=True)
    x, y, length = np.moveaxis(centered, -1, 0)
    xx, xy = (x*x).sum(1), (x*y).sum(1)
    xl, ll, ly = (x*length).sum(), (length*length).sum(), (length*y).sum()
    total_xx, total_xy = xx.sum(), xy.sum()
    good = xx > 1e-10
    if total_xx <= 1e-10:
        return {'alpha': None, 'question_slope_std': None, 'identified_questions': 0}
    slope = total_xy / total_xx
    per_question = xy[good] / xx[good]
    rng = np.random.default_rng(seed)
    index = rng.integers(len(xx), size=(1000, len(xx)))
    denominators = xx[index].sum(1)
    bootstrap = xy[index].sum(1)[denominators > 1e-10] / denominators[denominators > 1e-10]
    determinant = total_xx * ll - xl*xl
    adjusted = ((total_xy*ll - ly*xl) / determinant
                if determinant > max(total_xx*ll, 1)*1e-12 else None)
    return {'alpha': float(slope), 'alpha_ci95': np.quantile(bootstrap, [.025, .975]).tolist(),
            'alpha_bootstrap_se': float(bootstrap.std(ddof=1)),
            'question_slope_mean': float(per_question.mean()),
            'question_slope_std': float(per_question.std(ddof=1)) if len(per_question)>1 else None,
            'identified_questions': int(good.sum()), 'alpha_length_adjusted': adjusted,
            'residual_std': float((y-slope*x).std(ddof=1))}


def true_loss(base, policy, alpha, group_size):
    values = np.asarray(policy) - alpha*np.asarray(base)
    questions = values.reshape(-1, group_size).mean(1)
    return {'mean': float(values.mean()), 'response_std': float(values.std(ddof=1)),
            'question_se': float(questions.std(ddof=1)/np.sqrt(len(questions))) if len(questions)>1 else None}
