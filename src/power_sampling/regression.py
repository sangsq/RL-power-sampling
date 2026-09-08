"""Question-centered regression and clustered uncertainty for saved rollouts."""

from collections import defaultdict
import numpy as np

BOOTSTRAPS = 2000


def moments(rows):
    """Remove a separate intercept for every (update, question) group."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["step"], row["dataset_index"])].append(
            [row["base_logprob"], len(row["token_ids"]), row["old_logprob"]])
    output = []
    for (step, question), values in sorted(groups.items()):
        assert len(values) == 4
        a = np.asarray(values, dtype=np.float64)
        assert np.isfinite(a).all()
        a -= a.mean(axis=0)
        x, length, y = a.T
        output.append({"step": step, "question": question,
                       "sums": np.array([x @ x, x @ y, x @ length, length @ length,
                                         length @ y, y @ y])})
    return output


def coefficients(sums):
    """Raw slope and length-adjusted slope from within-group cross products."""
    xx, xy, xl, ll, ly, yy = np.moveaxis(np.asarray(sums), -1, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = xy / xx
        det = xx * ll - xl ** 2
        good = det > np.maximum(xx * ll, 1) * 1e-12
        adjusted = np.where(good, (xy * ll - ly * xl) / det, np.nan)
        length = np.where(good, (ly * xx - xy * xl) / det, np.nan)
        r2 = 1 - np.maximum(0, yy - 2 * raw * xy + raw ** 2 * xx) / yy
        adjusted_r2 = 1 - np.maximum(0, yy - adjusted * xy - length * ly) / yy
    return np.stack([raw, adjusted, length, r2, adjusted_r2], axis=-1)


def estimate(groups, seed):
    """Resample questions; repeated appearances of a question move together."""
    by_question = defaultdict(lambda: np.zeros(6))
    for group in groups:
        by_question[group["question"]] += group["sums"]
    matrix = np.array(list(by_question.values()))
    point = coefficients(matrix.sum(axis=0))
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(matrix), np.full(len(matrix), 1 / len(matrix)), size=BOOTSTRAPS)
    draws = coefficients(weights @ matrix)
    output = {"logged_step_start": min(g["step"] for g in groups),
              "logged_step_end": max(g["step"] for g in groups),
              "policy_update_start": min(g["step"] for g in groups) - 1,
              "policy_update_end": max(g["step"] for g in groups) - 1,
              "question_clusters": len(matrix), "response_groups": len(groups),
              "responses": 4 * len(groups), "within_r2": float(point[3]),
              "length_adjusted_within_r2": float(point[4]), "length_coefficient": float(point[2])}
    for index, label in [(0, "alpha"), (1, "alpha_length_adjusted")]:
        finite = draws[:, index][np.isfinite(draws[:, index])]
        output[label] = float(point[index])
        output[label + "_ci95"] = np.quantile(finite, [.025, .975]).tolist()
        output[label + "_bootstrap_se"] = float(finite.std(ddof=1))
        output[label + "_valid_bootstraps"] = len(finite)
    return output
