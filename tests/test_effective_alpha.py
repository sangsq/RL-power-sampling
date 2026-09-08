from importlib import import_module
"""Known slopes, normalization offsets and length-confounding diagnostics."""

import numpy as np

coefficients = import_module("power_sampling.regression").coefficients

estimate = import_module("power_sampling.regression").estimate

moments = import_module("power_sampling.regression").moments


def test_recovers_power_with_question_intercepts():
    rows = []
    for q in range(8):
        for j in range(4):
            base = -10 - q - j * (q + 1)
            rows.append({"step": 1, "dataset_index": q, "base_logprob": base,
                         "old_logprob": 4 * base + q * 100,
                         "token_ids": [1] * (1 + j % 2)})
    result = estimate(moments(rows), 1)
    assert np.isclose(result["alpha"], 4)
    assert np.isclose(result["alpha_length_adjusted"], 4)
    np.testing.assert_allclose(result["alpha_ci95"], [4, 4])


def test_length_adjustment_recovers_known_partial_slope():
    rows = []
    for q in range(8):
        for j in range(4):
            length = 5 + j
            base = -2 * length - (j % 2) - q
            rows.append({"step": 1, "dataset_index": q, "base_logprob": base,
                         "old_logprob": 2 * base + 3 * length - q,
                         "token_ids": [1] * length})
    result = estimate(moments(rows), 2)
    assert abs(result["alpha"] - 2) > .5
    assert np.isclose(result["alpha_length_adjusted"], 2)
    assert np.isclose(result["length_coefficient"], 3)


def test_singular_length_control_is_not_reported_as_identified():
    result = coefficients(np.array([4, 8, 4, 4, 8, 16]))
    assert result[0] == 2
    assert np.isnan(result[1])
