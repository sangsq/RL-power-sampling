from importlib import import_module
"""Check the finite-sample pass@N estimator against exhaustive subsets."""

import itertools
import pytest

from power_sampling.evaluation import pass_at_k


def test_pass_at_k_enumerated_subsets():
    for correct in range(7):
        for k in range(1, 7):
            subsets = list(itertools.combinations(range(6), k))
            expected = sum(any(i < correct for i in subset) for subset in subsets) / len(subsets)
            assert pass_at_k(6, correct, k) == pytest.approx(expected)
    assert pass_at_k(32, 0, 32) == 0
    assert pass_at_k(32, 1, 32) == 1
    with pytest.raises(ValueError):
        pass_at_k(32, 1, 33)
