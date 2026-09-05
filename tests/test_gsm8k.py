from decimal import Decimal

import pytest

from power_sampling.grading import canonical_answer, grade, last_boxed, number
from power_sampling.sampler import mh_log_ratio


def test_boxed_gsm8k_grading():
    assert last_boxed(r"work... \\boxed{1,234}") == "1,234"
    assert number("$1,234.0") == Decimal("1234.0")
    assert grade(r"work... \\boxed{1,234}", "1234")[0]
    assert not grade("the answer is 1234", "1234")[0]
    assert canonical_answer(r"work... \\boxed{1,234.0}") == "1234"


def test_mh_ratio_is_one_when_target_equals_proposal():
    proposed = [-0.2, -0.7]
    current = [-0.4, -0.6]
    assert mh_log_ratio(1.0, proposed, current, proposed, current) == pytest.approx(0.0)


def test_mh_ratio_with_base_proposal_reduces_to_sharpening():
    proposed = [-0.2, -0.3]
    current = [-0.4, -0.5]
    assert mh_log_ratio(4.0, proposed, current, proposed, current) == pytest.approx(
        3 * (sum(proposed) - sum(current))
    )


def test_identity_kernel_handles_different_suffix_lengths():
    proposed = [-0.2]
    current = [-0.1, -0.3, -0.4]
    assert mh_log_ratio(1.0, proposed, current, proposed, current) == pytest.approx(0.0)
