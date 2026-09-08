from importlib import import_module
import hashlib
import json
import random
from pathlib import Path

import pytest
import numpy as np

from math500_sampling.grading import boxed_answer, grade, prompt
from math500_sampling.paper_grading import paper_grade
from math500_sampling.sampler import Chain, json_rng_state, mh_log_ratio, restore_rng_state


ROOT = Path(__file__).resolve().parents[1] / "data" / "math500"


def test_official_dataset_identity():
    path = ROOT / "MATH500.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "838cd5ffc217ee852f460a5c649ea4825f777e1b99c590b38fc500c6561e1e06"
    assert len(json.loads(path.read_text())) == 500


def test_exact_prompt():
    assert prompt("What is 1+1?") == (
        r"Can you solve the following math problem? What is 1+1? Please reason step by step, "
        r"and put your final answer within \boxed{{}}."
    )


def test_nested_box_and_symbolic_grading():
    assert boxed_answer(r"first \boxed{0}; final \boxed{\frac{1}{\sqrt{2}}}") == r"\frac{1}{\sqrt{2}}"
    assert grade(r"Therefore \boxed{\frac{1}{2}}", "0.5")[0]
    assert grade(r"Therefore \boxed{106^\circ}", r"106^\circ")[0]
    assert not grade("The answer is 0.5", "0.5")[0]


def test_all_targets_self_verify():
    for example in json.loads((ROOT / "MATH500.json").read_text()):
        assert grade(r"\boxed{" + example["answer"] + "}", example["answer"])[0]
        assert paper_grade(example["answer"], example["answer"])


def test_paper_grader_edge_cases():
    assert paper_grade(r"10\%", "10")
    assert paper_grade("{36} miles", "36")
    assert not paper_grade("-2, 1", "1,-2")
    assert not paper_grade(r"\infty", "1")


def test_mh_ratio_identity_and_sharpening():
    proposed = [-0.2, -0.7]
    current = [-0.4, -0.6]
    assert mh_log_ratio(1.0, proposed, current, proposed, current) == pytest.approx(0.0)
    assert mh_log_ratio(4.0, proposed, current, proposed, current) == pytest.approx(
        3 * (sum(proposed) - sum(current))
    )


def test_checkpoint_state_round_trip():
    chain = Chain(prompt_ids=[1, 2], tokens=[3], base_logprobs=[-0.2], attempts=2, accepts=1)
    assert Chain.from_dict(chain.to_dict()) == chain
    rng = random.Random(7)
    state = json.loads(json.dumps(json_rng_state(rng.getstate())))
    expected = [rng.random() for _ in range(4)]
    restored = random.Random()
    restored.setstate(restore_rng_state(state))
    assert [restored.random() for _ in range(4)] == expected




