"""Numeric GSM8K scoring for the R1 prompt's closed answer tags."""

import re

from power_sampling.grading import grade, last_boxed


def r1_grade(response, target):
    """Score the first closed answer, separately tracking the requested tag format."""
    match = re.search(r"<answer>(.*?)</answer>", response, re.S)
    if match is None:
        return {"format_valid": False, "answer": None, "answer_correct": False, "correct": False}
    answer = match.group(1).strip()
    boxed = last_boxed(answer)
    parsed = boxed if boxed is not None else answer
    correct = grade(r"\boxed{" + parsed + "}", target)[0]
    # The opening think tag is already supplied by the prompt.
    formatted = response[:match.start()].endswith("</think> ")
    return {"format_valid": formatted, "answer": parsed, "answer_correct": correct,
            "correct": formatted and correct}
