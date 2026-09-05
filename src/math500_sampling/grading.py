"""MATH500 prompt, data loader, and symbolic boxed-answer grading."""

from __future__ import annotations

import json
from pathlib import Path

from math_verify import parse, verify


PREFIX = "Can you solve the following math problem? "
SUFFIX = r" Please reason step by step, and put your final answer within \boxed{{}}."


def load_math500(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def prompt(question: str) -> str:
    """Exact prompt used by the official Qwen2.5-Math runner."""
    return PREFIX + question + SUFFIX


def boxed_answer(text: str) -> str | None:
    """Return the contents of the last nested boxed or fbox expression."""
    start = max(text.rfind(r"\boxed"), text.rfind(r"\fbox"))
    if start < 0:
        return None
    left = text.find("{", start)
    if left < 0:
        return None
    depth = 0
    for index in range(left, len(text)):
        depth += text[index] == "{"
        depth -= text[index] == "}"
        if depth == 0:
            return text[left + 1 : index].strip()
    return None


def _parse_boxed(value: str):
    return parse(r"\boxed{" + value + "}")


def grade(response: str, target: str) -> tuple[bool, str | None]:
    """Symbolically compare the final boxed expression with the target."""
    predicted = boxed_answer(response)
    if predicted is None:
        return False, None
    try:
        return bool(verify(_parse_boxed(target), _parse_boxed(predicted))), predicted
    except Exception:
        return False, predicted
