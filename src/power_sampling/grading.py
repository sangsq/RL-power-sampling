"""Minimal GSM8K loader and question-only boxed-answer grader."""

from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def ground_truth(example: dict) -> str:
    return example["answer"].rsplit("####", 1)[-1].strip()


def last_boxed(text: str) -> str | None:
    start = text.rfind("\\boxed")
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


def number(text: str) -> Decimal | None:
    cleaned = text.strip().replace(",", "").replace("$", "")
    cleaned = re.sub(r"\\(?:text|mathrm)\{([^{}]*)\}", r"\1", cleaned)
    cleaned = cleaned.strip().rstrip(". %")
    try:
        if "/" in cleaned and re.fullmatch(r"[-+]?\d+(?:\.\d+)?/\d+(?:\.\d+)?", cleaned):
            numerator, denominator = cleaned.split("/")
            return Decimal(numerator) / Decimal(denominator)
        return Decimal(cleaned)
    except (InvalidOperation, ZeroDivisionError):
        return None


def grade(response: str, answer: str) -> tuple[bool, str | None]:
    parsed = last_boxed(response)
    if parsed is None:
        return False, None
    predicted, target = number(parsed), number(answer)
    if predicted is not None and target is not None:
        return abs(predicted - target) <= Decimal("1e-9"), parsed
    return parsed.replace(" ", "") == answer.replace(" ", ""), parsed


def canonical_answer(response: str) -> str | None:
    parsed = last_boxed(response)
    if parsed is None:
        return None
    value = number(parsed)
    return str(value.normalize()) if value is not None else parsed.replace(" ", "")
