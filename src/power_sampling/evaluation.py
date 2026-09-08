"""Shared GSM8K answer extraction, grading sensitivities, and pass@k estimation."""

import math
import re

from power_sampling.grading import grade, last_boxed, number


def extract_explicit_answer(response):
    """Post-hoc format sensitivity; do not infer an answer from arbitrary numbers."""
    boxed = last_boxed(response)
    if boxed is not None:
        return boxed
    if "####" in response:
        tail = response.rsplit("####", 1)[1]
    else:
        matches = list(re.finditer(r"\b(?:the\s+)?(?:final\s+)?answer\s*(?:is\b|:|=)", response, re.I))
        if not matches:
            return None
        tail = response[matches[-1].end():]
    match = re.match(r"\s*[:=$*\s]*([-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:/\d+(?:\.\d+)?)?)", tail)
    return match.group(1).rstrip(",") if match else None



def answer_scores(text, target):
    boxed = grade(text, target)[0]
    parsed = extract_explicit_answer(text)
    # The paper prompt contains literal double braces; accept nested answer braces.
    if parsed is not None:
        while parsed.startswith("{") and parsed.endswith("}"):
            parsed = parsed[1:-1].strip()
    correct = parsed is not None and grade("\\boxed{" + parsed + "}", target)[0]
    return {"boxed_correct": boxed, "answer_correct": correct, "parsed_answer": parsed}



def scores(text, target):
    base = answer_scores(text, target)
    normalized = re.sub(r"\\box(?=\s*\{)", lambda _: r"\boxed", text)
    conclusion = answer_scores(normalized, target)
    parsed = conclusion["parsed_answer"]
    if parsed is None:
        markers = list(re.finditer(r"\bfinal\s+(?:answer|result)\b\s*(?:is\b|:|=)?", normalized, re.I))
        if markers:
            line = normalized[markers[-1].end():].lstrip(" :=$*\t").split("\n", 1)[0]
            values = re.findall(r"[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:/\d+(?:\.\d+)?)?", line)
            unique = {number(x) for x in values}
            if values and len(unique) == 1 and None not in unique:
                parsed = values[-1].rstrip(",")
    conclusion_ok = parsed is not None and grade(r"\boxed{" + parsed + "}", target)[0]
    return {"correct": {"answer": base["answer_correct"], "boxed": base["boxed_correct"],
                        "conclusion": conclusion_ok},
            "parsed_answer": base["parsed_answer"], "conclusion_answer": parsed}



def pass_at_k(n, correct, k):
    if not 1 <= k <= n or not 0 <= correct <= n:
        raise ValueError("Require 1 <= k <= n and 0 <= correct <= n")
    return 1.0 if n - correct < k else 1 - math.comb(n - correct, k) / math.comb(n, k)

