"""Grading behavior compatible with the Reasoning with Sampling MATH runner.

This is a compact reimplementation of the normalization/equality rules used by
the paper's public evaluation code.  It is the primary metric for comparisons
to that paper; ``math500_sampling.grading.grade`` remains a sensitivity check.
"""

from __future__ import annotations

import re

import sympy
from pylatexenc import latex2text
from sympy.parsing import sympy_parser


TUPLE_CHARS = "()[]"
UNITS = (
    "degree", "cm", "centimeter", "meter", "mile", "second", "minute", "hour",
    "day", "week", "month", "year", "foot", "feet", "inch", "yard",
)


def _fix_fracs(value: str) -> str:
    pieces = value.split(r"\frac")
    if len(pieces) == 1:
        return value
    result = pieces[0]
    for piece in pieces[1:]:
        result += r"\frac"
        if not piece:
            return value
        if piece[0] == "{":
            result += piece
        elif len(piece) >= 2:
            numerator, denominator, rest = piece[0], piece[1], piece[2:]
            result += "{" + numerator + "}"
            result += denominator if denominator == "{" else "{" + denominator + "}"
            result += rest
        else:
            return value
    return result


def _fix_sqrt(value: str) -> str:
    pieces = value.split(r"\sqrt")
    result = pieces[0]
    for piece in pieces[1:]:
        result += r"\sqrt" + (piece if piece.startswith("{") else "{" + piece[:1] + "}" + piece[1:])
    return result


def _mathd_normalize(value: str | None) -> str | None:
    """Normalization pass inherited by the public runner from the MATH eval."""
    if value is None:
        return None
    try:
        value = value.strip()
        match = re.fullmatch(r"\\text\{(.+?)\}", value)
        if match:
            value = match.group(1).strip()
        value = value.replace("\n", "").replace(r"\!", "").replace(r"\\", "\\")
        value = value.replace("tfrac", "frac").replace("dfrac", "frac")
        value = value.replace(r"\left", "").replace(r"\right", "")
        value = value.replace(r"^{\circ}", "").replace(r"^\circ", "").replace(r"\$", "")
        if r"\text{ " in value:
            parts = value.split(r"\text{ ")
            if len(parts) == 2:
                value = parts[0]
        value = value.replace(r"\%", "")
        value = value.replace(" .", " 0.").replace("{.", "{0.")
        if not value:
            return value
        if value.startswith("."):
            value = "0" + value
        if len(value.split("=")) == 2 and len(value.split("=")[0]) <= 2:
            value = value.split("=")[1]
        value = _fix_sqrt(value).replace(" ", "")
        value = _fix_fracs(value)
        if value == "0.5":
            value = r"\frac{1}{2}"
        pieces = value.split("/")
        if len(pieces) == 2:
            try:
                a, b = map(int, pieces)
                if value == f"{a}/{b}":
                    value = rf"\frac{{{a}}}{{{b}}}"
            except ValueError:
                pass
        return value
    except Exception:
        return value


def _strip_commas(value: str) -> str:
    pattern = re.compile(r"(\d),(\d\d\d)($|\D)")
    while True:
        updated = pattern.sub(r"\1\2\3", value)
        if updated == value:
            return updated
        value = updated


def _is_int(value: str) -> bool:
    try:
        number = float(_strip_commas(value))
        return abs(number - round(number)) <= 1e-7
    except (TypeError, ValueError, OverflowError):
        return False


def _normalize(value: str | None) -> str | None:
    if value is None:
        return None
    match = re.fullmatch(r"\\text\{(.+?)\}", value)
    if match:
        value = match.group(1)
    value = value.replace(r"\%", "%").replace(r"\$", "$").replace("$", "").replace("%", "")
    value = value.replace(" or ", " , ").replace(" and ", " , ")
    value = value.replace("million", "*10^6").replace("billion", "*10^9").replace("trillion", "*10^12")
    for unit in UNITS:
        value = re.sub(rf"{unit}(es)?(s)? *(\^[0-9]+)?", "", value)
    value = re.sub(r"\^ *\\circ", "", value)
    if value.startswith("{") and value.endswith("}"):
        value = value[1:-1]
    value = re.sub(r",\\! *", "", value)
    try:
        number = float(value)
        if abs(number - round(number)) <= 1e-7:
            value = str(round(number))
    except ValueError:
        pass
    if "\\" in value:
        try:
            value = value.replace(r"\tfrac", r"\frac").replace(r"\dfrac", r"\frac")
            value = value.replace(r"\frac", r" \frac")
            value = latex2text.LatexNodes2Text().latex_to_text(value)
            translations = {"√": "sqrt", "π": "pi", "∞": "inf", "∪": "U", "·": "*", "×": "*"}
            value = "".join(translations.get(character, character) for character in value).strip()
        except Exception:
            pass
    value = re.sub(r"- *", "-", value)
    value = re.sub(r"([0-9]) +([0-9])", r"\1+\2", value).replace(" ", "")
    value = value.replace("{", "").replace("}", "").lower()
    if _is_int(value):
        value = str(int(float(value.replace(",", ""))))
    return value


def _split(value: str) -> list[str]:
    value = _strip_commas(value)
    if not value:
        return []
    if len(value) > 2 and value[0] in TUPLE_CHARS and value[-1] in TUPLE_CHARS:
        if all(character not in value[1:-1] for character in TUPLE_CHARS):
            return [part.strip() for part in value[1:-1].split(",")]
    return [value]


def _sympy_equal(left: str, right: str) -> bool:
    expression = f"({left})-({right})"
    letters = set(expression.replace("sqrt", "").replace("frac", "")) & set("abcdefghijklmnopqrstuvwxyz")
    if len(letters) > 2 or "^{" in expression or "^(" in expression:
        return False
    if re.search(r"\^[0-9]+\^|\^[0-9][0-9]+", expression):
        return False
    try:
        parsed = sympy_parser.parse_expr(
            expression.replace("^", "**"),
            transformations=sympy_parser.standard_transformations
            + (sympy_parser.implicit_multiplication_application,),
        )
        return sympy.simplify(parsed) == 0
    except Exception:
        return False


def paper_grade(predicted: str | None, target: str) -> bool:
    """Return the exact-style answer correctness used for paper comparison."""
    if predicted is None:
        return False
    if _mathd_normalize(predicted) == _mathd_normalize(target):
        return True
    predicted_normal, target_normal = _normalize(predicted), _normalize(target)
    if target_normal is None:
        return False
    if predicted_normal == target_normal:
        return True
    if not predicted_normal:
        return False
    predicted_parts, target_parts = _split(predicted_normal), _split(target_normal)
    if len(target_parts) > 1 and (
        target_normal[0] != predicted_normal[0] or target_normal[-1] != predicted_normal[-1]
    ):
        return False
    if len(predicted_parts) != len(target_parts):
        return False
    for target_part, predicted_part in zip(target_parts, predicted_parts, strict=True):
        both_fractions = all(re.fullmatch(r"-?[0-9]+.?/0*[1-9][0-9]*.?", part) for part in (target_part, predicted_part))
        if both_fractions:
            equal = target_part == predicted_part
        elif _is_int(target_part) != _is_int(predicted_part):
            equal = False
        else:
            equal = _sympy_equal(target_part, predicted_part)
        if not equal:
            return False
    return True
